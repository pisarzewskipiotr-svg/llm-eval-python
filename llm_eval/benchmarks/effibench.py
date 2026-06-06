"""
EffiBench benchmark loader and evaluator.

Adapts EffiBench (Huang et al., NeurIPS 2024) for use in the LLM-eval pipeline.
EffiBench provides 1000 LeetCode-style Python tasks with canonical (SOTA) solutions.

This module:
- Loads the stratified 50-task sample
- Provides task descriptions for prompting
- Validates generated code (canonical tests + small_test_cases)
- Measures execution time (warm-up + N runs + median)
- Computes eta_efficiency (canonical_time / generated_time)

"""

from __future__ import annotations

import json
import logging
import os
import statistics
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# === Konfiguracja ścieżek ===
DATASET_DIR = Path(r'J:\Praca wyniki\external\EffiBench')
DATASET_PATH = DATASET_DIR / 'dataset.jsonl'
SAMPLE_PATH = DATASET_DIR / 'sample_50.json'
CANONICAL_TIMES_PATH = DATASET_DIR / 'canonical_times.json'

# === Prefix wstrzykiwany do każdego wykonania ===
EFFIBENCH_PREFIX = """
from typing import List, Optional, Dict, Tuple, Set, Any, Union, Callable
from collections import defaultdict, Counter, deque, OrderedDict
from functools import lru_cache, reduce, cmp_to_key, cache
from itertools import combinations, permutations, product, chain, accumulate, pairwise
from math import inf, gcd, lcm, sqrt, floor, ceil, log, log2
import math
import heapq
import bisect
import re
import sys
import string

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Node:
    def __init__(self, val=None, children=None, next=None, left=None, right=None):
        self.val = val
        self.children = children
        self.next = next
        self.left = left
        self.right = right

def create_tree(values):
    if not values:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root

def decode_tree(tree):
    return tree

def encode_tree(tree):
    return tree

def create_linked_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    cur = head
    for v in values[1:]:
        cur.next = ListNode(v)
        cur = cur.next
    return head
"""


# === Data classes ===

@dataclass
class EffiBenchTask:
    """Pojedyncze zadanie z EffiBench."""
    problem_idx: int
    task_name: str
    description: str               # plain HTML description z LeetCode
    markdown_description: str      # markdown wersja (preferowana dla LLM)
    canonical_solution: str        # SOTA z LeetCode leaderboard
    test_case: str                 # pełne testy (100+ asercji)
    small_test_cases: str          # 3-5 asercji dla quick-check
    test_case_generator: str       # kod generujący nowe testy
    
    # Metadane z naszej stratyfikacji
    canonical_time_ms: Optional[float] = None
    time_bucket: Optional[str] = None      # FAST/MEDIUM/SLOW
    keyword_category: Optional[str] = None # array_sort/string/tree_graph/dp_recursion/other


@dataclass
class ExecutionResult:
    """Wynik wykonania kodu (canonical lub generated)."""
    success: bool
    median_ms: Optional[float] = None
    std_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    n_warmup: int = 0
    n_measurements: int = 0
    error_type: Optional[str] = None       # 'SYNTAX_ERROR' | 'ASSERTION_FAIL' | 'TIMEOUT' | 'RUNTIME_ERROR'
    error_message: Optional[str] = None
    raw_stderr: Optional[str] = None


@dataclass
class ValidationResult:
    """Wynik walidacji semantycznej."""
    pytest_canonical_pass: bool = False
    pytest_small_pass: bool = False
    property_test_pass: Optional[bool] = None  # None = not run
    n_property_examples: int = 0
    functional_status: str = "UNKNOWN"  # SUCCESS | LOGICAL_REGRESSION | SYNTAX_ERROR | TIMEOUT
    error_details: Optional[str] = None


# === Loader ===

def load_sample_50() -> list[EffiBenchTask]:
    """Załaduj stratyfikowaną próbkę 50 zadań z metadanymi."""
    with open(SAMPLE_PATH) as f:
        sample_meta = json.load(f)
    
    with open(DATASET_PATH) as f:
        all_tasks = {json.loads(line)['problem_idx']: json.loads(line) 
                     for line in f}
    
    tasks = []
    for meta in sample_meta:
        idx = meta['problem_idx']
        raw = all_tasks[idx]
        task = EffiBenchTask(
            problem_idx=idx,
            task_name=raw['task_name'],
            description=raw['description'],
            markdown_description=raw['markdown_description'],
            canonical_solution=raw['canonical_solution'],
            test_case=raw['test_case'],
            small_test_cases=raw['small_test_cases'],
            test_case_generator=raw['test_case_generator'],
            canonical_time_ms=meta['canonical_time_ms'],
            time_bucket=meta['time_bucket'],
            keyword_category=meta['keyword_category'],
        )
        tasks.append(task)
    
    logger.info(f"Loaded {len(tasks)} tasks from EffiBench sample")
    return tasks


# === Walidacja semantyczna ===

def validate_code(
    code: str, 
    task: EffiBenchTask, 
    use_small_tests: bool = True,
    timeout: int = 30
) -> ValidationResult:
    """
    Sprawdza czy kod przechodzi testy zadania.
    
    Args:
        code: wygenerowany kod (powinien zawierać class Solution)
        task: zadanie EffiBench
        use_small_tests: True = szybkie 3-5 asercji, False = pełne 100+
        timeout: sekundy
    
    Returns:
        ValidationResult z statusem i detalami błędów
    """
    tests = task.small_test_cases if use_small_tests else task.test_case
    
    # Wstrzykuj solution = Solution() jeśli small_test_cases — bo zawierają już
    # `solution = Solution()` na początku. Test_case nie zawiera tego.
    if use_small_tests:
        full_code = f"{EFFIBENCH_PREFIX}\n\n{code}\n\n{tests}"
    else:
        full_code = f"{EFFIBENCH_PREFIX}\n\n{code}\n\nsolution = Solution()\n\n{tests}"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(full_code)
        f.flush()
        path = f.name
    
    try:
        result = subprocess.run(
            ['python', path],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        
        if result.returncode == 0:
            status = "SUCCESS"
            return ValidationResult(
                pytest_canonical_pass=(not use_small_tests),
                pytest_small_pass=use_small_tests,
                functional_status=status,
            )
        
        # Klasyfikuj błąd
        stderr = result.stderr.strip()
        last_line = stderr.split('\n')[-1] if stderr else "UNKNOWN"
        
        if 'SyntaxError' in last_line:
            status = "SYNTAX_ERROR"
        elif 'AssertionError' in last_line:
            status = "LOGICAL_REGRESSION"
        else:
            status = "LOGICAL_REGRESSION"  # RuntimeError, IndexError, etc. też regresja
        
        return ValidationResult(
            functional_status=status,
            error_details=last_line[:500]
        )
    
    except subprocess.TimeoutExpired:
        return ValidationResult(
            functional_status="TIMEOUT",
            error_details=f"Exceeded {timeout}s"
        )
    finally:
        os.unlink(path)


# === Pomiar wydajności ===

def measure_execution_time(
    code: str,
    task: EffiBenchTask,
    n_warmup: int = 5,
    n_runs: int = 15,
    timeout: int = 120,
    use_small_tests: bool = True
) -> ExecutionResult:
    """
    Mierzy czas wykonania kodu metodą INTERNAL (pomiar wewnątrz subprocess).
    
    Pipeline:
    1. Warm-up: n_warmup wykonań bez pomiaru
    2. Pomiary: n_runs wykonań z time.perf_counter
    3. Statystyki: mediana, std, min, max
    
    Args:
        code: wygenerowany kod z class Solution
        task: zadanie
        n_warmup: ile rozgrzewek (default 5)
        n_runs: ile pomiarów (default 15)
        timeout: max czas subprocess (default 120s)
        use_small_tests: szybkie testy (rekomendowane dla pomiarów)
    
    Returns:
        ExecutionResult z medianą / statystykami lub error
    """
    tests = task.small_test_cases if use_small_tests else task.test_case
    
    # Wcięcie testów do try/except
    tests_indented = textwrap.indent(tests, ' ' * 8)
    
    measurement_script = f'''{EFFIBENCH_PREFIX}

{code}

# Warm-up — bez pomiaru
for _ in range({n_warmup}):
    try:
{tests_indented}
    except (AssertionError, Exception):
        pass

# Pomiary
import time as _time_module
import statistics as _stats
_times = []
for _ in range({n_runs}):
    _t0 = _time_module.perf_counter()
    try:
{tests_indented}
    except (AssertionError, Exception):
        pass
    _t1 = _time_module.perf_counter()
    _times.append((_t1 - _t0) * 1000)

print(f"{{_stats.median(_times):.6f}}")
print(f"{{_stats.stdev(_times) if len(_times) > 1 else 0:.6f}}")
print(f"{{min(_times):.6f}}")
print(f"{{max(_times):.6f}}")
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(measurement_script)
        f.flush()
        path = f.name
    
    try:
        result = subprocess.run(
            ['python', path],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        
        if result.returncode != 0:
            stderr = result.stderr.strip()
            last_line = stderr.split('\n')[-1] if stderr else "UNKNOWN"
            
            if 'SyntaxError' in last_line:
                err_type = 'SYNTAX_ERROR'
            else:
                err_type = 'RUNTIME_ERROR'
            
            return ExecutionResult(
                success=False,
                error_type=err_type,
                error_message=last_line[:300],
                raw_stderr=stderr[:1000],
                n_warmup=n_warmup,
                n_measurements=n_runs
            )
        
        lines = result.stdout.strip().split('\n')
        if len(lines) < 4:
            return ExecutionResult(
                success=False,
                error_type='PARSE_ERROR',
                error_message=f"Expected 4 lines, got {len(lines)}: {result.stdout[:200]}",
                n_warmup=n_warmup,
                n_measurements=n_runs
            )
        
        return ExecutionResult(
            success=True,
            median_ms=float(lines[0]),
            std_ms=float(lines[1]),
            min_ms=float(lines[2]),
            max_ms=float(lines[3]),
            n_warmup=n_warmup,
            n_measurements=n_runs
        )
    
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            error_type='TIMEOUT',
            error_message=f"Exceeded {timeout}s",
            n_warmup=n_warmup,
            n_measurements=n_runs
        )
    finally:
        os.unlink(path)


# === Wrapper integracyjny ===

@dataclass
class EvaluationResult:
    """Pełen wynik ewaluacji wygenerowanego kodu."""
    validation: ValidationResult
    measurement: ExecutionResult
    canonical_measurement: Optional[ExecutionResult] = None
    eta_efficiency: Optional[float] = None  # canonical_time / generated_time
    speedup_vs_canonical: Optional[float] = None
    
    def summary(self) -> str:
        if not self.validation.functional_status == "SUCCESS":
            return f"FAILED ({self.validation.functional_status})"
        if not self.measurement.success:
            return f"PASSED but couldn't measure ({self.measurement.error_type})"
        return (f"SUCCESS: {self.measurement.median_ms:.3f}ms "
                f"(canonical {self.canonical_measurement.median_ms:.3f}ms, "
                f"eta={self.eta_efficiency:.3f})")


def evaluate_generated_code(
    code: str,
    task: EffiBenchTask,
    n_warmup: int = 5,
    n_runs: int = 15
) -> EvaluationResult:
    """
    Pełna ewaluacja: walidacja semantyczna + pomiar wydajności + porównanie z canonical.
    
    Pipeline:
    1. Validate (small_test_cases) — czy kod poprawny?
    2. Jeśli SUCCESS: zmierz czas wykonania
    3. Zmierz czas canonical_solution
    4. Oblicz eta = canonical / generated
    
    Args:
        code: wygenerowany kod
        task: zadanie
        n_warmup: rozgrzewki
        n_runs: pomiary
    
    Returns:
        EvaluationResult
    """
    # Krok 1: Walidacja
    validation = validate_code(code, task, use_small_tests=True)
    
    if validation.functional_status != "SUCCESS":
        return EvaluationResult(
            validation=validation,
            measurement=ExecutionResult(success=False, error_type="VALIDATION_FAILED",
                                       error_message=validation.error_details)
        )
    
    # Krok 2: Pomiar generated
    measurement = measure_execution_time(code, task, n_warmup=n_warmup, n_runs=n_runs)
    
    if not measurement.success:
        return EvaluationResult(validation=validation, measurement=measurement)
    
    # Krok 3: Pomiar canonical (na tych samych warunkach)
    canonical_measurement = measure_execution_time(
        task.canonical_solution, task, n_warmup=n_warmup, n_runs=n_runs
    )
    
    if not canonical_measurement.success:
        # Mamy pomiar generated, ale nie canonical — zwróć bez eta
        return EvaluationResult(
            validation=validation,
            measurement=measurement,
            canonical_measurement=canonical_measurement
        )
    
    # Krok 4: Eta
    eta = canonical_measurement.median_ms / measurement.median_ms
    speedup = eta  # alias dla czytelności
    
    return EvaluationResult(
        validation=validation,
        measurement=measurement,
        canonical_measurement=canonical_measurement,
        eta_efficiency=eta,
        speedup_vs_canonical=speedup,
    )


# === CLI dla testów ===

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    print("=== TEST: Load sample ===")
    tasks = load_sample_50()
    print(f"Loaded {len(tasks)} tasks")
    print(f"First task: idx={tasks[0].problem_idx}, name='{tasks[0].task_name}'")
    print(f"Time bucket: {tasks[0].time_bucket}, category: {tasks[0].keyword_category}")
    print(f"Canonical time: {tasks[0].canonical_time_ms:.3f}ms")
    
    print("\n=== TEST: Evaluate canonical_solution (sanity check, eta should ≈ 1) ===")
    task = tasks[0]
    result = evaluate_generated_code(task.canonical_solution, task)
    print(result.summary())
    
    print("\n=== TEST: Evaluate broken code (should fail validation) ===")
    broken = "class Solution:\n    def wrong_method(self): return 'oops'"
    result = evaluate_generated_code(broken, task)
    print(result.summary())
    print(f"Status: {result.validation.functional_status}")
    print(f"Error: {result.validation.error_details}")
    
    print("\n=== TEST: Evaluate intentionally slower code ===")
    slower = """
class Solution:
    def lengthOfLongestSubstring(self, s):
        # Brute force O(n^3)
        best = 0
        for i in range(len(s)):
            for j in range(i+1, len(s)+1):
                substr = s[i:j]
                if len(set(substr)) == len(substr):
                    best = max(best, len(substr))
        return best
"""
    result = evaluate_generated_code(slower, task)
    print(result.summary())