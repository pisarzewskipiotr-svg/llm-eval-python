"""
Smart re-extraction dla DeepSeek z handling:
- BPE encoding (Ġ → space, Ċ → newline)
- Jupyter markers (<jupyter_text>, <pre>, <code>)
- Truncation recovery (usuwanie niekompletnych linii)
- Bare class/def extraction (bez markdown fences)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import json
import re
import platform
import logging
from tqdm import tqdm

from llm_eval.benchmarks.effibench import load_sample_50, evaluate_generated_code

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


JSONL_PATH = Path(r'external/kaggle_outputs/outputs_deepseek-ai_deepseek-coder-6.7b-instruct.jsonl')
DB_PATH = 'results/experiment.sqlite'


def fix_bpe_encoding(text):
    """Wymień BPE tokens na rzeczywiste znaki."""
    if not text:
        return text
    return (text
        .replace('Ġ', ' ')
        .replace('Ċ', '\n')
        .replace('ĉ', '\t')
        .replace('âĢĻ', "'")
        .replace('âĢľ', '"')
        .replace('âĢĿ', '"')
    )


def remove_jupyter_markers(text):
    """Usuń jupyter-style markers."""
    markers = [
        '<jupyter_text>', '</jupyter_text>',
        '<jupyter_code>', '</jupyter_code>',
        '<jupyter_output>', '</jupyter_output>',
        '<pre>', '</pre>',
        '<code>', '</code>',
        '<text>', '</text>',
    ]
    for m in markers:
        text = text.replace(m, '')
    return text


def try_compile(code):
    """Sprawdź czy kod kompiluje (warnings ignored)."""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            compile(code, '<test>', 'exec')
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def try_fix_truncated(code, max_attempts=30):
    """Usuń ostatnie linie aż syntax OK."""
    if try_compile(code):
        return code
    
    lines = code.split('\n')
    if len(lines) <= 2:
        return None
    
    for n in range(1, min(max_attempts, len(lines))):
        attempt = '\n'.join(lines[:-n]).rstrip()
        if try_compile(attempt):
            # Sprawdź czy zawiera class Solution albo def
            if 'class Solution' in attempt or 'def ' in attempt:
                return attempt
    return None


def smart_extract_code(raw_text):
    """
    Inteligentnie wyciąga kod z DeepSeek output.
    
    Returns: (code, extraction_method) lub (None, 'no_code_extracted')
    """
    if not raw_text:
        return None, 'no_raw_response'
    
    # Step 1: Fix encoding
    text = fix_bpe_encoding(raw_text)
    
    # Step 2: Usuń jupyter markers
    text = remove_jupyter_markers(text)
    
    # Step 3: Strategie ekstrakcji w kolejności preferencji
    
    # 3a. ```python ... ``` (kompletny markdown block)
    m = re.search(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if try_compile(code) and ('class Solution' in code or 'def ' in code):
            return code, 'markdown_python_complete'
    
    # 3b. ``` ... ``` (generic markdown block)
    m = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if try_compile(code) and ('class Solution' in code or 'def ' in code):
            return code, 'markdown_generic_complete'
    
    # 3c. ```python ... (truncated, no closing)
    m = re.search(r'```python\s*\n(.+?)(?:\n```|\Z)', text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        fixed = try_fix_truncated(code)
        if fixed:
            return fixed, 'markdown_python_truncated'
    
    # 3d. 'class Solution:' - znajdź od tego miejsca
    if 'class Solution' in text:
        idx = text.find('class Solution')
        code = text[idx:]
        
        # Obetnij na typowych granicach które wskazują koniec kodu
        for delimiter in [
            '\n# Example usage',
            '\n# Test',
            '\n# Edge case',
            '\nprint(',
            '\nif __name__',
            '\nExplanation:',
            '\nThe solution',
            '\nThis solution',
            '\n```',
            '\nNote:',
        ]:
            pos = code.find(delimiter)
            if pos != -1:
                code = code[:pos]
        
        code = code.rstrip()
        fixed = try_fix_truncated(code)
        if fixed:
            return fixed, 'bare_class_solution'
    
    # 3e. 'def ' - standalone function (do auto-wrap)
    m = re.search(r'^def\s+\w+\s*\(', text, re.MULTILINE)
    if m:
        idx = m.start()
        code = text[idx:]
        for delimiter in ['\n# Example', '\n# Test', '\nprint(', '\nif __name__', '\n```', '\nExplanation']:
            pos = code.find(delimiter)
            if pos != -1:
                code = code[:pos]
        code = code.rstrip()
        fixed = try_fix_truncated(code)
        if fixed:
            # Auto-wrap: dodaj self do funkcji i opakuj w class Solution
            wrapped = auto_wrap_function(fixed)
            if wrapped and try_compile(wrapped):
                return wrapped, 'standalone_function_wrapped'
    
    return None, 'no_code_extracted'


def auto_wrap_function(code):
    """Wrap standalone function w class Solution z self parameter."""
    pattern = r'^(def\s+\w+\s*\()(.*?)(\)\s*(?:->\s*[^:]+)?\s*:)'
    
    def add_self(match):
        prefix = match.group(1)
        params = match.group(2).strip()
        suffix = match.group(3)
        if params.startswith('self'):
            return match.group(0)
        if params:
            return f'{prefix}self, {params}{suffix}'
        else:
            return f'{prefix}self{suffix}'
    
    new_code = re.sub(pattern, add_self, code, flags=re.MULTILINE)
    
    # Indent o 4 spacje
    indented = '\n'.join('    ' + line if line else line for line in new_code.split('\n'))
    return f'class Solution:\n{indented}'


def main():
    logger.info(f"Loading: {JSONL_PATH}")
    
    if not JSONL_PATH.exists():
        logger.error(f"File not found: {JSONL_PATH}")
        return
    
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        raw_gens = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(raw_gens)} raw generations")
    
    # Najpierw test extractora na pierwszych 10 generacjach
    logger.info("Testing smart extractor on first 10 generations:")
    extraction_methods = {}
    for i, g in enumerate(raw_gens[:10]):
        code, method = smart_extract_code(g.get('raw_response', ''))
        extraction_methods.setdefault(method, 0)
        extraction_methods[method] += 1
        if code:
            logger.info(f"  [{i+1}] {g['strategy']}/{g['iteration']}: {method} - {len(code)} chars")
        else:
            logger.info(f"  [{i+1}] {g['strategy']}/{g['iteration']}: FAILED ({method})")
    
    logger.info(f"\nExtraction methods (first 10): {extraction_methods}")
    print()
    
    # Pełen extraction stats
    logger.info("Running full extraction analysis (no DB write yet):")
    extraction_stats = {}
    for g in tqdm(raw_gens, desc='Extracting'):
        code, method = smart_extract_code(g.get('raw_response', ''))
        extraction_stats.setdefault(method, 0)
        extraction_stats[method] += 1
    
    print()
    print('=== EXTRACTION METHOD STATS ===')
    for method, count in sorted(extraction_stats.items(), key=lambda x: -x[1]):
        print(f'  {method}: {count} ({100*count/len(raw_gens):.1f}%)')
    
    total_extracted = sum(v for k, v in extraction_stats.items() if k != 'no_code_extracted' and k != 'no_raw_response')
    print(f'\nTotal extracted: {total_extracted}/{len(raw_gens)} ({100*total_extracted/len(raw_gens):.1f}%)')
    print(f'Improvement vs previous: {total_extracted} vs 216 (gain: +{total_extracted - 216})')
    
    # Zapytaj o zgodę przed re-importem
    print()
    print("=" * 70)
    response = input("Continue with full re-import (DELETE existing DeepSeek + re-eval)? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # === Pełen re-import ===
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Wyczyść istniejące DeepSeek
    n_before = cursor.execute(
        "SELECT COUNT(*) FROM optimization_results WHERE model_id = ?",
        ('deepseek-coder-6.7b',)
    ).fetchone()[0]
    logger.info(f"Deleting {n_before} existing DeepSeek rows")
    cursor.execute("DELETE FROM optimization_results WHERE model_id = ?", ('deepseek-coder-6.7b',))
    conn.commit()
    
    # Załaduj tasks
    tasks = {t.problem_idx: t for t in load_sample_50()}
    
    eval_stats = {
        'success': 0, 'logical_regression': 0, 'syntax_error': 0,
        'no_code_extracted': 0, 'timeout': 0, 'exception': 0
    }
    
    for g in tqdm(raw_gens, desc='Processing'):
        problem_idx = g['problem_idx']
        if problem_idx not in tasks:
            continue
        task = tasks[problem_idx]
        
        raw_response = g.get('raw_response', '')
        raw_fixed = fix_bpe_encoding(raw_response)
        raw_clean = remove_jupyter_markers(raw_fixed)
        
        code, method = smart_extract_code(raw_response)
        
        if code is None:
            eval_stats['no_code_extracted'] += 1
            # Zapisz placeholder
            cursor.execute('''
                INSERT INTO optimization_results (
                    problem_idx, task_name, model_id, strategy, iteration, sample_idx,
                    canonical_time_bucket, keyword_category, prompt_template_version,
                    raw_response, extracted_code, extraction_status,
                    tokens_input, tokens_output, functional_status,
                    hardware_id, python_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                problem_idx, task.task_name, 'deepseek-coder-6.7b',
                g['strategy'], g['iteration'], g['sample_idx'],
                task.time_bucket, task.keyword_category, 'v1',
                raw_clean, None, method,
                g.get('tokens_in', 0), g.get('tokens_out', 0),
                'NO_CODE_EXTRACTED',
                'kaggle-t4-gen+local-measure',
                platform.python_version()
            ))
            conn.commit()
            continue
        
        # Eval
        try:
            eval_result = evaluate_generated_code(code, task, n_warmup=3, n_runs=10)
            status = eval_result.validation.functional_status
            
            # Update stats
            if status == 'SUCCESS':
                eval_stats['success'] += 1
            elif status == 'LOGICAL_REGRESSION':
                eval_stats['logical_regression'] += 1
            elif status == 'SYNTAX_ERROR':
                eval_stats['syntax_error'] += 1
            elif status == 'TIMEOUT':
                eval_stats['timeout'] += 1
            
            # Save to DB
            cursor.execute('''
                INSERT INTO optimization_results (
                    problem_idx, task_name, model_id, strategy, iteration, sample_idx,
                    canonical_time_bucket, keyword_category, prompt_template_version,
                    raw_response, extracted_code, extraction_status,
                    tokens_input, tokens_output, api_cost_usd,
                    functional_status, validation_error,
                    generated_time_median_ms, generated_time_std_ms, generated_time_min_ms,
                    canonical_time_median_ms, eta_efficiency,
                    n_warmup, n_measurements,
                    hardware_id, python_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                problem_idx, task.task_name, 'deepseek-coder-6.7b',
                g['strategy'], g['iteration'], g['sample_idx'],
                task.time_bucket, task.keyword_category, 'v1',
                raw_clean, code, method,
                g.get('tokens_in', 0), g.get('tokens_out', 0), 0.0,
                status,
                eval_result.validation.error_details,
                eval_result.measurement.median_ms if eval_result.measurement.success else None,
                eval_result.measurement.std_ms if eval_result.measurement.success else None,
                eval_result.measurement.min_ms if eval_result.measurement.success else None,
                eval_result.canonical_measurement.median_ms if eval_result.canonical_measurement and eval_result.canonical_measurement.success else None,
                eval_result.eta_efficiency,
                3, 10,
                'kaggle-t4-gen+local-measure',
                platform.python_version()
            ))
            conn.commit()
        except Exception as e:
            eval_stats['exception'] += 1
            logger.warning(f"Exception for task {problem_idx} ({g['strategy']}/{g['iteration']}): {e}")
    
    print()
    print('=== EVAL STATS ===')
    for k, v in eval_stats.items():
        print(f'  {k}: {v}')
    
    # Final summary
    print()
    cursor.execute('''
        SELECT model_id, COUNT(*) as n,
            SUM(CASE WHEN functional_status = 'SUCCESS' THEN 1 ELSE 0 END) as ok,
            AVG(eta_efficiency) as avg_eta
        FROM optimization_results
        GROUP BY model_id ORDER BY model_id
    ''')
    print(f'{"Model":25s} {"N":5s} {"OK":5s} {"Success%":10s} {"Avg eta":10s}')
    print('-' * 65)
    for row in cursor.fetchall():
        model, n, ok, eta = row
        eta_str = f'{eta:.3f}' if eta else 'N/A'
        print(f'{model:25s} {n:5d} {ok:5d} {100*ok/n:8.1f}% {eta_str:10s}')
    
    conn.close()


if __name__ == '__main__':
    main()