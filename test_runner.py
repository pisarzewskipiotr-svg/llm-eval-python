"""
Wykonanie testów dla wygenerowanego kodu w izolowanym środowisku.

Strategia: subprocess z timeout i (opcjonalnie) resource limits (Linux/macOS).
Każde wywołanie tworzy tymczasowy plik z kodem i testami,
uruchamia go w subprocess i analizuje wyniki.

UWAGA: To NIE jest pełna izolacja jak Docker/sandbox.
Jest to praktyczny kompromis dla pracy magisterskiej.
NIE uruchamiaj eksperymentu na maszynie produkcyjnej.

KOMPATYBILNOŚĆ:
- Linux/macOS: pełne wsparcie z limitami pamięci (resource module)
- Windows: subprocess + timeout (bez memory limit, ale wystarczające dla naszego użytku)
"""
import subprocess
import tempfile
import time
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# resource module is POSIX-only (Linux, macOS)
# Na Windowsie nie jest dostępny ale pipeline działa bez niego
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False


@dataclass
class TestResult:
    """Wynik wykonania testów."""
    passed: bool                       # czy WSZYSTKIE testy przeszły
    n_tests_total: int                 # ile testów ogółem
    n_tests_passed: int                # ile przeszło
    test_pass_rate: float              # n_passed / n_total
    execution_time_sec: float
    memory_peak_mb: Optional[float]
    timeout: bool
    error_type: Optional[str]          # 'syntax' | 'runtime' | 'assertion' | 'timeout' | 'other'
    error_message: Optional[str]
    stdout: str
    stderr: str


def set_resource_limits(memory_mb: int):
    """
    Ustawia limity zasobów dla procesu potomnego (Linux/macOS).
    Na Windowsie - no-op (Windows używa innego API, niekompatybilnego).
    """
    if not HAS_RESOURCE:
        return  # Windows - skip

    # Memory limit
    memory_bytes = memory_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        pass

    # CPU time limit (na wszelki wypadek)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    except (ValueError, OSError):
        pass


def run_test_in_subprocess(
    code_with_test: str,
    timeout_sec: int = 30,
    memory_limit_mb: int = 1024,
) -> TestResult:
    """
    Uruchamia kod + testy w izolowanym subprocess.

    Args:
        code_with_test: kod Pythona zawierający implementację i wywołania testowe
        timeout_sec: limit czasu wykonania
        memory_limit_mb: limit pamięci (tylko Linux/macOS)

    Returns:
        TestResult ze statystykami wykonania
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code_with_test)
        tmp_path = f.name

    start = time.time()
    timeout_hit = False

    try:
        # preexec_fn działa tylko na POSIX (Linux/macOS)
        # Na Windowsie subprocess automatycznie ignoruje ten parametr,
        # ale lepiej go nie przekazywać explicite
        kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": timeout_sec,
        }

        if HAS_RESOURCE and sys.platform != "win32":
            # Linux/macOS - ustaw limity pamięci dla procesu potomnego
            kwargs["preexec_fn"] = lambda: set_resource_limits(memory_limit_mb)
        # Windows - tylko timeout (memory limit nieobsługiwany w subprocess)

        result = subprocess.run(
            [sys.executable, tmp_path],
            **kwargs,
        )

        elapsed = time.time() - start
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode

    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start
        timeout_hit = True
        stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        returncode = -1

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Klasyfikacja wyniku
    if timeout_hit:
        return TestResult(
            passed=False,
            n_tests_total=0,
            n_tests_passed=0,
            test_pass_rate=0.0,
            execution_time_sec=elapsed,
            memory_peak_mb=None,
            timeout=True,
            error_type="timeout",
            error_message=f"Execution exceeded {timeout_sec}s timeout",
            stdout=stdout,
            stderr=stderr,
        )

    # Klasyfikacja błędu na podstawie stderr
    error_type = None
    error_message = None

    if returncode != 0:
        if "SyntaxError" in stderr:
            error_type = "syntax"
        elif "AssertionError" in stderr:
            error_type = "assertion"
        elif "MemoryError" in stderr:
            error_type = "memory"
        else:
            error_type = "runtime"

        error_message = stderr.strip().split("\n")[-1] if stderr else "Unknown error"

    # Próba sparsowania liczby testów z stdout/stderr
    n_total, n_passed = parse_test_counts(stdout, stderr, returncode)

    passed = (returncode == 0)
    return TestResult(
        passed=passed,
        n_tests_total=n_total,
        n_tests_passed=n_passed,
        test_pass_rate=n_passed / n_total if n_total > 0 else 0.0,
        execution_time_sec=elapsed,
        memory_peak_mb=None,  # subprocess nie raportuje tego łatwo
        timeout=False,
        error_type=error_type,
        error_message=error_message,
        stdout=stdout[:2000],  # ograniczenie rozmiaru
        stderr=stderr[:2000],
    )


def parse_test_counts(stdout: str, stderr: str, returncode: int) -> tuple[int, int]:
    """
    Próbuje sparsować liczbę testów z output.

    Konwencja: jeśli kod używa wielu assert, każdy jest osobnym testem.
    Jeśli zawiódł, parsujemy z stderr który assert zawiódł.
    """
    if returncode == 0:
        # Wszystko przeszło - próbujemy zliczyć "passed" w stdout
        if "passed" in stdout.lower():
            # Format pytest: "5 passed"
            import re
            match = re.search(r"(\d+)\s+passed", stdout)
            if match:
                n = int(match.group(1))
                return n, n
        # Domyślnie: jeden block testów = 1 test
        return 1, 1
    else:
        # Coś się wywaliło - jeden test, zero przeszło
        return 1, 0


def build_test_program(
    extracted_code: str,
    test_code: str,
    entry_point: str = "",
) -> str:
    """
    Składa pełny program testowy: implementacja + testy.

    Args:
        extracted_code: kod wygenerowany przez model (definicja funkcji)
        test_code: kod testowy z benchmarku (np. funkcja `check`)
        entry_point: nazwa funkcji do testowania

    Returns:
        Pełny program gotowy do wykonania
    """
    parts = [
        "# Wygenerowany kod:",
        extracted_code,
        "",
        "# Testy z benchmarku:",
        test_code,
        "",
    ]

    # HumanEval format: testy są w funkcji `check`, którą trzeba wywołać
    if "def check(" in test_code and entry_point:
        parts.append(f"check({entry_point})")
        parts.append('print("All tests passed!")')

    return "\n".join(parts)


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print(f"Platform: {sys.platform}")
    print(f"Resource module: {'available' if HAS_RESOURCE else 'NOT available (Windows)'}")
    print()

    print("Test runner - poprawny kod:")
    code = """
def add(a, b):
    return a + b

assert add(2, 3) == 5
assert add(0, 0) == 0
print("All tests passed!")
"""
    result = run_test_in_subprocess(code, timeout_sec=10)
    print(f"  Passed: {result.passed}, time: {result.execution_time_sec:.3f}s")
    print(f"  Error type: {result.error_type}")

    print("\nTest runner - kod z błędem:")
    code = """
def add(a, b):
    return a - b  # bug

assert add(2, 3) == 5
"""
    result = run_test_in_subprocess(code, timeout_sec=10)
    print(f"  Passed: {result.passed}")
    print(f"  Error type: {result.error_type}")
    print(f"  Error msg: {result.error_message}")

    print("\nTest runner - timeout:")
    code = """
while True:
    pass
"""
    result = run_test_in_subprocess(code, timeout_sec=2)
    print(f"  Passed: {result.passed}, timeout: {result.timeout}")
    print(f"  Error type: {result.error_type}")