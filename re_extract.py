"""
Re-ekstraktor: ponowna ekstrakcja kodu dla wpisów w bazie ze statusem
'no_code_block' lub 'invalid_syntax'.

Operuje WYŁĄCZNIE lokalnie - bez wywołań API. Surowe odpowiedzi są w bazie.

Workflow per rekord:
1. Re-ekstrakcja z naprawionym extract_code() (5 metod)
2. Jeśli sukces:
   a. UPDATE generations SET extracted_code, extraction_status='success'
   b. INSERT do test_results (jeśli generation/audit_generator i nie istnieje)
   c. INSERT do quality_metrics (jeśli nie istnieje)
   d. INSERT do security_findings (Bandit findings - jeśli są)
"""
import json
import sys
import sqlite3
from pathlib import Path

# Dostosuj ścieżkę żeby importować z venv (gdzie są pliki Piotra)
SCRIPT_DIR = Path(__file__).parent
# Próbuj importować z parent (jeśli skrypt jest w scripts/) albo z bieżącego folderu
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from extractors import extract_code
from test_runner import run_test_in_subprocess, build_test_program
from static_analysis import analyze_code


# =============================================================================
# Konfiguracja
# =============================================================================
DB_PATH = r"J:\Praca wyniki\results\experiment.sqlite"
DATA_DIR = Path(r"J:\Praca wyniki\data")


def load_tasks_for_benchmark(benchmark: str) -> dict:
    """Wczytuje zadania benchmarku - mapowanie task_id -> task_data."""
    path = DATA_DIR / f"{benchmark}_tasks.jsonl"
    tasks = {}
    if not path.exists():
        print(f"OSTRZEŻENIE: Nie znaleziono {path}")
        return tasks
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            tasks[t["task_id"]] = t
    return tasks


def insert_test_result(cur, gen_id: int, test_result):
    """Wstawia wynik testu do bazy."""
    cur.execute(
        """
        INSERT INTO test_results
        (generation_id, passed, n_tests_total, n_tests_passed,
         test_pass_rate, execution_time_sec, memory_peak_mb,
         timeout, error_type, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            gen_id,
            1 if test_result.passed else 0,
            test_result.n_tests_total,
            test_result.n_tests_passed,
            test_result.test_pass_rate,
            test_result.execution_time_sec,
            test_result.memory_peak_mb,
            1 if test_result.timeout else 0,
            test_result.error_type,
            test_result.error_message[:1000] if test_result.error_message else None,
        ),
    )


def insert_quality_metrics(cur, gen_id: int, quality):
    """Wstawia metryki jakości do bazy. quality to QualityMetrics z static_analysis."""
    cur.execute(
        """
        INSERT INTO quality_metrics
        (generation_id, cyclomatic_complexity_avg, cyclomatic_complexity_max,
         halstead_volume, halstead_difficulty, halstead_effort,
         maintainability_index,
         pep8_violations_total, pep8_violations_naming,
         pep8_violations_formatting, pep8_violations_imports,
         lines_of_code, n_functions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            gen_id,
            quality.cc_avg,
            quality.cc_max,
            quality.halstead_volume,
            quality.halstead_difficulty,
            quality.halstead_effort,
            quality.maintainability_index,
            quality.pep8_violations_total,
            quality.pep8_violations_naming,
            quality.pep8_violations_formatting,
            quality.pep8_violations_imports,
            quality.lines_of_code,
            quality.n_functions,
        ),
    )


def insert_security_findings(cur, gen_id: int, findings):
    """Wstawia wykrycia podatności do bazy."""
    for f in findings:
        cur.execute(
            """
            INSERT INTO security_findings
            (generation_id, cwe_id, test_id, severity, confidence,
             line_number, code_snippet, issue_text, tool)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gen_id,
                f.cwe_id,
                f.test_id,
                f.severity,
                f.confidence,
                f.line_number,
                f.code_snippet[:1000] if f.code_snippet else None,
                f.issue_text[:1000] if f.issue_text else None,
                f.tool,
            ),
        )


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Znajdź wszystkie wpisy do re-ekstrakcji
    rows = cur.execute(
        """
        SELECT id, model_name, benchmark, experiment_type,
               task_id, sample_idx, raw_response, extraction_status
        FROM generations
        WHERE extraction_status IN ('no_code_block', 'invalid_syntax')
          AND raw_response != ''
          AND length(raw_response) > 0
        ORDER BY benchmark, model_name, task_id
        """
    ).fetchall()

    print(f"Znaleziono {len(rows)} wpisów do re-ekstrakcji")

    if not rows:
        print("Nic do zrobienia.")
        conn.close()
        return

    # Cache: zadania per benchmark
    tasks_by_benchmark = {
        "humaneval_plus": load_tasks_for_benchmark("humaneval_plus"),
        "securityeval": load_tasks_for_benchmark("securityeval"),
    }

    # Sprawdź czy w bazie nie ma już test_results / quality_metrics dla danej generacji
    def has_test_result(gen_id):
        return cur.execute(
            "SELECT 1 FROM test_results WHERE generation_id = ? LIMIT 1", (gen_id,)
        ).fetchone() is not None

    def has_quality_metrics(gen_id):
        return cur.execute(
            "SELECT 1 FROM quality_metrics WHERE generation_id = ? LIMIT 1", (gen_id,)
        ).fetchone() is not None

    # Statystyki
    stats = {
        "recovered": 0,
        "still_failing": 0,
        "tests_run": 0,
        "tests_passed": 0,
        "metrics_added": 0,
        "findings_added": 0,
        "errors": 0,
    }
    method_counts = {}

    pbar = tqdm(rows, desc="Re-extracting")
    for row in pbar:
        try:
            # === Re-ekstrakcja ===
            result = extract_code(row["raw_response"])

            if result.status != "success":
                stats["still_failing"] += 1
                continue

            # === Sukces - aktualizuj rekord ===
            cur.execute(
                """
                UPDATE generations
                SET extracted_code = ?,
                    extraction_status = 'success'
                WHERE id = ?
                """,
                (result.code, row["id"]),
            )
            stats["recovered"] += 1
            method_counts[result.method] = method_counts.get(result.method, 0) + 1

            # === Uruchom testy (tylko dla generation - HumanEval+) ===
            if (
                row["experiment_type"] == "generation"
                and row["benchmark"] == "humaneval_plus"
                and not has_test_result(row["id"])
            ):
                tasks = tasks_by_benchmark.get(row["benchmark"], {})
                task = tasks.get(row["task_id"])

                if task is not None:
                    try:
                        test_program = build_test_program(
                            extracted_code=result.code,
                            test_code=task.get("test", ""),
                            entry_point=task.get("entry_point", ""),
                        )
                        test_result = run_test_in_subprocess(
                            test_program, timeout_sec=30
                        )
                        insert_test_result(cur, row["id"], test_result)
                        stats["tests_run"] += 1
                        if test_result.passed:
                            stats["tests_passed"] += 1
                    except Exception as e:
                        stats["errors"] += 1

            # === Analiza statyczna ===
            if not has_quality_metrics(row["id"]):
                try:
                    analysis = analyze_code(result.code)
                    insert_quality_metrics(cur, row["id"], analysis.quality)
                    stats["metrics_added"] += 1

                    if analysis.findings:
                        insert_security_findings(cur, row["id"], analysis.findings)
                        stats["findings_added"] += len(analysis.findings)
                except Exception as e:
                    stats["errors"] += 1

            # Commit co 50 wpisów
            if stats["recovered"] % 50 == 0:
                conn.commit()
                pbar.set_postfix(
                    rec=stats["recovered"],
                    fail=stats["still_failing"],
                    test=stats["tests_run"],
                )

        except Exception as e:
            stats["errors"] += 1

    conn.commit()
    conn.close()

    print("\n" + "=" * 60)
    print("PODSUMOWANIE RE-EKSTRAKCJI")
    print("=" * 60)
    print(f"Odzyskanych wpisów:     {stats['recovered']}")
    print(f"Nadal nie udało się:    {stats['still_failing']}")
    print(f"Testów uruchomionych:   {stats['tests_run']}")
    print(f"Testów przeszło:        {stats['tests_passed']}")
    print(f"Metryk dodanych:        {stats['metrics_added']}")
    print(f"Findings dodanych:      {stats['findings_added']}")
    print(f"Błędów:                 {stats['errors']}")
    print()
    print("Metody odzyskania:")
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method:35} {count:>5}")


if __name__ == "__main__":
    main()