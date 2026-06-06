"""
Główny orkiestrator eksperymentu.

Workflow per zadanie × model × powtórzenie:
  1. Konstrukcja promptu z szablonu
  2. Wywołanie modelu (API)
  3. Ekstrakcja kodu z odpowiedzi
  4. Wykonanie testów (jeśli generowanie)
  5. Analiza statyczna (Radon, Ruff, Bandit)
  6. Zapis do bazy SQLite

Pipeline obsługuje:
- Wznawianie po przerwaniu (sprawdza, co już jest w bazie)
- Retry przy błędach API
- Logowanie postępu

Użycie:
    python orchestrator.py --benchmark humaneval_plus --model claude-haiku-3.5
"""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Dodaj src do PATH
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from config import MODELS, CONFIG, DATA_DIR, DB_PATH, LOGS_DIR
from database import init_db, insert_generation, already_generated, get_db
from api_clients import get_client
from prompts import get_template
from extractors import extract_code, extract_audit_response
from test_runner import run_test_in_subprocess, build_test_program
from static_analysis import analyze_code

from tqdm import tqdm


def load_tasks(benchmark: str) -> list[dict]:
    """Wczytuje zadania z pliku JSONL."""
    path = DATA_DIR / f"{benchmark}_tasks.jsonl"
    if not path.exists():
        print(f"BŁĄD: Plik {path} nie istnieje.")
        print("Najpierw uruchom: python scripts/download_benchmarks.py")
        sys.exit(1)

    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def get_model_config(model_name: str):
    """Pobiera konfigurację modelu z config.MODELS."""
    for m in MODELS:
        if m.name == model_name:
            return m
    raise ValueError(f"Nieznany model: {model_name}. "
                     f"Dostępne: {[m.name for m in MODELS]}")


def process_one_generation(
    task: dict,
    model_cfg,
    sample_idx: int,
    template_id: str,
    experiment_type: str,
):
    """
    Przetwarza pojedynczą generację: prompt -> model -> ekstrakcja -> ewaluacja.

    Returns dict z wynikami (do zapisu w bazie).
    """
    template = get_template(template_id)

    # Konstrukcja promptu
    if experiment_type == "audit_auditor":
        # Audytor: kod do audytu (z wcześniejszej generacji lub canonical)
        audited_code = task.get("audited_code", task.get("canonical_solution", ""))
        system, user = template.format(code=audited_code)
    else:
        # Generator: prompt z benchmarku
        system, user = template.format(prompt=task["prompt"])

    # Wywołanie modelu
    client = get_client(model_cfg.provider)

    if model_cfg.provider == "huggingface":
        # OSS modele - placeholder, prawdziwa implementacja w Kaggle notebook
        return {"error": "OSS models run in Kaggle notebook - skip here"}

    result = client.generate(
        system_prompt=system,
        user_prompt=user,
        model_id=model_cfg.api_model_id,
        temperature=CONFIG.temperature,
        max_tokens=CONFIG.max_tokens,
        top_p=CONFIG.top_p,
    )

    # Ekstrakcja kodu (jeśli generation, nie audytor)
    extracted_code = None
    extraction_status = "n/a"
    if experiment_type != "audit_auditor":
        extr = extract_code(result.raw_response)
        extracted_code = extr.code
        extraction_status = extr.status

    # Zapis do bazy
    gen_id = insert_generation(
        db_path=DB_PATH,
        experiment_type=experiment_type,
        benchmark=task["benchmark"],
        task_id=task["task_id"],
        model_name=model_cfg.name,
        sample_idx=sample_idx,
        temperature=CONFIG.temperature,
        top_p=CONFIG.top_p,
        max_tokens=CONFIG.max_tokens,
        raw_response=result.raw_response,
        extracted_code=extracted_code,
        extraction_status=extraction_status,
        generation_time_sec=result.generation_time_sec,
        tokens_input=result.tokens_input,
        tokens_output=result.tokens_output,
        error_message=result.error,
    )

    # Ewaluacja zależnie od typu eksperymentu
    if experiment_type == "generation" and extracted_code:
        evaluate_generation(gen_id, task, extracted_code)
    elif experiment_type == "audit_generator" and extracted_code:
        evaluate_audit_generator(gen_id, task, extracted_code)
    elif experiment_type == "audit_auditor":
        evaluate_audit_auditor(gen_id, task, result.raw_response)

    return {
        "gen_id": gen_id,
        "extraction_status": extraction_status,
        "tokens": result.tokens_input + result.tokens_output,
        "time": result.generation_time_sec,
    }


def evaluate_generation(gen_id: int, task: dict, code: str):
    """Ewaluacja dla zadań generowania kodu (HumanEval+, LiveCodeBench)."""
    # Test
    test_program = build_test_program(
        extracted_code=code,
        test_code=task.get("test", ""),
        entry_point=task.get("entry_point", ""),
    )
    test_result = run_test_in_subprocess(
        test_program,
        timeout_sec=CONFIG.test_timeout_sec,
        memory_limit_mb=CONFIG.test_memory_limit_mb,
    )

    # Analiza statyczna
    analysis = analyze_code(code)

    # Zapis
    with get_db(DB_PATH) as conn:
        # test_results
        conn.execute(
            """
            INSERT OR REPLACE INTO test_results
            (generation_id, passed, n_tests_total, n_tests_passed,
             test_pass_rate, execution_time_sec, timeout, error_type, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (gen_id, test_result.passed, test_result.n_tests_total,
             test_result.n_tests_passed, test_result.test_pass_rate,
             test_result.execution_time_sec, test_result.timeout,
             test_result.error_type, test_result.error_message),
        )

        # quality_metrics
        q = analysis.quality
        conn.execute(
            """
            INSERT OR REPLACE INTO quality_metrics
            (generation_id, cyclomatic_complexity_avg, cyclomatic_complexity_max,
             halstead_volume, halstead_difficulty, halstead_effort,
             maintainability_index, pep8_violations_total,
             pep8_violations_naming, pep8_violations_formatting, pep8_violations_imports,
             lines_of_code, n_functions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (gen_id, q.cc_avg, q.cc_max, q.halstead_volume, q.halstead_difficulty,
             q.halstead_effort, q.maintainability_index, q.pep8_violations_total,
             q.pep8_violations_naming, q.pep8_violations_formatting,
             q.pep8_violations_imports, q.lines_of_code, q.n_functions),
        )


def evaluate_audit_generator(gen_id: int, task: dict, code: str):
    """Ewaluacja dla generowania kodu w SecurityEval (audyt rola generatora)."""
    analysis = analyze_code(code)

    with get_db(DB_PATH) as conn:
        # quality_metrics (jak wyżej)
        q = analysis.quality
        conn.execute(
            """
            INSERT OR REPLACE INTO quality_metrics
            (generation_id, cyclomatic_complexity_avg, cyclomatic_complexity_max,
             halstead_volume, halstead_difficulty, halstead_effort,
             maintainability_index, pep8_violations_total,
             pep8_violations_naming, pep8_violations_formatting, pep8_violations_imports,
             lines_of_code, n_functions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (gen_id, q.cc_avg, q.cc_max, q.halstead_volume, q.halstead_difficulty,
             q.halstead_effort, q.maintainability_index, q.pep8_violations_total,
             q.pep8_violations_naming, q.pep8_violations_formatting,
             q.pep8_violations_imports, q.lines_of_code, q.n_functions),
        )

        # security_findings
        for f in analysis.findings:
            conn.execute(
                """
                INSERT INTO security_findings
                (generation_id, cwe_id, test_id, severity, confidence,
                 line_number, code_snippet, issue_text, tool)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (gen_id, f.cwe_id, f.test_id, f.severity, f.confidence,
                 f.line_number, f.code_snippet, f.issue_text, f.tool),
            )


def evaluate_audit_auditor(gen_id: int, task: dict, raw_response: str):
    """Ewaluacja dla audytu (rola audytora) - parsowanie odpowiedzi modelu."""
    parsed = extract_audit_response(raw_response)

    ground_truth_cwe = task.get("cwe_id")
    is_actually_vulnerable = ground_truth_cwe is not None

    if "parse_error" in parsed:
        detected = None
        classification = "PARSE_ERROR"
        detected_cwe = None
        confidence = None
    else:
        detected = bool(parsed.get("vulnerable", False))
        detected_cwe = parsed.get("cwe_id")
        confidence = parsed.get("confidence")

        # Klasyfikacja
        if is_actually_vulnerable and detected:
            classification = "TP"
        elif is_actually_vulnerable and not detected:
            classification = "FN"
        elif not is_actually_vulnerable and detected:
            classification = "FP"
        else:
            classification = "TN"

    with get_db(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO audit_responses
            (generation_id, audited_code, ground_truth_cwe,
             detected_vulnerability, detected_cwe, confidence_score, classification)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (gen_id, task.get("audited_code", ""), ground_truth_cwe,
             detected, detected_cwe, confidence, classification),
        )


# =============================================================================
# MAIN
# =============================================================================
def run_experiment(
    benchmark: str,
    model_name: str,
    experiment_type: str = "generation",
    template_id: str = "gen_zeroshot_v1",
    n_samples: int = None,
    limit: int = None,
):
    """Uruchamia eksperyment dla jednego modelu na jednym benchmarku."""

    # Inicjalizacja bazy
    if not DB_PATH.exists():
        init_db(DB_PATH)

    # Wczytanie zadań
    tasks = load_tasks(benchmark)
    if limit:
        tasks = tasks[:limit]

    # Konfiguracja modelu
    model_cfg = get_model_config(model_name)

    if model_cfg.provider == "huggingface":
        print(f"BŁĄD: Model {model_name} jest typu HuggingFace - "
              f"uruchom go w Kaggle notebook, nie tutaj.")
        return

    # n_samples zależy od typu eksperymentu
    if n_samples is None:
        if experiment_type == "audit_auditor":
            n_samples = CONFIG.n_samples_audit
        else:
            n_samples = CONFIG.n_samples_generation

    # Logging
    log_file = LOGS_DIR / f"{benchmark}_{model_name}_{experiment_type}_{datetime.now():%Y%m%d_%H%M%S}.log"

    total = len(tasks) * n_samples
    print(f"\n{'='*60}")
    print(f"Eksperyment: {experiment_type}")
    print(f"Benchmark:   {benchmark} ({len(tasks)} zadań)")
    print(f"Model:       {model_name} ({model_cfg.api_model_id})")
    print(f"Powtórzenia: {n_samples}")
    print(f"Łącznie:     {total} generacji")
    print(f"Log:         {log_file}")
    print(f"{'='*60}\n")

    # Pętla główna z progress bar
    skipped = 0
    completed = 0
    errors = 0

    with open(log_file, "w") as log:
        log.write(f"Start: {datetime.now()}\n")
        log.write(f"Config: {experiment_type}/{benchmark}/{model_name}/n={n_samples}\n\n")

        with tqdm(total=total, desc=f"{model_name}/{benchmark}") as pbar:
            for task in tasks:
                for sample_idx in range(n_samples):
                    # Sprawdzenie, czy już wygenerowano
                    if already_generated(
                        DB_PATH, experiment_type, benchmark,
                        task["task_id"], model_name, sample_idx,
                    ):
                        skipped += 1
                        pbar.update(1)
                        continue

                    try:
                        result = process_one_generation(
                            task=task,
                            model_cfg=model_cfg,
                            sample_idx=sample_idx,
                            template_id=template_id,
                            experiment_type=experiment_type,
                        )
                        completed += 1
                        log.write(f"[OK] {task['task_id']}/{sample_idx}: "
                                 f"{result.get('extraction_status')}, "
                                 f"{result.get('time', 0):.2f}s\n")

                    except Exception as e:
                        errors += 1
                        log.write(f"[ERR] {task['task_id']}/{sample_idx}: {e}\n")
                        # Krótka pauza po błędzie - może rate limit
                        time.sleep(2)

                    pbar.update(1)
                    pbar.set_postfix(done=completed, skip=skipped, err=errors)

        log.write(f"\nKoniec: {datetime.now()}\n")
        log.write(f"Completed: {completed}, Skipped: {skipped}, Errors: {errors}\n")

    print(f"\nZakończono: {completed} nowych, {skipped} pominiętych, {errors} błędów")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orkiestrator eksperymentu LLM")
    parser.add_argument("--benchmark", required=True,
                       choices=["humaneval_plus", "livecodebench", "securityeval"])
    parser.add_argument("--model", required=True,
                       choices=[m.name for m in MODELS])
    parser.add_argument("--experiment", default="generation",
                       choices=["generation", "audit_generator", "audit_auditor"])
    parser.add_argument("--template", default="gen_zeroshot_v1")
    parser.add_argument("--n-samples", type=int, default=None,
                       help="Liczba powtórzeń (default z config)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit zadań (do testów / pilotażu)")
    args = parser.parse_args()

    # Auto-dobór szablonu na podstawie typu eksperymentu
    if args.template == "gen_zeroshot_v1":
        if args.experiment == "audit_generator":
            args.template = "audit_gen_v1"
        elif args.experiment == "audit_auditor":
            args.template = "auditor_v1"

    run_experiment(
        benchmark=args.benchmark,
        model_name=args.model,
        experiment_type=args.experiment,
        template_id=args.template,
        n_samples=args.n_samples,
        limit=args.limit,
    )
