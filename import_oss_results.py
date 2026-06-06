"""
Import wyników z notebooka Kaggle do lokalnej bazy SQLite.

Workflow:
1. Pobierz pliki .jsonl z Kaggle (z folderu outputs/) - wgraj lokalnie do data/oss_results/
2. Uruchom ten skrypt
3. Skrypt:
   a) Wczyta JSONL
   b) Wstawi rekordy generations do SQLite
   c) Wykona ekstrakcję kodu i analizę statyczną LOKALNIE
   d) Wykona testy LOKALNIE (jeśli generation)
   e) Zapisze wszystko do bazy

Użycie:
    python src/import_oss_results.py data/oss_results/qwen2.5-coder-7b_humaneval_plus_generation.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from config import DB_PATH, DATA_DIR, CONFIG
from database import init_db, insert_generation, already_generated
from extractors import extract_code
from orchestrator import (
    evaluate_generation,
    evaluate_audit_generator,
    evaluate_audit_auditor,
    load_tasks,
)

from tqdm import tqdm


def import_jsonl(jsonl_path: Path):
    """Importuje rekordy z JSONL do bazy."""
    if not jsonl_path.exists():
        print(f"BŁĄD: {jsonl_path} nie istnieje")
        sys.exit(1)

    if not DB_PATH.exists():
        init_db(DB_PATH)

    # Wczytaj wszystkie rekordy
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        print("Brak rekordów do importu")
        return

    benchmark = records[0]["benchmark"]
    experiment_type = records[0]["experiment_type"]
    model_name = records[0]["model_name"]

    print(f"Import: {len(records)} rekordów")
    print(f"  Benchmark: {benchmark}")
    print(f"  Experiment: {experiment_type}")
    print(f"  Model: {model_name}")

    # Wczytaj zadania benchmarku (potrzebne do ewaluacji)
    tasks = {t["task_id"]: t for t in load_tasks(benchmark)}

    imported = 0
    skipped = 0
    errors = 0

    with tqdm(total=len(records), desc="Import") as pbar:
        for rec in records:
            try:
                # Sprawdź, czy już importowano
                if already_generated(
                    DB_PATH, experiment_type, benchmark,
                    rec["task_id"], model_name, rec["sample_idx"],
                ):
                    skipped += 1
                    pbar.update(1)
                    continue

                # Ekstrakcja kodu
                if experiment_type != "audit_auditor":
                    extr = extract_code(rec["raw_response"])
                    extracted_code = extr.code
                    extraction_status = extr.status
                else:
                    extracted_code = None
                    extraction_status = "n/a"

                # Wstaw do bazy
                gen_id = insert_generation(
                    db_path=DB_PATH,
                    experiment_type=experiment_type,
                    benchmark=benchmark,
                    task_id=rec["task_id"],
                    model_name=model_name,
                    sample_idx=rec["sample_idx"],
                    temperature=rec["temperature"],
                    top_p=rec.get("top_p"),
                    raw_response=rec["raw_response"],
                    extracted_code=extracted_code,
                    extraction_status=extraction_status,
                    generation_time_sec=rec.get("generation_time_sec"),
                    tokens_input=rec.get("tokens_input"),
                    tokens_output=rec.get("tokens_output"),
                )

                # Ewaluacja (testy + analiza statyczna)
                task = tasks.get(rec["task_id"])
                if task and extracted_code:
                    if experiment_type == "generation":
                        evaluate_generation(gen_id, task, extracted_code)
                    elif experiment_type == "audit_generator":
                        evaluate_audit_generator(gen_id, task, extracted_code)

                if experiment_type == "audit_auditor" and task:
                    evaluate_audit_auditor(gen_id, task, rec["raw_response"])

                imported += 1

            except Exception as e:
                errors += 1
                print(f"Błąd przy {rec.get('task_id')}/{rec.get('sample_idx')}: {e}")

            pbar.update(1)
            pbar.set_postfix(imp=imported, skip=skipped, err=errors)

    print(f"\nZakończono: {imported} zaimportowano, {skipped} pominięto, {errors} błędów")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", help="Ścieżka do pliku JSONL z Kaggle")
    args = parser.parse_args()

    import_jsonl(Path(args.jsonl_path))
