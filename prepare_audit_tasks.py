"""
Przygotowanie zadań SecurityEval do roli audytora.

Pipeline orchestratora dla audit_auditor wymaga, żeby każde zadanie miało pole
'audited_code' - kod, który model ma ocenić pod kątem bezpieczeństwa.

Strategia: dla każdego zadania używamy `insecure_code` (znana podatna implementacja
z SecurityEval) lub `prompt` jako kodu do audytu. Ground truth dla klasyfikacji
TP/FN to obecność cwe_id - jeśli zadanie ma przypisany CWE, kod jest podatny.
"""
import json
import sys
from pathlib import Path

# Dodaj src do PATH
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import DATA_DIR

INPUT_PATH = DATA_DIR / "securityeval_tasks.jsonl"
OUTPUT_PATH = DATA_DIR / "securityeval_tasks.jsonl"  # nadpisuje (z backupem)
BACKUP_PATH = DATA_DIR / "securityeval_tasks.jsonl.backup"


def main():
    if not INPUT_PATH.exists():
        print(f"BŁĄD: {INPUT_PATH} nie istnieje")
        print("Najpierw uruchom: python scripts/download_benchmarks.py")
        sys.exit(1)

    # Backup oryginału
    if not BACKUP_PATH.exists():
        import shutil
        shutil.copy(INPUT_PATH, BACKUP_PATH)
        print(f"Utworzono backup: {BACKUP_PATH}")

    # Wczytaj zadania
    tasks = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))

    print(f"Wczytano {len(tasks)} zadań")

    # Dodaj pole audited_code dla każdego
    updated = 0
    skipped = 0
    for task in tasks:
        if "audited_code" in task and task["audited_code"]:
            skipped += 1
            continue

        # Strategia priorytetowa:
        # 1. insecure_code (jawna podatna implementacja z SecurityEval)
        # 2. source_code (jeśli istnieje)
        # 3. prompt (zawiera szablon zadania z możliwą podatnością)
        if task.get("insecure_code"):
            task["audited_code"] = task["insecure_code"]
        elif task.get("source_code"):
            task["audited_code"] = task["source_code"]
        else:
            task["audited_code"] = task.get("prompt", "")

        updated += 1

    # Zapisz
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"Zaktualizowano: {updated} zadań")
    print(f"Pominięto (już miały audited_code): {skipped}")
    print(f"Zapisano: {OUTPUT_PATH}")

    # Walidacja
    n_with_code = sum(1 for t in tasks if t.get("audited_code"))
    n_with_cwe = sum(1 for t in tasks if t.get("cwe_id"))
    print(f"\nWalidacja:")
    print(f"  Zadania z audited_code:   {n_with_code}/{len(tasks)}")
    print(f"  Zadania z cwe_id:         {n_with_cwe}/{len(tasks)}")
    print(f"  Pierwsze 3 cwe_id: {[t.get('cwe_id') for t in tasks[:3]]}")


if __name__ == "__main__":
    main()