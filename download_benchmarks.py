"""
Pobieranie benchmarków eksperymentu
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Dodaj parent dir do PATH dla importu config
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import DATA_DIR


def download_humaneval_plus():
    """
    HumanEval+ przez evalplus (pip install).
    Format wyjściowy: data/humaneval_plus_tasks.jsonl
    """
    print("=" * 60)
    print("Pobieranie HumanEval+ (przez evalplus)")
    print("=" * 60)

    try:
        from evalplus.data import get_human_eval_plus
    except ImportError:
        print("BŁĄD: evalplus nie jest zainstalowany.")
        print("Uruchom: pip install evalplus==0.3.1")
        sys.exit(1)

    tasks = get_human_eval_plus()
    output_path = DATA_DIR / "humaneval_plus_tasks.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for task_id, task_data in tasks.items():
            standardized = {
                "benchmark": "humaneval_plus",
                "task_id": task_id,
                "prompt": task_data["prompt"],
                "entry_point": task_data["entry_point"],
                "canonical_solution": task_data["canonical_solution"],
                "test": task_data["test"],
                "test_plus": task_data.get("test", ""),
                "raw": task_data,
            }
            f.write(json.dumps(standardized, ensure_ascii=False) + "\n")

    print(f"Zapisano {len(tasks)} zadań do: {output_path}")
    return len(tasks)


def download_livecodebench():
    """
    LiveCodeBench - pobranie z HuggingFace przez huggingface_hub.

    Pomija load_dataset (które w nowych wersjach 'datasets' blokuje
    loading scripts). Zamiast tego pobiera bezpośrednio pliki JSONL/parquet.

    Wybiera 50 zadań z 2024+ (contamination-free dla naszych modeli).
    """
    print("=" * 60)
    print("Pobieranie LiveCodeBench (przez huggingface_hub)")
    print("=" * 60)

    try:
        from huggingface_hub import snapshot_download, list_repo_files
    except ImportError:
        print("BŁĄD: huggingface_hub nie zainstalowany.")
        print("Uruchom: pip install huggingface_hub")
        return 0

    # Pobranie tylko plików JSONL z datasetu (nie cały repo, żeby nie pobierać skryptów)
    cache_dir = DATA_DIR / "livecodebench_cache"
    cache_dir.mkdir(exist_ok=True)

    print("Listowanie plików w datasecie...")
    try:
        files = list_repo_files(
            "livecodebench/code_generation_lite",
            repo_type="dataset",
        )
    except Exception as e:
        print(f"BŁĄD przy listowaniu plików HuggingFace: {e}")
        print("Sprawdź połączenie sieciowe.")
        return 0

    # Szukamy plików JSONL z zadaniami (test*.jsonl)
    jsonl_files = [f for f in files if f.endswith(".jsonl") and "test" in f.lower()]
    print(f"Znaleziono {len(jsonl_files)} plików JSONL: {jsonl_files[:5]}...")

    if not jsonl_files:
        # Fallback: spróbuj parquet
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if parquet_files:
            print(f"Brak JSONL, używam parquet: {parquet_files[:5]}")
            return _download_livecodebench_parquet(parquet_files, cache_dir)
        print("BŁĄD: nie znaleziono plików z zadaniami w datasecie")
        print("Wszystkie pliki:", files[:20])
        return 0

    # Pobierz pliki JSONL (z najnowszej wersji - sortowanie alfabetyczne, ostatni release)
    jsonl_files.sort()
    target_file = jsonl_files[-1]  # najnowsza wersja
    print(f"Pobieranie: {target_file}")

    try:
        local_path = snapshot_download(
            "livecodebench/code_generation_lite",
            repo_type="dataset",
            allow_patterns=[target_file],
            cache_dir=str(cache_dir),
        )
    except Exception as e:
        print(f"BŁĄD przy pobieraniu pliku: {e}")
        return 0

    target_jsonl = Path(local_path) / target_file
    if not target_jsonl.exists():
        # snapshot_download może zwrócić inną strukturę
        target_jsonl = next(Path(local_path).rglob(Path(target_file).name), None)

    if not target_jsonl or not target_jsonl.exists():
        print(f"BŁĄD: nie znaleziono pobranego pliku")
        return 0

    print(f"Pobrany plik: {target_jsonl}")

    # Wczytaj i przefiltruj zadania
    cutoff = datetime(2024, 1, 1)
    all_tasks = []
    with open(target_jsonl, encoding="utf-8") as f:
        for line in f:
            try:
                task = json.loads(line)
                all_tasks.append(task)
            except json.JSONDecodeError:
                continue

    print(f"Wczytano {len(all_tasks)} zadań z pliku")

    # Filtrowanie - LiveCodeBench ma pole 'contest_date' lub 'release_date'
    selected = []
    for task in all_tasks:
        date_str = task.get("contest_date") or task.get("release_date") or ""
        try:
            if isinstance(date_str, str) and "T" in date_str:
                date_str = date_str.split("T")[0]
            if date_str:
                d = datetime.fromisoformat(date_str)
                if d >= cutoff:
                    selected.append(task)
            else:
                # Brak daty - dodajemy (lepiej mieć więcej kandydatów)
                selected.append(task)
        except (ValueError, TypeError):
            continue

    print(f"Po filtrze daty (>= 2024): {len(selected)} zadań")

    if len(selected) < 50:
        print(f"UWAGA: tylko {len(selected)} zadań dostępnych - biorę wszystkie")

    # Próbka 50 zadań (lub mniej, jeśli brak)
    import random
    random.seed(42)
    if len(selected) > 50:
        selected = random.sample(selected, 50)

    # Zapis w jednolitym formacie
    output_path = DATA_DIR / "livecodebench_tasks.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for task in selected:
            standardized = {
                "benchmark": "livecodebench",
                "task_id": task.get("question_id", task.get("task_id", "unknown")),
                "prompt": task.get("question_content", task.get("prompt", "")),
                "entry_point": task.get("starter_code", ""),
                "test": task.get("public_test_cases", task.get("test", "")),
                "test_plus": task.get("private_test_cases", ""),
                "difficulty": task.get("difficulty", "unknown"),
                "contest_date": task.get("contest_date", task.get("release_date", "")),
                "raw": dict(task),
            }
            f.write(json.dumps(standardized, ensure_ascii=False, default=str) + "\n")

    print(f"Zapisano {len(selected)} zadań do: {output_path}")
    return len(selected)


def _download_livecodebench_parquet(parquet_files, cache_dir):
    """Fallback - pobranie parquet zamiast JSONL."""
    from huggingface_hub import snapshot_download
    import pandas as pd

    target = sorted(parquet_files)[-1]
    print(f"Pobieranie parquet: {target}")

    local_path = snapshot_download(
        "livecodebench/code_generation_lite",
        repo_type="dataset",
        allow_patterns=[target],
        cache_dir=str(cache_dir),
    )

    target_file = next(Path(local_path).rglob(Path(target).name), None)
    if not target_file:
        return 0

    df = pd.read_parquet(target_file)
    print(f"Wczytano {len(df)} zadań z parquet")

    # Filtrowanie i zapis
    output_path = DATA_DIR / "livecodebench_tasks.jsonl"
    selected = df.head(50)  # uproszczona selekcja

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in selected.iterrows():
            task = row.to_dict()
            standardized = {
                "benchmark": "livecodebench",
                "task_id": str(task.get("question_id", task.get("task_id", "unknown"))),
                "prompt": str(task.get("question_content", task.get("prompt", ""))),
                "entry_point": str(task.get("starter_code", "")),
                "test": str(task.get("public_test_cases", "")),
                "test_plus": str(task.get("private_test_cases", "")),
                "difficulty": str(task.get("difficulty", "unknown")),
                "contest_date": str(task.get("contest_date", "")),
            }
            f.write(json.dumps(standardized, ensure_ascii=False) + "\n")

    print(f"Zapisano {len(selected)} zadań do: {output_path}")
    return len(selected)


def download_securityeval():
    """
    SecurityEval - wczytanie z dataset.jsonl (rzeczywista struktura repo).

    Repozytorium s2e-lab/SecurityEval ma plik dataset.jsonl w głównym katalogu,
    NIE katalog dataset/CWE-*.py jak pierwotnie zakładałem.
    """
    print("=" * 60)
    print("Pobieranie SecurityEval (z dataset.jsonl)")
    print("=" * 60)

    securityeval_dir = DATA_DIR / "SecurityEval"

    if not securityeval_dir.exists():
        print(f"BŁĄD: Katalog {securityeval_dir} nie istnieje.")
        print("Wykonaj: cd data && git clone https://github.com/s2e-lab/SecurityEval")
        return 0

    # Plik dataset.jsonl w głównym katalogu repo
    dataset_file = securityeval_dir / "dataset.jsonl"
    if not dataset_file.exists():
        print(f"BŁĄD: {dataset_file} nie istnieje.")
        print("Sprawdź zawartość katalogu SecurityEval - być może struktura repo się zmieniła")
        # Listing dla debugowania
        contents = list(securityeval_dir.iterdir())
        print(f"Zawartość {securityeval_dir}:")
        for item in contents[:20]:
            print(f"  {item.name}")
        return 0

    # Wczytaj zadania z JSONL
    tasks = []
    with open(dataset_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_task = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"OSTRZEŻENIE: błąd parsowania linii: {e}")
                continue

            # Standaryzacja - SecurityEval format
            # Typowe pola: 'ID', 'Prompt', 'Insecure_code', 'Source_code'
            task = {
                "benchmark": "securityeval",
                "task_id": raw_task.get("ID", raw_task.get("id", f"task_{len(tasks)}")),
                "prompt": raw_task.get("Prompt", raw_task.get("prompt", "")),
                "cwe_id": _extract_cwe_from_id(raw_task.get("ID", "")),
                "entry_point": "",
                "insecure_code": raw_task.get("Insecure_code", ""),
                "source_code": raw_task.get("Source_code", ""),
                "raw": raw_task,
            }
            tasks.append(task)

    output_path = DATA_DIR / "securityeval_tasks.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"Zapisano {len(tasks)} zadań do: {output_path}")
    return len(tasks)


def _extract_cwe_from_id(task_id: str) -> str:
    """
    Wyciąga CWE-XXX z task_id.

    Format SecurityEval ID: typowo "CWE-78_subtypename_NN" lub podobny.
    """
    if not task_id:
        return ""

    # Szukamy wzorca CWE-XXX
    import re
    match = re.search(r"CWE-?(\d+)", task_id, re.IGNORECASE)
    if match:
        return f"CWE-{match.group(1)}"
    return ""


def main():
    """Pobiera wszystkie benchmarki."""
    print("\n" + "#" * 60)
    print("# POBIERANIE BENCHMARKÓW EKSPERYMENTU")
    print("#" * 60 + "\n")

    n_he = download_humaneval_plus()
    print()

    n_lcb = download_livecodebench()
    print()

    n_se = download_securityeval()
    print()

    print("=" * 60)
    print("PODSUMOWANIE:")
    print(f"  HumanEval+:    {n_he} zadań")
    print(f"  LiveCodeBench: {n_lcb} zadań")
    print(f"  SecurityEval:  {n_se} zadań")
    print(f"  RAZEM:         {n_he + n_lcb + n_se} zadań")
    print("=" * 60)


if __name__ == "__main__":
    main()