"""
Schema bazy SQLite dla eksperymentu + funkcje pomocnicze.

Baza przechowuje wszystkie generacje, ich ewaluacje i metryki.
Schema jest zaprojektowany pod 3 eksperymenty (generowanie + audyt).
"""
import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Optional


SCHEMA = """
-- =============================================================================
-- TABELA: generations
-- Przechowuje pojedyncze wywołania modeli (jedna generacja = jeden wiersz)
-- =============================================================================
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Identyfikacja eksperymentu
    experiment_type TEXT NOT NULL,        -- 'generation' | 'audit_generator' | 'audit_auditor'
    benchmark TEXT NOT NULL,              -- 'humaneval_plus' | 'livecodebench' | 'securityeval'
    task_id TEXT NOT NULL,                -- ID zadania w benchmarku
    model_name TEXT NOT NULL,             -- nazwa modelu z config.MODELS
    sample_idx INTEGER NOT NULL,          -- indeks próbki (0..n-1)

    -- Parametry generowania
    temperature REAL NOT NULL,
    top_p REAL,
    max_tokens INTEGER,
    seed INTEGER,                         -- jeśli kontrolowany (OSS)

    -- Wynik generowania
    raw_response TEXT NOT NULL,           -- pełna odpowiedź modelu
    extracted_code TEXT,                  -- wyodrębniony kod Pythona
    extraction_status TEXT NOT NULL,      -- 'success' | 'no_code_block' | 'error'
    generation_time_sec REAL,
    tokens_input INTEGER,
    tokens_output INTEGER,

    -- Metadane
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,                   -- jeśli generation_status != 'success'

    UNIQUE(experiment_type, benchmark, task_id, model_name, sample_idx)
);

CREATE INDEX IF NOT EXISTS idx_gen_lookup ON generations(experiment_type, benchmark, model_name);
CREATE INDEX IF NOT EXISTS idx_gen_task ON generations(task_id);


-- =============================================================================
-- TABELA: test_results
-- Wyniki wykonania testów dla wygenerowanego kodu (filar generowania)
-- =============================================================================
CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL REFERENCES generations(id),

    -- Wynik
    passed BOOLEAN NOT NULL,              -- czy wszystkie testy przeszły
    n_tests_total INTEGER,
    n_tests_passed INTEGER,
    test_pass_rate REAL,                  -- n_passed / n_total

    -- Wykonanie
    execution_time_sec REAL,              -- czas wykonania kodu
    memory_peak_mb REAL,                  -- szczytowe zużycie pamięci
    timeout BOOLEAN DEFAULT FALSE,
    error_type TEXT,                      -- 'syntax' | 'runtime' | 'assertion' | 'timeout' | 'other'
    error_message TEXT,

    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(generation_id)
);

CREATE INDEX IF NOT EXISTS idx_test_gen ON test_results(generation_id);


-- =============================================================================
-- TABELA: quality_metrics
-- Metryki jakości statycznej kodu (Radon, Ruff)
-- =============================================================================
CREATE TABLE IF NOT EXISTS quality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL REFERENCES generations(id),

    -- Złożoność cyklomatyczna (Radon)
    cyclomatic_complexity_avg REAL,
    cyclomatic_complexity_max REAL,

    -- Halstead (Radon)
    halstead_volume REAL,
    halstead_difficulty REAL,
    halstead_effort REAL,

    -- Maintainability Index (Radon)
    maintainability_index REAL,

    -- PEP 8 (Ruff)
    pep8_violations_total INTEGER,
    pep8_violations_naming INTEGER,
    pep8_violations_formatting INTEGER,
    pep8_violations_imports INTEGER,

    -- Inne
    lines_of_code INTEGER,
    n_functions INTEGER,

    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(generation_id)
);

CREATE INDEX IF NOT EXISTS idx_quality_gen ON quality_metrics(generation_id);


-- =============================================================================
-- TABELA: security_findings
-- Wykrycia podatności (Bandit) - jeden wiersz per wykrycie
-- =============================================================================
CREATE TABLE IF NOT EXISTS security_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL REFERENCES generations(id),

    -- Identyfikacja podatności
    cwe_id TEXT,                          -- np. 'CWE-78'
    test_id TEXT,                         -- np. 'B602' dla Bandit
    severity TEXT,                        -- 'LOW' | 'MEDIUM' | 'HIGH'
    confidence TEXT,                      -- 'LOW' | 'MEDIUM' | 'HIGH'

    -- Lokalizacja
    line_number INTEGER,
    code_snippet TEXT,

    -- Opis
    issue_text TEXT,
    tool TEXT NOT NULL,                   -- 'bandit' | 'semgrep' (jeśli dodany)

    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_security_gen ON security_findings(generation_id);
CREATE INDEX IF NOT EXISTS idx_security_cwe ON security_findings(cwe_id);


-- =============================================================================
-- TABELA: audit_responses
-- Odpowiedzi modeli w roli audytora (eksperyment audytu)
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL REFERENCES generations(id),

    -- Co model audytował
    audited_code TEXT NOT NULL,
    ground_truth_cwe TEXT,                -- znana podatność (z SecurityEval)

    -- Co model odpowiedział
    detected_vulnerability BOOLEAN,       -- model zidentyfikował podatność?
    detected_cwe TEXT,                    -- jaka kategoria CWE?
    confidence_score REAL,                -- jeśli model wyraził pewność

    -- Klasyfikacja TP/FP/TN/FN
    classification TEXT,                  -- 'TP' | 'FP' | 'TN' | 'FN'

    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(generation_id)
);

CREATE INDEX IF NOT EXISTS idx_audit_gen ON audit_responses(generation_id);


-- =============================================================================
-- TABELA: experiment_meta
-- Metadane przebiegu eksperymentu (kiedy uruchomiony, parametry, etc.)
-- =============================================================================
CREATE TABLE IF NOT EXISTS experiment_meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    config_json TEXT NOT NULL,            -- JSON z config snapshot
    git_commit TEXT,
    notes TEXT
);
"""


def init_db(db_path: Path) -> None:
    """Inicjalizuje bazę SQLite ze schematem."""
    db_path.parent.mkdir(exist_ok=True, parents=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()
    print(f"Baza zainicjalizowana: {db_path}")


@contextmanager
def get_db(db_path: Path):
    """Context manager dla połączenia z bazą."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_generation(
    db_path: Path,
    experiment_type: str,
    benchmark: str,
    task_id: str,
    model_name: str,
    sample_idx: int,
    temperature: float,
    raw_response: str,
    extracted_code: Optional[str],
    extraction_status: str,
    generation_time_sec: float,
    tokens_input: Optional[int] = None,
    tokens_output: Optional[int] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    error_message: Optional[str] = None,
) -> int:
    """Wstawia rekord generowania, zwraca ID."""
    with get_db(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO generations
            (experiment_type, benchmark, task_id, model_name, sample_idx,
             temperature, top_p, max_tokens, seed,
             raw_response, extracted_code, extraction_status,
             generation_time_sec, tokens_input, tokens_output, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_type, benchmark, task_id, model_name, sample_idx,
                temperature, top_p, max_tokens, seed,
                raw_response, extracted_code, extraction_status,
                generation_time_sec, tokens_input, tokens_output, error_message,
            ),
        )
        return cursor.lastrowid


def already_generated(
    db_path: Path,
    experiment_type: str,
    benchmark: str,
    task_id: str,
    model_name: str,
    sample_idx: int,
) -> bool:
    """Sprawdza, czy ta kombinacja była już wygenerowana (do wznawiania)."""
    with get_db(db_path) as conn:
        result = conn.execute(
            """
            SELECT 1 FROM generations
            WHERE experiment_type=? AND benchmark=? AND task_id=?
                  AND model_name=? AND sample_idx=?
            """,
            (experiment_type, benchmark, task_id, model_name, sample_idx),
        ).fetchone()
        return result is not None


if __name__ == "__main__":
    # Test: inicjalizacja bazy
    from config import DB_PATH
    init_db(DB_PATH)
    print("OK")
