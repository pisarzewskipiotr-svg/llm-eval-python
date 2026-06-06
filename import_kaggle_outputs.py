"""
Import Kaggle outputs do lokalnej bazy + wykonanie pomiarów wydajności LOKALNIE.

Kaggle wykonuje tylko generacje (GPU shared = noisy timing).
Pomiary wykonujemy lokalnie (CPU - stabilne, single-process).

Workflow:
1. Wczytaj outputs_*.jsonl
2. Dla każdej generacji:
   - Wczytaj extracted_code
   - Walidacja: czy przechodzi small_test_cases
   - Pomiar: warm-up + 15 runs + median (jak w effibench.py)
   - Pomiar canonical_solution na tej samej maszynie
   - Oblicz eta
3. Zapisz do optimization_results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import sqlite3
import time
import platform
import logging
from datetime import datetime
from tqdm import tqdm

from llm_eval.benchmarks.effibench import (
    load_sample_50, evaluate_generated_code, EffiBenchTask,
    validate_code, measure_execution_time
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

KAGGLE_OUTPUTS_DIR = Path(r'J:\Praca wyniki\external\kaggle_outputs')
DB_PATH = 'results/experiment.sqlite'

# Pomiary lokalne — finalne wartości
N_WARMUP = 5
N_MEASUREMENTS = 15


def normalize_model_id(raw_id: str) -> str:
    """Maps Kaggle full names to our DB short names."""
    mapping = {
        "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen2.5-coder-7b",
        "deepseek-ai/deepseek-coder-6.7b-instruct": "deepseek-coder-6.7b",
    }
    return mapping.get(raw_id, raw_id)


def already_done(conn, problem_idx, model_id, strategy, iteration, sample_idx):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT functional_status, api_error_type FROM optimization_results
        WHERE problem_idx=? AND model_id=? AND strategy=? AND iteration=? AND sample_idx=?
    """, (problem_idx, model_id, strategy, iteration, sample_idx))
    row = cursor.fetchone()
    if row is None:
        return False
    status, api_err = row
    if status == 'API_FAILED' or api_err is not None:
        return False
    return status is not None


def import_file(jsonl_path: Path, conn):
    """Wczytuje JSONL z Kaggle, mierzy lokalnie, zapisuje do bazy."""
    
    # Załaduj tasks (mamy metadata: time_bucket, keyword_category)
    tasks = {t.problem_idx: t for t in load_sample_50()}
    
    # Wczytaj wszystkie generacje
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        generations = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(generations)} generations from {jsonl_path.name}")
    
    # Filtruj te które mają extracted_code (część może mieć tylko 'error')
    valid_gens = [g for g in generations if g.get('extracted_code')]
    logger.info(f"Valid (with extracted_code): {len(valid_gens)}")
    
    pbar = tqdm(valid_gens, desc="Measuring", unit="gen")
    
    stats = {'imported': 0, 'skipped': 0, 'measure_failed': 0, 'validation_failed': 0}
    
    for gen in pbar:
        problem_idx = gen['problem_idx']
        model_id = normalize_model_id(gen['model_id'])
        strategy = gen['strategy']
        iteration = gen['iteration']
        sample_idx = gen['sample_idx']
        
        # Skip jeśli już done
        if already_done(conn, problem_idx, model_id, strategy, iteration, sample_idx):
            stats['skipped'] += 1
            continue
        
        # Znajdź task
        if problem_idx not in tasks:
            logger.warning(f"Task {problem_idx} not in sample_50, skipping")
            continue
        task = tasks[problem_idx]
        
        code = gen['extracted_code']
        if not code:
            stats['validation_failed'] += 1
            continue
        
        # Walidacja + pomiar lokalnie
        try:
            eval_result = evaluate_generated_code(code, task, 
                                                  n_warmup=N_WARMUP, 
                                                  n_runs=N_MEASUREMENTS)
        except Exception as e:
            logger.warning(f"Measure failed for task {problem_idx}, {strategy}/{iteration}/{sample_idx}: {e}")
            stats['measure_failed'] += 1
            continue
        
        # Zapisz do bazy
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO optimization_results (
                problem_idx, task_name, model_id, strategy, iteration, sample_idx,
                canonical_time_bucket, keyword_category,
                prompt_template_version,
                raw_response, extracted_code, extraction_status,
                tokens_input, tokens_output, api_cost_usd, generation_duration_s,
                functional_status, validation_error,
                generated_time_median_ms, generated_time_std_ms, generated_time_min_ms,
                canonical_time_median_ms, eta_efficiency,
                n_warmup, n_measurements,
                hardware_id, python_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            problem_idx, task.task_name, model_id, strategy, iteration, sample_idx,
            task.time_bucket, task.keyword_category,
            'v1',  # template version
            gen.get('raw_response', ''),
            code, 'success',
            gen.get('tokens_in', 0), gen.get('tokens_out', 0), 0.0,  # cost=0 (Kaggle local)
            0.0,  # generation duration not tracked for Kaggle
            eval_result.validation.functional_status,
            eval_result.validation.error_details,
            eval_result.measurement.median_ms if eval_result.measurement.success else None,
            eval_result.measurement.std_ms if eval_result.measurement.success else None,
            eval_result.measurement.min_ms if eval_result.measurement.success else None,
            eval_result.canonical_measurement.median_ms if eval_result.canonical_measurement and eval_result.canonical_measurement.success else None,
            eval_result.eta_efficiency,
            N_WARMUP, N_MEASUREMENTS,
            f"kaggle-t4-gen+local-{platform.system()}-measure",
            platform.python_version(),
        ))
        conn.commit()
        
        stats['imported'] += 1
        
        # Update progress
        pbar.set_postfix({
            'imported': stats['imported'],
            'status': eval_result.validation.functional_status
        })
    
    return stats


def main():
    if not KAGGLE_OUTPUTS_DIR.exists():
        logger.error(f"Directory not found: {KAGGLE_OUTPUTS_DIR}")
        logger.error("Create it and place outputs_*.jsonl files there")
        return
    
    jsonl_files = list(KAGGLE_OUTPUTS_DIR.glob("outputs_*.jsonl"))
    if not jsonl_files:
        logger.error(f"No outputs_*.jsonl files in {KAGGLE_OUTPUTS_DIR}")
        return
    
    logger.info(f"Found {len(jsonl_files)} output files:")
    for f in jsonl_files:
        logger.info(f"  - {f.name}")
    
    conn = sqlite3.connect(DB_PATH)
    
    total_stats = {'imported': 0, 'skipped': 0, 'measure_failed': 0, 'validation_failed': 0}
    
    for f in jsonl_files:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {f.name}")
        logger.info(f"{'='*70}")
        stats = import_file(f, conn)
        for k, v in stats.items():
            total_stats[k] += v
    
    # Podsumowanie
    logger.info(f"\n{'='*70}")
    logger.info("IMPORT COMPLETE")
    logger.info(f"{'='*70}")
    for k, v in total_stats.items():
        logger.info(f"  {k}: {v}")
    
    # Statystyki per model
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model_id, COUNT(*) as n,
               SUM(CASE WHEN functional_status = 'SUCCESS' THEN 1 ELSE 0 END) as ok,
               AVG(eta_efficiency) as avg_eta
        FROM optimization_results
        GROUP BY model_id
        ORDER BY model_id
    """)
    print(f"\n{'Model':25s} {'N':5s} {'OK':5s} {'Success%':10s} {'Avg eta':10s}")
    print('-' * 65)
    for row in cursor.fetchall():
        model, n, ok, eta = row
        succ_pct = f"{100*ok/n:.1f}%" if n else "0%"
        eta_str = f"{eta:.2f}" if eta else "N/A"
        print(f"{model:25s} {n:5d} {ok:5d} {succ_pct:10s} {eta_str:10s}")
    
    conn.close()


if __name__ == '__main__':
    main()