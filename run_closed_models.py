"""
Full run dla modeli zamkniętych: Claude Haiku 4.5 + GPT-3.5 Turbo.
50 zadań × 3 strategie × n=2 = 300 generacji per model.

Features:
- Resumption: pomija już wykonane (model, task, strategy, iter, sample)
- Error handling: zapisuje API errors do bazy, kontynuuje
- Progress monitoring: tqdm + lokalne logi
- Rate limiting protection: sleep między calls dla bezpieczeństwa
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import time
import platform
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

from llm_eval.benchmarks.effibench import (
    load_sample_50, evaluate_generated_code, EffiBenchTask
)
from llm_eval.strategies import ZeroShotStrategy, ChainOfThoughtStrategy, SelfRefineStrategy
from llm_eval.strategies.base import StrategyState, PromptStrategy
from llm_eval.api_clients_opt import generate, GenerationResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/full_run.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Konfiguracja ===
MODELS = ['claude-haiku-4-5']
N_SAMPLES = 2
N_WARMUP = 5
N_MEASUREMENTS = 15  # finalne wartości dla full runu
MAX_REFINE_ITERATIONS = 2  # initial + 2 = 3 generacje total dla self_refine
SLEEP_BETWEEN_CALLS = 0.5  # sekund, throttle dla bezpieczeństwa
DB_PATH = 'results/experiment.sqlite'


def already_done(conn, problem_idx, model_id, strategy, iteration, sample_idx) -> bool:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT functional_status, api_error_type FROM optimization_results
        WHERE problem_idx = ? AND model_id = ? AND strategy = ? 
              AND iteration = ? AND sample_idx = ?
    """, (problem_idx, model_id, strategy, iteration, sample_idx))
    row = cursor.fetchone()
    if row is None:
        return False
    status, api_err = row
    # API_FAILED - traktuj jako NIE-done, żeby ponowić
    if status == 'API_FAILED' or api_err is not None:
        return False
    return status is not None


def run_strategy(
    strategy: PromptStrategy,
    task: EffiBenchTask,
    model_key: str,
    sample_idx: int,
    max_iterations: int,
    canonical_time_ms: float,
    conn
) -> None:
    """Wykonuje strategię i zapisuje wyniki do bazy."""
    state = StrategyState(task=task, canonical_time_ms=canonical_time_ms)
    
    for iteration in range(max_iterations + 1):
        # Skip jeśli już done
        if already_done(conn, task.problem_idx, model_key, strategy.strategy_name, 
                       iteration, sample_idx):
            # Załaduj poprzedni kod dla self_refine continuation
            if iteration < max_iterations and strategy.strategy_name == 'self_refine':
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT extracted_code, generated_time_median_ms, functional_status
                    FROM optimization_results
                    WHERE problem_idx=? AND model_id=? AND strategy=? AND iteration=? AND sample_idx=?
                """, (task.problem_idx, model_key, strategy.strategy_name, iteration, sample_idx))
                row = cursor.fetchone()
                if row and row[0]:
                    state.previous_code = row[0]
                    state.previous_time_ms = row[1]
                    state.previous_functional_status = row[2]
            continue
        
        state.iteration = iteration
        
        try:
            prompt = strategy.build_prompt(state)
        except ValueError as e:
            # Self-Refine bez previous code  pomijamy iterację
            logger.warning(f"Skip iter {iteration} for {task.problem_idx}: {e}")
            return
        
        # API call
        time.sleep(SLEEP_BETWEEN_CALLS)
        gen_result = generate(prompt, model_key)
        
        if not gen_result.success:
            # Zapisz error do bazy
            save_failed_generation(conn, task, model_key, strategy.strategy_name,
                                   iteration, sample_idx, gen_result)
            logger.warning(f"  API FAILED iter={iteration}: {gen_result.error_type}: {gen_result.error_message}")
            return  # nie próbujemy kolejnych iteracji
        
        # Ekstrakcja
        code, ext_status = strategy.extract_code(gen_result.raw_response)
        
        if code is None:
            save_extraction_failed(conn, task, model_key, strategy.strategy_name,
                                  iteration, sample_idx, gen_result, ext_status)
            logger.warning(f"  EXTRACTION FAILED iter={iteration}: {ext_status}")
            return
        
        # Ewaluacja (walidacja + pomiar)
        eval_result = evaluate_generated_code(
            code, task, n_warmup=N_WARMUP, n_runs=N_MEASUREMENTS
        )
        
        # Zapisz pełny wynik
        save_full_result(conn, task, model_key, strategy.strategy_name,
                        iteration, sample_idx, gen_result, code, ext_status, eval_result)
        
        # Update state dla self_refine
        if iteration < max_iterations and eval_result.measurement.success:
            state.previous_code = code
            state.previous_time_ms = eval_result.measurement.median_ms
            state.previous_functional_status = eval_result.validation.functional_status
        elif iteration < max_iterations:
            # Nie mamy pomiaru — nie możemy kontynuować self_refine
            logger.warning(f"  Cannot continue self_refine: no valid measurement")
            return


def save_full_result(conn, task, model_key, strategy_name, iteration, sample_idx,
                    gen_result, code, ext_status, eval_result):
    """Zapisz pełny wynik (sukces lub fail walidacji)."""
    ev = eval_result
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO optimization_results (
            problem_idx, task_name, model_id, strategy, iteration, sample_idx,
            canonical_time_bucket, keyword_category,
            prompt_template_version,
            raw_response, extracted_code, extraction_status,
            tokens_input, tokens_output, api_cost_usd, generation_duration_s,
            finish_reason, api_error_type, api_error_message,
            functional_status, validation_error,
            generated_time_median_ms, generated_time_std_ms, generated_time_min_ms,
            canonical_time_median_ms, eta_efficiency,
            n_warmup, n_measurements,
            hardware_id, python_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        task.problem_idx, task.task_name, model_key, strategy_name, iteration, sample_idx,
        task.time_bucket, task.keyword_category,
        'v1',
        gen_result.raw_response, code, ext_status,
        gen_result.tokens_input, gen_result.tokens_output, gen_result.cost_usd, gen_result.duration_s,
        gen_result.finish_reason, gen_result.error_type, gen_result.error_message,
        ev.validation.functional_status,
        ev.validation.error_details,
        ev.measurement.median_ms if ev.measurement.success else None,
        ev.measurement.std_ms if ev.measurement.success else None,
        ev.measurement.min_ms if ev.measurement.success else None,
        ev.canonical_measurement.median_ms if ev.canonical_measurement and ev.canonical_measurement.success else None,
        ev.eta_efficiency,
        N_WARMUP, N_MEASUREMENTS,
        f"local-{platform.system()}-{platform.machine()}",
        platform.python_version(),
    ))
    conn.commit()


def save_failed_generation(conn, task, model_key, strategy_name, iteration, sample_idx, gen_result):
    """Zapisz fail API."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO optimization_results (
            problem_idx, task_name, model_id, strategy, iteration, sample_idx,
            canonical_time_bucket, keyword_category,
            prompt_template_version,
            raw_response, tokens_input, tokens_output, generation_duration_s,
            api_error_type, api_error_message,
            functional_status,
            hardware_id, python_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        task.problem_idx, task.task_name, model_key, strategy_name, iteration, sample_idx,
        task.time_bucket, task.keyword_category, 'v1',
        '', gen_result.tokens_input, gen_result.tokens_output, gen_result.duration_s,
        gen_result.error_type, gen_result.error_message,
        'API_FAILED',
        f"local-{platform.system()}-{platform.machine()}", platform.python_version(),
    ))
    conn.commit()


def save_extraction_failed(conn, task, model_key, strategy_name, iteration, sample_idx, gen_result, ext_status):
    """Zapisz fail ekstrakcji kodu."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO optimization_results (
            problem_idx, task_name, model_id, strategy, iteration, sample_idx,
            canonical_time_bucket, keyword_category,
            prompt_template_version,
            raw_response, extraction_status,
            tokens_input, tokens_output, api_cost_usd, generation_duration_s,
            finish_reason, functional_status,
            hardware_id, python_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        task.problem_idx, task.task_name, model_key, strategy_name, iteration, sample_idx,
        task.time_bucket, task.keyword_category, 'v1',
        gen_result.raw_response, ext_status,
        gen_result.tokens_input, gen_result.tokens_output, gen_result.cost_usd, gen_result.duration_s,
        gen_result.finish_reason, 'EXTRACTION_FAILED',
        f"local-{platform.system()}-{platform.machine()}", platform.python_version(),
    ))
    conn.commit()


def main():
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("FULL RUN: Closed Models (Claude Haiku 4.5 + GPT-3.5 Turbo)")
    logger.info("=" * 80)
    
    tasks = load_sample_50()
    conn = sqlite3.connect(DB_PATH)
    
    strategies_config = [
        ('zero_shot', ZeroShotStrategy(), 0),
        ('cot', ChainOfThoughtStrategy(), 0),
        ('self_refine', SelfRefineStrategy(), MAX_REFINE_ITERATIONS),
    ]
    
    # Oblicz całkowitą liczbę kroków
    total_steps = 0
    for _ in MODELS:
        for _, _, max_iter in strategies_config:
            total_steps += len(tasks) * N_SAMPLES * (max_iter + 1)
    
    logger.info(f"Total combinations: {total_steps}")
    logger.info(f"Estimated time: {total_steps * 8 / 60:.1f} minutes")
    logger.info(f"Estimated cost: ~${total_steps * 0.0015:.2f}")
    
    start_time = time.time()
    pbar = tqdm(total=total_steps, desc="Generating", unit="call")
    
    for model_key in MODELS:
        logger.info(f"\n{'='*80}\nMODEL: {model_key}\n{'='*80}")
        
        for task_idx, task in enumerate(tasks):
            logger.info(f"\n[{task_idx+1}/{len(tasks)}] Task: {task.task_name} "
                       f"(idx={task.problem_idx}, {task.time_bucket}/{task.keyword_category})")
            
            for strategy_name, strategy, max_iter in strategies_config:
                for sample_idx in range(N_SAMPLES):
                    run_strategy(
                        strategy, task, model_key, sample_idx,
                        max_iterations=max_iter,
                        canonical_time_ms=task.canonical_time_ms,
                        conn=conn
                    )
                    pbar.update(max_iter + 1)
        
        # Statystyki per model po zakończeniu
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*), 
                   SUM(CASE WHEN functional_status = 'SUCCESS' THEN 1 ELSE 0 END),
                   AVG(eta_efficiency),
                   SUM(api_cost_usd)
            FROM optimization_results WHERE model_id = ?
        """, (model_key,))
        n, n_succ, avg_eta, total_cost = cursor.fetchone()
        logger.info(f"\n=== {model_key} stats ===")
        logger.info(f"  Total: {n}, Success: {n_succ} ({100*n_succ/n if n else 0:.1f}%)")
        logger.info(f"  Avg eta: {avg_eta:.3f}" if avg_eta else "  Avg eta: N/A")
        logger.info(f"  Total cost: ${total_cost:.2f}" if total_cost else "  Total cost: $0")
    
    pbar.close()
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPLETED in {timedelta(seconds=int(elapsed))}")
    logger.info(f"{'='*80}")
    
    conn.close()


if __name__ == '__main__':
    main()