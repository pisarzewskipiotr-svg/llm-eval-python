"""
Pilot test end-to-end:
1 zadanie × 1 model × 3 strategie (zero_shot, cot, self_refine)
= 5 wywołań API (1+1+3 dla self_refine z 2 iter)

Ce zweryfikować że cały pipeline działa od końca do końca
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import json
import platform
from datetime import datetime

from llm_eval.benchmarks.effibench import (
    load_sample_50, evaluate_generated_code, validate_code, 
    measure_execution_time, EffiBenchTask
)
from llm_eval.strategies import ZeroShotStrategy, ChainOfThoughtStrategy, SelfRefineStrategy
from llm_eval.strategies.base import StrategyState, PromptStrategy
from llm_eval.api_clients_opt import generate, GenerationResult


def run_strategy(
    strategy: PromptStrategy,
    task: EffiBenchTask,
    model_key: str,
    max_iterations: int = 1,  # dla self_refine: 0=initial + 1=refine = 2 iter
    canonical_time_ms: float = None
) -> list[dict]:
    """Wykonuje strategię (z mozliwą iteracją dla self_refine)."""
    results = []
    state = StrategyState(task=task, canonical_time_ms=canonical_time_ms)
    
    for iteration in range(max_iterations + 1):
        state.iteration = iteration
        prompt = strategy.build_prompt(state)
        
        print(f"\n  Iteration {iteration}: calling API...")
        gen_result = generate(prompt, model_key)
        
        if not gen_result.success:
            print(f"    ✗ API FAILED: {gen_result.error_type} - {gen_result.error_message}")
            results.append({
                'iteration': iteration,
                'gen_result': gen_result,
                'extracted_code': None,
                'eval_result': None,
            })
            break  # nie próbujemy iteracji jeśli initial failuje
        
        # Ekstrakcja
        code, ext_status = strategy.extract_code(gen_result.raw_response)
        print(f"    ✓ Generated: {gen_result.tokens_output} tokens, ${gen_result.cost_usd:.4f}")
        print(f"    Extraction: {ext_status}")
        
        if code is None:
            results.append({
                'iteration': iteration,
                'gen_result': gen_result,
                'extracted_code': None,
                'eval_result': None,
            })
            break
        
        # Ewaluacja
        eval_result = evaluate_generated_code(code, task, n_warmup=3, n_runs=10)
        print(f"    Status: {eval_result.validation.functional_status}")
        if eval_result.eta_efficiency is not None:
            print(f"    Eta: {eval_result.eta_efficiency:.3f} "
                  f"({eval_result.measurement.median_ms:.3f}ms vs canonical {eval_result.canonical_measurement.median_ms:.3f}ms)")
        
        results.append({
            'iteration': iteration,
            'gen_result': gen_result,
            'extracted_code': code,
            'eval_result': eval_result,
        })
        
        # Update state dla self_refine
        if iteration < max_iterations and eval_result.eta_efficiency is not None:
            state.previous_code = code
            state.previous_time_ms = eval_result.measurement.median_ms
            state.previous_functional_status = eval_result.validation.functional_status
    
    return results


def save_to_db(results, task, model_key, strategy_name, sample_idx, conn):
    """Zapisz wyniki do bazy."""
    cursor = conn.cursor()
    
    for r in results:
        gen = r['gen_result']
        ev = r['eval_result']
        
        cursor.execute("""
            INSERT OR REPLACE INTO optimization_results (
                problem_idx, task_name, model_id, strategy, iteration, sample_idx,
                canonical_time_bucket, keyword_category,
                prompt_template_version,
                raw_response, extracted_code,
                tokens_input, tokens_output, api_cost_usd, generation_duration_s,
                finish_reason, api_error_type, api_error_message,
                functional_status, validation_error,
                generated_time_median_ms, generated_time_std_ms, generated_time_min_ms,
                canonical_time_median_ms, eta_efficiency,
                n_warmup, n_measurements,
                hardware_id, python_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.problem_idx, task.task_name, model_key, strategy_name, r['iteration'], sample_idx,
            task.time_bucket, task.keyword_category,
            'v1',
            gen.raw_response, r.get('extracted_code'),
            gen.tokens_input, gen.tokens_output, gen.cost_usd, gen.duration_s,
            gen.finish_reason, gen.error_type, gen.error_message,
            ev.validation.functional_status if ev else None,
            ev.validation.error_details if ev else None,
            ev.measurement.median_ms if ev and ev.measurement.success else None,
            ev.measurement.std_ms if ev and ev.measurement.success else None,
            ev.measurement.min_ms if ev and ev.measurement.success else None,
            ev.canonical_measurement.median_ms if ev and ev.canonical_measurement and ev.canonical_measurement.success else None,
            ev.eta_efficiency if ev else None,
            3, 10,  # pilot: mniej runs
            f"local-{platform.system()}-{platform.machine()}",
            platform.python_version(),
        ))
    
    conn.commit()


def main():
    """Rozszerzony pilot: 3 modele × 5 zadań × 3 strategie."""
    tasks = load_sample_50()
    
    # Wybierz 5 zadań stratyfikowanych: po jednym z każdej kategorii FAST/MEDIUM/SLOW
    # + 2 dodatkowe średnie
    selected_tasks = [
        tasks[0],   # Wildcard Matching (MEDIUM/other)
        # Znajdź FAST i SLOW
    ]
    # Dla pilotu wybierz po jednym z różnych kategorii:
    fast_tasks = [t for t in tasks if t.time_bucket == 'FAST']
    medium_tasks = [t for t in tasks if t.time_bucket == 'MEDIUM']
    slow_tasks = [t for t in tasks if t.time_bucket == 'SLOW']
    
    selected_tasks = fast_tasks[:2] + medium_tasks[:2] + slow_tasks[:1]
    
    models = ['claude-haiku-4-5', 'gpt-3.5-turbo', 'gemini-2.5-flash']
    strategies_config = [
        ('zero_shot', ZeroShotStrategy(), 0),
        ('cot', ChainOfThoughtStrategy(), 0),
        ('self_refine', SelfRefineStrategy(), 2),
    ]
    
    conn = sqlite3.connect('results/experiment.sqlite')
    
    total_calls = len(models) * len(selected_tasks) * (1 + 1 + 3)  # zs + cot + sr(3 iter)
    estimated_cost = total_calls * 0.001  # ~$0.001 per call średnio
    print(f"Pilot scope: {len(models)} models × {len(selected_tasks)} tasks × 3 strategies")
    print(f"Estimated total API calls: {total_calls}")
    print(f"Estimated cost: ~${estimated_cost:.3f}")
    
    for task_idx, task in enumerate(selected_tasks):
        print(f"\n{'='*80}")
        print(f"TASK {task_idx+1}/{len(selected_tasks)}: {task.task_name} "
              f"(idx={task.problem_idx}, {task.time_bucket}/{task.keyword_category})")
        print(f"Canonical time: {task.canonical_time_ms:.3f}ms")
        print('='*80)
        
        for model_key in models:
            print(f"\n--- Model: {model_key} ---")
            for strategy_name, strategy, max_iter in strategies_config:
                print(f"\n  Strategy: {strategy_name}")
                results = run_strategy(strategy, task, model_key, max_iterations=max_iter,
                                       canonical_time_ms=task.canonical_time_ms)
                save_to_db(results, task, model_key, strategy_name, 0, conn)
    
    # Aggregated summary
    print("\n" + "="*80)
    print("PILOT SUMMARY: średnie eta per (model × strategia)")
    print("="*80)
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model_id, strategy, 
               COUNT(*) as n,
               AVG(eta_efficiency) as avg_eta,
               SUM(CASE WHEN functional_status = 'SUCCESS' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
        FROM optimization_results 
        WHERE iteration = 0  -- tylko initial dla porównania
        GROUP BY model_id, strategy
        ORDER BY model_id, strategy
    """)
    
    print(f"\n{'Model':25s} {'Strategy':15s} {'N':3s} {'Avg eta':10s} {'Success':8s}")
    print("-" * 65)
    for row in cursor.fetchall():
        model, strat, n, eta, succ = row
        eta_str = f"{eta:.2f}" if eta else "N/A"
        succ_str = f"{succ*100:.0f}%" if succ else "0%"
        print(f"{model:25s} {strat:15s} {n:3d} {eta_str:10s} {succ_str:8s}")
    
    conn.close()
    print("\n✓ Extended pilot complete.")


if __name__ == '__main__':
    main()