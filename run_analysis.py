"""
Pełna analiza statystyczna modułu optymalizacji EffiBench.

Wykonuje:
- Statystyki deskryptywne per (model, strategy, iteration)
- H5: Wilcoxon CoT vs zero-shot per model
- H6: Wilcoxon ΔS₀₁ i ΔS₁₂ per model (jednostronne)
- H7: Kendall τ pass@1 vs regression rate
- Bootstrap CI dla η per (model, strategy)
- Macierz po taksonomii (bucket × strategia)
- Friedman test dla porównania między-modelowego

Wymaga: scipy, numpy. Zainstaluj jeśli brak: pip install scipy numpy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import json
import numpy as np
from scipy import stats
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


DB_PATH = 'results/experiment.sqlite'
ALPHA = 0.05

# Baseline pass@1 z HumanEval+ — KONIECZNIE zaktualizuj te wartości
# bazując na wynikach z Twojego rozdziału HumanEval+
PASS_AT_1_HUMANEVAL = {
    'claude-haiku-4-5': 0.938,    # zaktualizuj!
    'gpt-3.5-turbo': 0.699,        # zaktualizuj!
    'qwen2.5-coder-7b': 0.865,    # zaktualizuj!
}


def fetch_etas(conn, model_id, strategy, iteration=0):
    """Pobierz wektor eta dla model × strategia × iteration."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT eta_efficiency
        FROM optimization_results
        WHERE model_id = ? AND strategy = ? AND iteration = ?
        AND eta_efficiency IS NOT NULL
        AND functional_status = 'SUCCESS'
    """, (model_id, strategy, iteration))
    return np.array([r[0] for r in cursor.fetchall()])


def fetch_paired_etas(conn, model_id, strat_a, strat_b, iter_a=0, iter_b=0):
    """Pobierz pary eta dla tego samego problem_idx × sample_idx."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.eta_efficiency, b.eta_efficiency
        FROM optimization_results a
        INNER JOIN optimization_results b 
            ON a.problem_idx = b.problem_idx 
            AND a.sample_idx = b.sample_idx
            AND a.model_id = b.model_id
        WHERE a.model_id = ? AND a.strategy = ? AND a.iteration = ?
            AND b.strategy = ? AND b.iteration = ?
            AND a.eta_efficiency IS NOT NULL 
            AND b.eta_efficiency IS NOT NULL
            AND a.functional_status = 'SUCCESS'
            AND b.functional_status = 'SUCCESS'
    """, (model_id, strat_a, iter_a, strat_b, iter_b))
    rows = cursor.fetchall()
    if not rows:
        return np.array([]), np.array([])
    a = np.array([r[0] for r in rows])
    b = np.array([r[1] for r in rows])
    return a, b


def bootstrap_ci(data, n_iter=10000, ci=0.95, statistic=np.median):
    """Bootstrap CI dla statystyki."""
    if len(data) < 3:
        return None, None, None
    boot_stats = []
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic(sample))
    boot_stats = np.array(boot_stats)
    alpha_low = (1 - ci) / 2
    alpha_high = 1 - alpha_low
    return statistic(data), np.quantile(boot_stats, alpha_low), np.quantile(boot_stats, alpha_high)


def section_header(title):
    print('\n' + '=' * 75)
    print(title)
    print('=' * 75)


def section_subheader(title):
    print('\n' + '-' * 75)
    print(title)
    print('-' * 75)


def main():
    conn = sqlite3.connect(DB_PATH)
    
    models = ['claude-haiku-4-5', 'gpt-3.5-turbo', 'qwen2.5-coder-7b']
    
    # === SEKCJA 1: STATYSTYKI DESKRYPTYWNE ===
    section_header('SEKCJA 1: STATYSTYKI DESKRYPTYWNE')
    
    print(f'\n{"Model":25s} {"Strategy":15s} {"Iter":5s} {"N":4s} {"Median η":10s} {"Mean η":10s} {"CI 95%":20s}')
    print('-' * 95)
    
    for model in models:
        for strategy in ['zero_shot', 'cot', 'self_refine']:
            max_iter = 2 if strategy == 'self_refine' else 0
            for iter_ in range(max_iter + 1):
                etas = fetch_etas(conn, model, strategy, iter_)
                if len(etas) == 0:
                    continue
                med, lo, hi = bootstrap_ci(etas)
                med_str = f'{med:.3f}' if med else 'N/A'
                mean_str = f'{np.mean(etas):.3f}'
                ci_str = f'[{lo:.3f}, {hi:.3f}]' if lo is not None else 'N/A'
                print(f'{model:25s} {strategy:15s} {iter_:5d} {len(etas):4d} {med_str:10s} {mean_str:10s} {ci_str:20s}')
    
    # === SEKCJA 2: H5 — CoT vs zero-shot per model ===
    section_header('SEKCJA 2: H5 — Wilcoxon CoT vs zero-shot per model')
    
    h5_results = {}
    
    for model in models:
        section_subheader(f'Model: {model}')
        
        a, b = fetch_paired_etas(conn, model, 'zero_shot', 'cot')
        
        if len(a) < 6:
            print(f'  Insufficient paired data: n={len(a)}, skip')
            continue
        
        # Statystyki deskryptywne
        print(f'  N paired:           {len(a)}')
        print(f'  Median zero_shot η: {np.median(a):.3f}')
        print(f'  Median CoT η:       {np.median(b):.3f}')
        print(f'  Median Δ (CoT-ZS):  {np.median(b - a):+.3f}')
        
        # Wilcoxon two-sided
        diff = b - a
        non_zero = diff[diff != 0]
        if len(non_zero) < 6:
            print(f'  Too few non-zero differences for Wilcoxon')
            continue
        
        stat, p_two = stats.wilcoxon(non_zero, alternative='two-sided')
        stat_g, p_g = stats.wilcoxon(non_zero, alternative='greater')  # CoT > ZS
        stat_l, p_l = stats.wilcoxon(non_zero, alternative='less')     # CoT < ZS
        
        print(f'  Wilcoxon (two-sided): W={stat:.2f}, p={p_two:.4f}')
        print(f'  Wilcoxon (CoT > ZS):  W={stat_g:.2f}, p={p_g:.4f}')
        print(f'  Wilcoxon (CoT < ZS):  W={stat_l:.2f}, p={p_l:.4f}')
        
        # Effect size: r = Z / sqrt(N)
        z = stats.norm.ppf(1 - p_two/2)
        r = z / np.sqrt(len(non_zero))
        print(f'  Effect size r:        {r:.3f}')
        
        # Interpretation
        if p_two < ALPHA:
            direction = 'CoT > ZS' if np.median(diff) > 0 else 'CoT < ZS'
            print(f'  WYNIK: ISTOTNE ({direction})')
        else:
            print(f'  WYNIK: nieistotne, brak przewagi strategii')
        
        h5_results[model] = {
            'n': len(a),
            'median_zs': float(np.median(a)),
            'median_cot': float(np.median(b)),
            'median_delta': float(np.median(diff)),
            'p_two_sided': float(p_two),
            'p_greater': float(p_g),
            'p_less': float(p_l),
            'effect_r': float(r),
        }
    
    # === SEKCJA 3: H6 — Self-Refine progression ===
    section_header('SEKCJA 3: H6 — Self-Refine progression (ΔS₀₁, ΔS₁₂)')
    
    h6_results = {}
    
    for model in models:
        section_subheader(f'Model: {model}')
        
        # ΔS₀₁
        a, b = fetch_paired_etas(conn, model, 'self_refine', 'self_refine', iter_a=0, iter_b=1)
        if len(a) < 6:
            print(f'  ΔS₀₁: insufficient n={len(a)}')
            continue
        
        delta_01 = b - a
        non_zero = delta_01[delta_01 != 0]
        
        print(f'  Iter 0→1:')
        print(f'    N paired: {len(a)}')
        print(f'    Median iter 0 η: {np.median(a):.3f}')
        print(f'    Median iter 1 η: {np.median(b):.3f}')
        print(f'    Median ΔS₀₁: {np.median(delta_01):+.3f}')
        
        if len(non_zero) >= 6:
            stat, p_g = stats.wilcoxon(non_zero, alternative='greater')
            print(f'    Wilcoxon (iter 1 > iter 0): W={stat:.2f}, p={p_g:.4f}')
            phase_a_significant = p_g < ALPHA
        else:
            phase_a_significant = False
            p_g = None
        
        # ΔS₁₂
        a2, b2 = fetch_paired_etas(conn, model, 'self_refine', 'self_refine', iter_a=1, iter_b=2)
        if len(a2) < 6:
            print(f'  ΔS₁₂: insufficient n={len(a2)}')
            continue
        
        delta_12 = b2 - a2
        non_zero_12 = delta_12[delta_12 != 0]
        
        print(f'  Iter 1→2:')
        print(f'    N paired: {len(a2)}')
        print(f'    Median iter 1 η: {np.median(a2):.3f}')
        print(f'    Median iter 2 η: {np.median(b2):.3f}')
        print(f'    Median ΔS₁₂: {np.median(delta_12):+.3f}')
        
        if len(non_zero_12) >= 6:
            stat, p_l = stats.wilcoxon(non_zero_12, alternative='less')
            print(f'    Wilcoxon (iter 2 < iter 1): W={stat:.2f}, p={p_l:.4f}')
            phase_b_significant = p_l < ALPHA
        else:
            phase_b_significant = False
            p_l = None
        
        # Interpretation
        if phase_a_significant and phase_b_significant:
            verdict = 'H6 PEŁNI POTWIERDZONA (obie fazy istotne)'
        elif phase_a_significant or phase_b_significant:
            phase = 'tylko poprawa' if phase_a_significant else 'tylko regresja'
            verdict = f'H6 CZĘŚCIOWO POTWIERDZONA ({phase})'
        else:
            verdict = 'H6 ODRZUCONA (żadna faza nieistotna)'
        
        print(f'  WYNIK: {verdict}')
        
        h6_results[model] = {
            'median_iter_0': float(np.median(a)),
            'median_iter_1': float(np.median(b)),
            'median_iter_2': float(np.median(b2)),
            'delta_01': float(np.median(delta_01)),
            'delta_12': float(np.median(delta_12)),
            'p_phase_a': float(p_g) if p_g is not None else None,
            'p_phase_b': float(p_l) if p_l is not None else None,
            'verdict': verdict,
        }
    
    # === SEKCJA 4: H7 — Korelacja regresji z pass@1 ===
    section_header('SEKCJA 4: H7 — Regression rate vs pass@1 HumanEval+')
    
    cursor = conn.cursor()
    h7_data = []
    
    print(f'\n{"Model":25s} {"pass@1 HE+":12s} {"Success EffiBench":18s} {"Regression":10s}')
    print('-' * 70)
    
    for model in models:
        cursor.execute("""
            SELECT COUNT(*), SUM(CASE WHEN functional_status = 'SUCCESS' THEN 1 ELSE 0 END)
            FROM optimization_results
            WHERE model_id = ? AND iteration = 0
        """, (model,))
        total, success = cursor.fetchone()
        sr_effibench = success / total if total else 0
        pass_at_1 = PASS_AT_1_HUMANEVAL[model]
        regression = pass_at_1 - sr_effibench
        
        h7_data.append({
            'model': model,
            'pass_at_1': pass_at_1,
            'sr_effibench': sr_effibench,
            'regression': regression,
        })
        
        print(f'{model:25s} {pass_at_1*100:8.1f}%   {sr_effibench*100:14.1f}%  {regression*100:+9.1f} pp')
    
    # Kendall τ
    pass_vec = [d['pass_at_1'] for d in h7_data]
    reg_vec = [d['regression'] for d in h7_data]
    
    print()
    if len(pass_vec) >= 3:
        tau, p_kendall = stats.kendalltau(pass_vec, reg_vec)
        print(f'Kendall τ: {tau:.3f}, p-value: {p_kendall:.4f}')
        print(f'(N=3 modeli — niska siła, traktować jako wskazanie kierunku)')
        
        if tau < -0.3:
            print('KIERUNEK ZGODNY Z H7: regresja maleje z większym pass@1')
        elif tau > 0.3:
            print('KIERUNEK PRZECIWNY DO H7: regresja rośnie z większym pass@1')
        else:
            print('Brak wyraźnego kierunku korelacji')
    
    # Per-strategy regression
    section_subheader('Regression rate per (model × strategia, iter 0)')
    
    print(f'\n{"Model":25s} {"Strategy":15s} {"N":4s} {"Success%":10s} {"Failure%":10s}')
    print('-' * 65)
    
    for model in models:
        for strategy in ['zero_shot', 'cot', 'self_refine']:
            cursor.execute("""
                SELECT COUNT(*), SUM(CASE WHEN functional_status = 'SUCCESS' THEN 1 ELSE 0 END)
                FROM optimization_results
                WHERE model_id = ? AND strategy = ? AND iteration = 0
            """, (model, strategy))
            total, success = cursor.fetchone()
            success = success or 0
            sr = success / total if total else 0
            failure = 1 - sr
            print(f'{model:25s} {strategy:15s} {total:4d} {sr*100:8.1f}% {failure*100:8.1f}%')
    
    # === SEKCJA 5: Friedman test (porównanie modeli) ===
    section_header('SEKCJA 5: Friedman test — porównanie modeli (3 strategie × 3 modele)')
    
    # Dla każdego modelu, zbierz median eta per (strategy)
    friedman_data = []
    for model in models:
        row = []
        for strategy in ['zero_shot', 'cot', 'self_refine']:
            etas = fetch_etas(conn, model, strategy, iteration=0)
            row.append(np.median(etas) if len(etas) > 0 else np.nan)
        friedman_data.append(row)
    
    friedman_data = np.array(friedman_data)
    print('\nMacierz median η (rows=models, cols=[zero_shot, cot, self_refine]):')
    print(friedman_data)
    
    if not np.isnan(friedman_data).any():
        try:
            stat, p = stats.friedmanchisquare(*friedman_data.T)
            print(f'\nFriedman χ²: {stat:.3f}, p: {p:.4f}')
            print(f'(test różnic między strategiami w obrębie 3 modeli)')
        except Exception as e:
            print(f'\nFriedman test failed: {e}')
    
    # === SEKCJA 6: Macierz po taksonomii ===
    section_header('SEKCJA 6: Macierz η per (time_bucket × strategy)')
    
    for model in models:
        section_subheader(f'Model: {model}')
        print(f'\n{"Bucket":10s} {"zero_shot":12s} {"cot":12s} {"self_refine":12s}')
        for bucket in ['FAST', 'MEDIUM', 'SLOW']:
            row = [bucket]
            for strategy in ['zero_shot', 'cot', 'self_refine']:
                cursor.execute("""
                    SELECT AVG(eta_efficiency), COUNT(*)
                    FROM optimization_results
                    WHERE model_id = ? AND strategy = ? AND iteration = 0
                        AND canonical_time_bucket = ?
                        AND functional_status = 'SUCCESS'
                """, (model, strategy, bucket))
                avg, n = cursor.fetchone()
                if avg is not None:
                    row.append(f'{avg:.2f} (n={n})')
                else:
                    row.append('N/A')
            print(f'{row[0]:10s} {row[1]:12s} {row[2]:12s} {row[3]:12s}')
    
    # === ZAPISZ WYNIKI ===
    output = {
        'h5': h5_results,
        'h6': h6_results,
        'h7': {
            'data': h7_data,
            'kendall_tau': float(tau) if len(pass_vec) >= 3 else None,
            'kendall_p': float(p_kendall) if len(pass_vec) >= 3 else None,
        },
    }
    
    output_path = Path('results/analysis_h5_h6_h7.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f'\n✓ Wyniki zapisane do {output_path}')
    
    conn.close()


if __name__ == '__main__':
    main()