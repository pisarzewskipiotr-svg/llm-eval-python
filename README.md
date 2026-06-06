# LLM Code Evaluation Framework

Kod źródłowy eksperymentu z pracy magisterskiej:
"Badanie efektywności sztucznej inteligencji w generowaniu,
optymalizacji i audycie kodu Pythona"

## Struktura projektu
- `llm_eval/` — główny moduł ewaluacji
- `data/` — zadania benchmarków (HumanEval+, SecurityEval)
- `oss_results/` — wyniki modeli OSS (Qwen, DeepSeek)
- `run_closed_models.py` — uruchamianie modeli przez API
- `run_analysis.py` — analiza statystyczna wyników

## Instalacja
pip install -r requirements.txt

## Konfiguracja
Uzupełnij klucze API:
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...

## Replikacja
python run_closed_models.py --benchmark humaneval_plus --model claude-haiku-4-5
python run_analysis.py