"""
Konfiguracja eksperymentu - praca magisterska.

Wszystkie parametry eksperymentalne w jednym miejscu, aby zapewnić
reprodukowalność i łatwą modyfikację.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# =============================================================================
# ŚCIEŻKI
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

DB_PATH = RESULTS_DIR / "experiment.sqlite"
LOGS_DIR = RESULTS_DIR / "logs"

# Tworzenie katalogów, jeśli nie istnieją
for d in [DATA_DIR, RESULTS_DIR, PROMPTS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True, parents=True)


# =============================================================================
# MODELE
# =============================================================================
@dataclass
class ModelConfig:
    name: str                        # identyfikator w bazie
    provider: Literal["anthropic", "openai", "google", "huggingface"]
    api_model_id: str                # ID modelu u dostawcy
    is_specialized: bool             # czy wyspecjalizowany w kodzie
    release_year: int                # rok udostępnienia
    notes: str = ""


MODELS = [
    ModelConfig(
        name="claude-haiku-4.5",
        provider="anthropic",
        api_model_id="claude-haiku-4-5",
        is_specialized=False,
        release_year=2025,
        notes="Flagowy model Haiku Anthropic, październik 2025",
    ),
    ModelConfig(
        name="gpt-3.5-turbo",
        provider="openai",
        api_model_id="gpt-3.5-turbo-0125",
        is_specialized=False,
        release_year=2022,  # bazowa generacja, snapshot z 2024
        notes="Model historyczny - punkt odniesienia dla H8",
    ),  
    ModelConfig(
        name="gemini-flash-2.5",
        provider="google",
        api_model_id="gemini-2.5-flash",
        is_specialized=False,
        release_year=2025,
        notes="Model Google Gemini 2.5 Flash, marzec 2025 (paid tier)",
    ),
    ModelConfig(
        name="qwen2.5-coder-7b",
        provider="huggingface",
        api_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        is_specialized=True,
        release_year=2024,
        notes="OSS wyspecjalizowany - uruchamiany w Kaggle/Colab",
    ),
    ModelConfig(
        name="deepseek-coder-6.7b",
        provider="huggingface",
        api_model_id="deepseek-ai/deepseek-coder-6.7b-instruct",
        is_specialized=True,
        release_year=2024,
        notes="OSS wyspecjalizowany - uruchamiany w Kaggle/Colab",
    ),
]


# =============================================================================
# PARAMETRY EKSPERYMENTU
# =============================================================================
@dataclass
class ExperimentConfig:
    # Próbkowanie
    temperature: float = 0.2           # główne eksperymenty
    top_p: float = 0.95
    max_tokens: int = 2048
    n_samples_generation: int = 5      # n dla generowania
    n_samples_audit: int = 3           # n dla audytu

    # Wykonanie testów
    test_timeout_sec: int = 30
    test_memory_limit_mb: int = 1024

    # API
    api_max_retries: int = 3
    api_retry_base_delay: float = 2.0
    api_request_timeout_sec: int = 60

    # Eksperyment wrażliwości (tylko Qwen2.5-Coder)
    sensitivity_temperatures: tuple = (0.0, 0.2, 0.5, 0.8, 1.0)
    sensitivity_n_tasks: int = 50      # podzbiór HumanEval+
    sensitivity_n_samples: int = 5

    # Statystyka
    bootstrap_iterations: int = 10000
    confidence_level: float = 0.95
    significance_level: float = 0.05
    multiple_testing_correction: str = "fdr_bh"  # Benjamini-Hochberg


CONFIG = ExperimentConfig()


# =============================================================================
# BENCHMARKI
# =============================================================================
@dataclass
class BenchmarkConfig:
    name: str
    n_tasks: int                       # liczba zadań w pełnym zestawie
    n_tasks_used: int                  # liczba zadań używanych w eksperymencie
    description: str
    download_url: str = ""


BENCHMARKS = {
    "humaneval_plus": BenchmarkConfig(
        name="HumanEval+",
        n_tasks=164,
        n_tasks_used=164,
        description="EvalPlus rozszerzenie HumanEval - klasyczny benchmark generowania",
    ),
    "livecodebench": BenchmarkConfig(
        name="LiveCodeBench (subset 2024)",
        n_tasks=50,
        n_tasks_used=50,
        description="Podzbiór 50 zadań z 2024 - contamination-free",
        download_url="https://github.com/LiveCodeBench/LiveCodeBench",
    ),
    "securityeval": BenchmarkConfig(
        name="SecurityEval",
        n_tasks=121,
        n_tasks_used=121,
        description="Benchmark CWE - 69 kategorii podatności",
        download_url="https://github.com/s2e-lab/SecurityEval",
    ),
}


# =============================================================================
# WERYFIKOWANE HIPOTEZY
# =============================================================================
HYPOTHESES = {
    "H1": "Modele wyspecjalizowane w kodzie (Qwen2.5-Coder, DeepSeek-Coder) "
          "osiągają wyższy pass@1 niż modele ogólnego przeznaczenia "
          "(Claude Haiku, GPT-3.5) - przy podobnej skali parametrów.",
    "H2": "Wyniki na klasycznych benchmarkach (HumanEval+) są systematycznie "
          "wyższe niż na contamination-free (LiveCodeBench), różnica zależy "
          "od modelu i wskazuje na zróżnicowane narażenie na zapamiętywanie.",
    "H3": "Jakość kodu (CC, MI, PEP 8) i poprawność funkcjonalna (pass@1) "
          "są częściowo niezależne (korelacja Pearsona w zakresie 0.3-0.7).",
    "H4": "Wariancja wyników między powtórzeniami jest niezerowa, różni się "
          "między modelami, i jest ujemnie skorelowana z średnim pass@1.",
    "H8": "Modele udostępnione w 2024 (Claude Haiku, Qwen2.5-Coder, "
          "DeepSeek-Coder) generują kod o niższym vulnerability rate niż "
          "GPT-3.5, ale słabości w CWE-20/78/89 utrzymują się.",
    "H9": "LLM jako audytorzy są mniej skuteczni niż Bandit dla podatności "
          "strukturalnych (np. CWE-78), ale skuteczniejsi dla kontekstowych "
          "(np. CWE-862).",
    "H10": "Korelacja między zdolnością generowania bezpiecznego kodu a "
           "zdolnością audytu jest słaba (|r| < 0.5).",
}
