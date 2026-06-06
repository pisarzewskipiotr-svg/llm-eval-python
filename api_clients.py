"""
Klienty API dla Anthropic, OpenAI i Google (Gemini).

Każdy klient implementuje wspólny interfejs `generate(prompt) -> GenerationResult`.
Retry logic z eksponencjalnym backoffem dla błędów rate limit i sieciowych.
"""
import os
import time
from dataclasses import dataclass
from typing import Optional, Protocol
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Wczytanie .env (jeśli istnieje)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv nie zainstalowany - klucze muszą być w env

import anthropic
from openai import OpenAI

# Google GenAI SDK (nowy, GA od maja 2025)
# Stary 'google-generativeai' jest deprecated od listopada 2025
try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


@dataclass
class GenerationResult:
    """Ujednolicony wynik generowania - niezależnie od dostawcy."""
    raw_response: str
    tokens_input: int
    tokens_output: int
    generation_time_sec: float
    finish_reason: str = "stop"
    error: Optional[str] = None


class APIClient(Protocol):
    """Interfejs wspólny dla wszystkich klientów API."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.95,
    ) -> GenerationResult:
        ...


# =============================================================================
# ANTHROPIC (Claude)
# =============================================================================
class AnthropicClient:
    """Klient dla Anthropic API (Claude)."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        )),
        reraise=True,
    )
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.95,
    ) -> GenerationResult:
        start = time.time()

        try:
            response = self.client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=60.0,
            )

            elapsed = time.time() - start

            # Concatenate text blocks (Claude może zwracać multiple blocks)
            raw_text = "".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )

            return GenerationResult(
                raw_response=raw_text,
                tokens_input=response.usage.input_tokens,
                tokens_output=response.usage.output_tokens,
                generation_time_sec=elapsed,
                finish_reason=response.stop_reason or "unknown",
            )

        except anthropic.BadRequestError as e:
            return GenerationResult(
                raw_response="",
                tokens_input=0,
                tokens_output=0,
                generation_time_sec=time.time() - start,
                error=f"BadRequest: {str(e)}",
            )


# =============================================================================
# OPENAI (GPT)
# =============================================================================
class OpenAIClient:
    """Klient dla OpenAI API (GPT)."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=60.0,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.95,
    ) -> GenerationResult:
        start = time.time()

        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            elapsed = time.time() - start

            return GenerationResult(
                raw_response=response.choices[0].message.content or "",
                tokens_input=response.usage.prompt_tokens,
                tokens_output=response.usage.completion_tokens,
                generation_time_sec=elapsed,
                finish_reason=response.choices[0].finish_reason,
            )

        except Exception as e:
            return GenerationResult(
                raw_response="",
                tokens_input=0,
                tokens_output=0,
                generation_time_sec=time.time() - start,
                error=f"Error: {str(e)}",
            )


# =============================================================================
# GOOGLE (Gemini)
# =============================================================================
class GoogleClient:
    """
    Klient dla Google Gemini API (przez Google GenAI SDK).

    Używa nowego pakietu `google-genai` (GA od maja 2025).
    Stary `google-generativeai` jest deprecated od listopada 2025.

    UWAGA: Free tier ma limity:
    - gemini-2.0-flash: ~15 RPM, ~1500 RPD (zapytań na dzień)
    - gemini-2.5-flash: ~10 RPM, ~250 RPD
    Sprawdź aktualne limity: https://ai.google.dev/pricing
    """

    def __init__(self, api_key: Optional[str] = None):
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "google-genai nie jest zainstalowany. "
                "Uruchom: pip install google-genai"
            )

        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Brak klucza API Google. Ustaw zmienną GOOGLE_API_KEY lub GEMINI_API_KEY."
            )

        self.client = genai.Client(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.95,
    ) -> GenerationResult:
        start = time.time()

        try:
            # Konfiguracja generowania - Gemini ma osobny obiekt config
            #
            # safety_settings=BLOCK_NONE: rozluźnione filtry bezpieczeństwa
            # Standardowa praktyka w badaniach ewaluacyjnych LLM. Domyślne
            # filtry Gemini są zbyt agresywne dla zadań kodowych - blokują
            # nawet niewinne zadania (np. operacje statystyczne na liście).
            # W pilotażu zaobserwowano ~7% pustych odpowiedzi przy domyślnych
            # ustawieniach. BLOCK_NONE zapewnia porównywalność z Claude/GPT.
            safety_settings = [
                genai_types.SafetySetting(
                    category=cat,
                    threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                )
                for cat in [
                    genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                ]
            ]

            config = genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
                safety_settings=safety_settings,
            )

            response = self.client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config=config,
            )

            elapsed = time.time() - start

            # Pobranie tekstu - Gemini zwraca content z parts
            raw_text = response.text if hasattr(response, "text") and response.text else ""

            # Token counting
            tokens_input = 0
            tokens_output = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_input = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                tokens_output = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            # Finish reason
            finish_reason = "stop"
            if hasattr(response, "candidates") and response.candidates:
                fr = getattr(response.candidates[0], "finish_reason", None)
                if fr is not None:
                    finish_reason = str(fr).lower().split(".")[-1]  # np. "STOP" -> "stop"

            return GenerationResult(
                raw_response=raw_text,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                generation_time_sec=elapsed,
                finish_reason=finish_reason,
            )

        except genai_errors.APIError as e:
            return GenerationResult(
                raw_response="",
                tokens_input=0,
                tokens_output=0,
                generation_time_sec=time.time() - start,
                error=f"GoogleAPIError: {str(e)}",
            )
        except Exception as e:
            return GenerationResult(
                raw_response="",
                tokens_input=0,
                tokens_output=0,
                generation_time_sec=time.time() - start,
                error=f"Error: {str(e)}",
            )


# =============================================================================
# FACTORY
# =============================================================================
def get_client(provider: str) -> APIClient:
    """Factory dla klientów API."""
    if provider == "anthropic":
        return AnthropicClient()
    elif provider == "openai":
        return OpenAIClient()
    elif provider == "google":
        return GoogleClient()
    else:
        raise ValueError(f"Nieznany dostawca: {provider}")


# =============================================================================
# TEST - sprawdzenie kluczy API i podstawowej komunikacji
# =============================================================================
if __name__ == "__main__":
    print("Test klientów API...")
    print("(wymaga zmiennych: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY)\n")

    test_prompt = "Napisz funkcję Python która dodaje dwie liczby. Zwróć tylko kod."
    test_system = "Jesteś asystentem programisty Pythona."

    # --- Anthropic ---
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("--- Anthropic Claude Haiku 3.5 ---")
        try:
            client = AnthropicClient()
            result = client.generate(
                system_prompt=test_system,
                user_prompt=test_prompt,
                model_id="claude-3-5-haiku-20241022",
                temperature=0.2,
                max_tokens=512,
            )
            if result.error:
                print(f"  BŁĄD: {result.error}")
            else:
                print(f"  Tokens: in={result.tokens_input}, out={result.tokens_output}")
                print(f"  Czas: {result.generation_time_sec:.2f}s")
                print(f"  Odpowiedź:\n{result.raw_response[:300]}")
        except Exception as e:
            print(f"  WYJĄTEK: {e}")
    else:
        print("ANTHROPIC_API_KEY nie ustawiony - pomijam test\n")

    # --- OpenAI ---
    if os.environ.get("OPENAI_API_KEY"):
        print("\n--- OpenAI GPT-3.5 ---")
        try:
            client = OpenAIClient()
            result = client.generate(
                system_prompt=test_system,
                user_prompt=test_prompt,
                model_id="gpt-3.5-turbo-0125",
                temperature=0.2,
                max_tokens=512,
            )
            if result.error:
                print(f"  BŁĄD: {result.error}")
            else:
                print(f"  Tokens: in={result.tokens_input}, out={result.tokens_output}")
                print(f"  Czas: {result.generation_time_sec:.2f}s")
                print(f"  Odpowiedź:\n{result.raw_response[:300]}")
        except Exception as e:
            print(f"  WYJĄTEK: {e}")
    else:
        print("\nOPENAI_API_KEY nie ustawiony - pomijam test")

    # --- Google ---
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if google_key and GOOGLE_AVAILABLE:
        print("\n--- Google Gemini 2.0 Flash ---")
        try:
            client = GoogleClient()
            result = client.generate(
                system_prompt=test_system,
                user_prompt=test_prompt,
                model_id="gemini-2.0-flash",
                temperature=0.2,
                max_tokens=512,
            )
            if result.error:
                print(f"  BŁĄD: {result.error}")
            else:
                print(f"  Tokens: in={result.tokens_input}, out={result.tokens_output}")
                print(f"  Czas: {result.generation_time_sec:.2f}s")
                print(f"  Odpowiedź:\n{result.raw_response[:300]}")
        except Exception as e:
            print(f"  WYJĄTEK: {e}")
    elif not google_key:
        print("\nGOOGLE_API_KEY (lub GEMINI_API_KEY) nie ustawiony - pomijam test")
    elif not GOOGLE_AVAILABLE:
        print("\ngoogle-genai nie zainstalowany - uruchom: pip install google-genai")