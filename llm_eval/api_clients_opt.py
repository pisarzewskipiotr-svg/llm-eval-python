"""
API clients dla modułu optymalizacji EffiBench.

Obsługuje 5 modeli:
- Claude Haiku 4.5 (Anthropic API)         — WAŻNE: BEZ top_p z temperature!
- GPT-3.5 Turbo (OpenAI API)
- Gemini 2.5 Flash (Google AI API)         — WAŻNE: handling refusals
- Qwen2.5-Coder-7B (Kaggle GPU)
- DeepSeek-Coder-6.7B (Kaggle GPU)

Returns standardized GenerationResult.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from llm_eval.strategies.base import PromptResult

from dotenv import load_dotenv
load_dotenv() 
logger = logging.getLogger(__name__)


# === Konfiguracja modeli ===

MODEL_CONFIGS = {
    'claude-haiku-4-5': {
        'provider': 'anthropic',
        'model_id': 'claude-haiku-4-5',           
        'temperature': 0.2,
        # UWAGA: NIE używamy top_p z Claude 4.5+
        'max_tokens': 2048,
        'cost_input_per_1m': 0.80,   # USD per 1M input tokens (przybliżone)
        'cost_output_per_1m': 4.00,
    },
    'gpt-3.5-turbo': {
        'provider': 'openai',
        'model_id': 'gpt-3.5-turbo',
        'temperature': 0.2,
        'top_p': 0.95,
        'max_tokens': 2048,
        'cost_input_per_1m': 0.50,
        'cost_output_per_1m': 1.50,
    },
    'gemini-2.5-flash': {
        'provider': 'google',
        'model_id': 'gemini-2.5-flash',
        'temperature': 1.0,           # Gemini wymaga
        'top_p': 0.95,
        'max_tokens': 4096,
        'cost_input_per_1m': 0.075,
        'cost_output_per_1m': 0.30,
    },
    'qwen2.5-coder-7b': {
        'provider': 'kaggle_local',
        'model_id': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'temperature': 0.2,
        'top_p': 0.95,
        'max_tokens': 2048,
        'cost_input_per_1m': 0.0,    # local
        'cost_output_per_1m': 0.0,
    },
    'deepseek-coder-6.7b': {
        'provider': 'kaggle_local',
        'model_id': 'deepseek-ai/deepseek-coder-6.7b-instruct',
        'temperature': 0.2,
        'top_p': 0.95,
        'max_tokens': 2048,
        'cost_input_per_1m': 0.0,    # local
        'cost_output_per_1m': 0.0,
    },
}


# === Result ===

@dataclass
class GenerationResult:
    """Wynik wygenerowania kodu przez API."""
    success: bool
    raw_response: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    model_id: str = ""
    error_type: Optional[str] = None      # 'API_ERROR' | 'REFUSAL' | 'EMPTY_RESPONSE' | 'TIMEOUT'
    error_message: Optional[str] = None
    finish_reason: Optional[str] = None   # 'stop' | 'max_tokens' | 'safety' | ...


# === Anthropic (Claude) ===

def generate_anthropic(prompt: PromptResult, model_key: str = 'claude-haiku-4-5') -> GenerationResult:
    """Wywołuje Anthropic API. UWAGA: NIE używaj top_p z Claude 4.5+!"""
    import anthropic
    
    config = MODEL_CONFIGS[model_key]
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return GenerationResult(success=False, error_type='API_ERROR',
                                error_message='ANTHROPIC_API_KEY not set')
    
    client = anthropic.Anthropic(api_key=api_key)
    
    start = time.perf_counter()
    try:
        # KLUCZOWE: top_p USUNIĘTE
        response = client.messages.create(
            model=config['model_id'],
            max_tokens=config['max_tokens'],
            temperature=config['temperature'],
            system=prompt.system_message,
            messages=[
                {"role": "user", "content": prompt.user_message}
            ]
        )
        duration = time.perf_counter() - start
        
        # Ekstrakcja tekstu
        if not response.content or len(response.content) == 0:
            return GenerationResult(
                success=False, model_id=config['model_id'],
                error_type='EMPTY_RESPONSE', duration_s=duration,
                finish_reason=response.stop_reason
            )
        
        raw_text = response.content[0].text if hasattr(response.content[0], 'text') else ""
        
        if not raw_text.strip():
            return GenerationResult(
                success=False, model_id=config['model_id'],
                error_type='EMPTY_RESPONSE', duration_s=duration,
                finish_reason=response.stop_reason
            )
        
        # Cost
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = (tokens_in / 1e6) * config['cost_input_per_1m'] + \
               (tokens_out / 1e6) * config['cost_output_per_1m']
        
        return GenerationResult(
            success=True, raw_response=raw_text,
            tokens_input=tokens_in, tokens_output=tokens_out,
            cost_usd=cost, duration_s=duration,
            model_id=config['model_id'],
            finish_reason=response.stop_reason
        )
    except anthropic.BadRequestError as e:
        return GenerationResult(success=False, model_id=config['model_id'],
                                error_type='API_ERROR',
                                error_message=f'BadRequest: {e}',
                                duration_s=time.perf_counter() - start)
    except Exception as e:
        return GenerationResult(success=False, model_id=config['model_id'],
                                error_type='API_ERROR',
                                error_message=f'{type(e).__name__}: {e}',
                                duration_s=time.perf_counter() - start)


# === OpenAI (GPT-3.5 Turbo) ===

def generate_openai(prompt: PromptResult, model_key: str = 'gpt-3.5-turbo') -> GenerationResult:
    """Wywołuje OpenAI API."""
    import openai
    
    config = MODEL_CONFIGS[model_key]
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return GenerationResult(success=False, error_type='API_ERROR',
                                error_message='OPENAI_API_KEY not set')
    
    client = openai.OpenAI(api_key=api_key)
    
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=config['model_id'],
            temperature=config['temperature'],
            top_p=config['top_p'],
            max_tokens=config['max_tokens'],
            messages=[
                {"role": "system", "content": prompt.system_message},
                {"role": "user", "content": prompt.user_message},
            ]
        )
        duration = time.perf_counter() - start
        
        if not response.choices or len(response.choices) == 0:
            return GenerationResult(success=False, model_id=config['model_id'],
                                    error_type='EMPTY_RESPONSE', duration_s=duration)
        
        choice = response.choices[0]
        raw_text = choice.message.content or ""
        
        if not raw_text.strip():
            return GenerationResult(success=False, model_id=config['model_id'],
                                    error_type='EMPTY_RESPONSE', duration_s=duration,
                                    finish_reason=choice.finish_reason)
        
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        cost = (tokens_in / 1e6) * config['cost_input_per_1m'] + \
               (tokens_out / 1e6) * config['cost_output_per_1m']
        
        return GenerationResult(
            success=True, raw_response=raw_text,
            tokens_input=tokens_in, tokens_output=tokens_out,
            cost_usd=cost, duration_s=duration,
            model_id=config['model_id'],
            finish_reason=choice.finish_reason
        )
    except Exception as e:
        return GenerationResult(success=False, model_id=config['model_id'],
                                error_type='API_ERROR',
                                error_message=f'{type(e).__name__}: {e}',
                                duration_s=time.perf_counter() - start)


# === Google (Gemini) ===

def generate_google(prompt: PromptResult, model_key: str = 'gemini-2.5-flash') -> GenerationResult:
    """Wywołuje Google AI API. UWAGA: handling refusals i truncation."""
    import google.generativeai as genai
    
    config = MODEL_CONFIGS[model_key]
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return GenerationResult(success=False, error_type='API_ERROR',
                                error_message='GOOGLE_API_KEY not set')
    
    genai.configure(api_key=api_key)
    
    # Safety settings: BLOCK_NONE 
    safety = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    ]
    
    model = genai.GenerativeModel(
        model_name=config['model_id'],
        system_instruction=prompt.system_message,
        safety_settings=safety,
        generation_config={
            'temperature': config['temperature'],
            'top_p': config['top_p'],
            'max_output_tokens': config['max_tokens'],
        }
    )
    
    start = time.perf_counter()
    try:
        response = model.generate_content(prompt.user_message)
        duration = time.perf_counter() - start
        
        # Sprawdź czy odpowiedź jest empty (refusal Gemini, znane z SecurityEval)
        try:
            raw_text = response.text or ""
        except (ValueError, AttributeError):
            # Gemini czasem rzuca exception przy próbie .text gdy refusal
            return GenerationResult(success=False, model_id=config['model_id'],
                                    error_type='REFUSAL', duration_s=duration,
                                    error_message='Gemini refused (no text in response)')
        
        if not raw_text.strip():
            return GenerationResult(success=False, model_id=config['model_id'],
                                    error_type='EMPTY_RESPONSE', duration_s=duration)
        
        # Tokens count - Gemini ma usage_metadata
        tokens_in = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
        tokens_out = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
        cost = (tokens_in / 1e6) * config['cost_input_per_1m'] + \
               (tokens_out / 1e6) * config['cost_output_per_1m']
        
        # Finish reason
        finish_reason = None
        if hasattr(response, 'candidates') and response.candidates:
            finish_reason = response.candidates[0].finish_reason.name if hasattr(response.candidates[0], 'finish_reason') else None
        # Wykrywanie truncation
        if finish_reason == 'MAX_TOKENS':
            logger.warning(f"Gemini hit MAX_TOKENS limit (tokens_out={tokens_out}). "
                   f"Response may be truncated. Increase max_tokens if persistent.")
        return GenerationResult(
            success=True, raw_response=raw_text,
            tokens_input=tokens_in, tokens_output=tokens_out,
            cost_usd=cost, duration_s=duration,
            model_id=config['model_id'],
            finish_reason=finish_reason
        )
    except Exception as e:
        return GenerationResult(success=False, model_id=config['model_id'],
                                error_type='API_ERROR',
                                error_message=f'{type(e).__name__}: {e}',
                                duration_s=time.perf_counter() - start)


# === Dispatcher ===

def generate(prompt: PromptResult, model_key: str) -> GenerationResult:
    """
    Główna funkcja-dispatcher.
    Wybiera odpowiedni API client na podstawie model_key.
    """
    if model_key not in MODEL_CONFIGS:
        return GenerationResult(success=False, model_id=model_key,
                                error_type='API_ERROR',
                                error_message=f'Unknown model: {model_key}')
    
    provider = MODEL_CONFIGS[model_key]['provider']
    
    if provider == 'anthropic':
        return generate_anthropic(prompt, model_key)
    elif provider == 'openai':
        return generate_openai(prompt, model_key)
    elif provider == 'google':
        return generate_google(prompt, model_key)
    elif provider == 'kaggle_local':
        return GenerationResult(success=False, model_id=model_key,
                                error_type='API_ERROR',
                                error_message='Use Kaggle notebook for OSS models (not API)')
    else:
        return GenerationResult(success=False, model_id=model_key,
                                error_type='API_ERROR',
                                error_message=f'Unknown provider: {provider}')


# === CLI dla testów ===

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    from llm_eval.benchmarks.effibench import load_sample_50
    from llm_eval.strategies import ZeroShotStrategy
    from llm_eval.strategies.base import StrategyState
    
    tasks = load_sample_50()
    task = tasks[0]
    
    strategy = ZeroShotStrategy()
    state = StrategyState(task=task)
    prompt = strategy.build_prompt(state)
    
    print(f"=== Testing API call for task: {task.task_name} ===")
    
    # Test każdego API jeśli klucz jest w env
    for model_key in ['claude-haiku-4-5', 'gpt-3.5-turbo', 'gemini-2.5-flash']:
        provider = MODEL_CONFIGS[model_key]['provider']
        env_var = {'anthropic': 'ANTHROPIC_API_KEY', 'openai': 'OPENAI_API_KEY', 'google': 'GOOGLE_API_KEY'}[provider]
        
        if not os.getenv(env_var):
            print(f"\n[SKIP] {model_key} ({env_var} not set)")
            continue
        
        print(f"\n=== {model_key} ===")
        result = generate(prompt, model_key)
        
        if result.success:
            print(f"✓ SUCCESS")
            print(f"  Duration: {result.duration_s:.2f}s")
            print(f"  Tokens: in={result.tokens_input}, out={result.tokens_output}")
            print(f"  Cost: ${result.cost_usd:.6f}")
            print(f"  Finish: {result.finish_reason}")
            print(f"  Response (first 300 chars):\n{result.raw_response[:300]}")
        else:
            print(f"✗ FAILED")
            print(f"  Error type: {result.error_type}")
            print(f"  Message: {result.error_message}")