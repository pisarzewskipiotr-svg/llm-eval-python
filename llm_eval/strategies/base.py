"""Base class dla strategii promptowania."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from llm_eval.benchmarks.effibench import EffiBenchTask


@dataclass
class PromptResult:
    """Wynik buildowania promptu — co poszło do modelu."""
    system_message: str
    user_message: str
    strategy_name: str
    template_version: str
    iteration: int = 0  # 0 = initial, 1+ = refine iterations


@dataclass
class StrategyState:
    """
    Stan strategii podczas wykonania (głównie dla Self-Refine).
    Trzyma historię iteracji, poprzednie wyniki, feedback.
    """
    task: EffiBenchTask
    iteration: int = 0
    previous_code: Optional[str] = None
    previous_time_ms: Optional[float] = None
    canonical_time_ms: Optional[float] = None
    previous_functional_status: Optional[str] = None
    history: list[dict] = field(default_factory=list)
    
    def is_complete(self, max_iterations: int) -> bool:
        return self.iteration >= max_iterations


class PromptStrategy(ABC):
    """Bazowa klasa strategii promptowania."""
    
    strategy_name: str = "base"
    template_version: str = "v1"
    
    @abstractmethod
    def build_prompt(self, state: StrategyState) -> PromptResult:
        """Konstruuje prompt na podstawie stanu (zadanie + ewentualny feedback)."""
        ...
    
    def extract_code(self, raw_response: str) -> tuple[Optional[str], str]:
        """
        Ekstraktuje kod Python z odpowiedzi modelu.
        
        Returns:
            (code, status) gdzie status to:
                'success' | 'no_code_block' | 'syntax_error_in_extraction'
        """
        # Method 1: markdown ```python ... ```
        pattern = r'```python\s*\n?(.*?)```'
        match = re.search(pattern, raw_response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if self._is_valid_syntax(code):
                return code, 'success'
        
        # Method 2: generic ``` ... ```
        pattern = r'```\s*\n?(.*?)```'
        match = re.search(pattern, raw_response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if self._is_valid_syntax(code):
                return code, 'success'
        
        # Method 3: niezamknięty blok ```python (truncation)
        pattern = r'```python\s*\n(.*?)$'
        match = re.search(pattern, raw_response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Próbuj naprawić - usuwaj ostatnie linie aż syntax OK
            fixed = self._try_fix_truncated(code)
            if fixed:
                return fixed, 'success'
        
        # Method 4: heurystyka - znajdź `class Solution:` i bierz od tamtego miejsca
        if 'class Solution' in raw_response:
            start_idx = raw_response.find('class Solution')
            code = raw_response[start_idx:].strip()
            fixed = self._try_fix_truncated(code)
            if fixed:
                return fixed, 'success'
        
        return None, 'no_code_block'
    
    def _is_valid_syntax(self, code: str) -> bool:
        """Sprawdza czy kod ma poprawną składnię Python."""
        try:
            compile(code, '<test>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _try_fix_truncated(self, code: str, max_attempts: int = 15) -> Optional[str]:
        """Próbuje naprawić ucięty kod przez usuwanie ostatnich linii."""
        if self._is_valid_syntax(code):
            return code
        
        lines = code.split('\n')
        for attempt in range(min(max_attempts, len(lines))):
            truncated = '\n'.join(lines[:-attempt-1])
            if self._is_valid_syntax(truncated):
                return truncated
        return None