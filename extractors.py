"""
Ekstraktor kodu Pythona z odpowiedzi modeli.

Modele zazwyczaj zwracają kod w blokach markdown (```python ... ```),
ale czasem bez markdown, w nieoczekiwanych formatach lub uciety
(truncation z max_tokens).

Strategia: 5 poziomów fallbacków od najściślejszego do najluźniejszego:
1. Zamknięty blok ```python ... ``` z poprawną składnią
2. Zamknięty blok ``` ... ``` (bez language tag) jeśli wygląda jak Python
3. Cała odpowiedź jako kod (jeśli parsuje się jako Python)
4. NOWE: Niezamknięty blok ```python ... <koniec> (truncation)
5. NOWE: Smart extraction - wyłuskanie kodu z mieszanego tekstu+kodu
"""
import re
import ast
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractionResult:
    """Wynik ekstrakcji kodu."""
    code: Optional[str]
    status: str  # 'success' | 'no_code_block' | 'invalid_syntax' | 'empty'
    method: str  # która metoda znalazła kod


# Standardowe regexy dla zamkniętych bloków markdown
PATTERN_PYTHON_BLOCK = re.compile(
    r"```(?:python|py|Python|PY)?\s*\n(.*?)\n```",
    re.DOTALL | re.MULTILINE,
)
PATTERN_GENERIC_BLOCK = re.compile(
    r"```\s*\n(.*?)\n```",
    re.DOTALL | re.MULTILINE,
)

# NOWE regexy dla niezamkniętych bloków (truncation z max_tokens)
PATTERN_PYTHON_BLOCK_OPEN = re.compile(
    r"```(?:python|py|Python|PY)?\s*\n(.*?)$",
    re.DOTALL,
)
PATTERN_GENERIC_BLOCK_OPEN = re.compile(
    r"```\s*\n(.*?)$",
    re.DOTALL,
)


def is_valid_python(code: str) -> bool:
    """Sprawdza, czy kod jest składniowo poprawny."""
    if not code or not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def looks_like_python(text: str) -> bool:
    """
    Heurystyka: czy tekst wygląda na kod Pythona?
    Sprawdza obecność typowych konstrukcji.
    """
    if not text or not text.strip():
        return False

    python_keywords = [
        r"^\s*(def|class|import|from)\s",
        r"^\s*(if|for|while|try|with)\s.+:",
        r"^\s*(return|yield|raise|pass|continue|break)",
        r"^\s*@\w+",  # dekoratory
    ]
    lines = text.split("\n")
    matched_lines = 0
    for line in lines[:30]:
        for pattern in python_keywords:
            if re.match(pattern, line):
                matched_lines += 1
                break

    return matched_lines >= 1


def try_fix_truncated(code: str) -> Optional[str]:
    """
    Próbuje naprawić uciety kod żeby się parsował.

    Strategie:
    1. Usuwanie ostatnich N linii (truncation często urywa się w środku)
    2. Domykanie nawiasów (jeśli więcej otwartych niż zamkniętych)
    """
    if not code or not code.strip():
        return None

    # Strategia 1: stripuj po jednej linii od końca
    lines = code.split("\n")
    for n_remove in range(0, min(15, len(lines))):
        if n_remove == 0:
            candidate = code
        else:
            candidate = "\n".join(lines[:-n_remove])

        if is_valid_python(candidate):
            return candidate

    # Strategia 2: domknij niedomknięte nawiasy
    open_parens = code.count("(") - code.count(")")
    open_brackets = code.count("[") - code.count("]")
    open_braces = code.count("{") - code.count("}")

    if open_parens > 0 or open_brackets > 0 or open_braces > 0:
        suffix = ")" * max(0, open_parens) + "]" * max(0, open_brackets) + "}" * max(0, open_braces)
        for n_remove in range(0, min(15, len(lines))):
            if n_remove == 0:
                candidate = code + suffix
            else:
                candidate = "\n".join(lines[:-n_remove]) + suffix
            if is_valid_python(candidate):
                return candidate

    return None


def smart_extract_python(text: str) -> Optional[str]:
    """
    Smart extraction: wyłuskuje największy zwarty blok kodu Pythona
    z mieszaniny tekstu i kodu.
    """
    if not text or not text.strip():
        return None

    lines = text.split("\n")

    code_starts = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if re.match(r"^(def|class|import|from|@\w+)", stripped):
            if (
                stripped.endswith(":")
                or "(" in stripped
                or stripped.startswith(("import ", "from "))
                or stripped.startswith("@")
            ):
                code_starts.append(i)

    if not code_starts:
        return None

    candidate = "\n".join(lines[code_starts[0]:])

    if is_valid_python(candidate):
        return candidate

    fixed = try_fix_truncated(candidate)
    if fixed:
        return fixed

    return None


def extract_code(response: str) -> ExtractionResult:
    """
    Ekstraktuje kod Pythona z odpowiedzi modelu.

    Pięć poziomów ekstrakcji od najbardziej restrykcyjnej do permisywnej.
    Zwraca ExtractionResult ze status w {'success', 'no_code_block', 'invalid_syntax', 'empty'}.
    """
    if not response or not response.strip():
        return ExtractionResult(code=None, status="empty", method="none")

    # === Metoda 1: zamknięty blok ```python ===
    matches = PATTERN_PYTHON_BLOCK.findall(response)
    if matches:
        code = max(matches, key=len).strip()
        if is_valid_python(code):
            return ExtractionResult(code=code, status="success", method="python_block")
        # Nieprawidłowa składnia - spróbuj naprawić
        fixed = try_fix_truncated(code)
        if fixed:
            return ExtractionResult(
                code=fixed, status="success", method="python_block_fixed"
            )
        return ExtractionResult(
            code=code, status="invalid_syntax", method="python_block"
        )

    # === Metoda 2: zamknięty blok ``` (bez tag) ===
    matches = PATTERN_GENERIC_BLOCK.findall(response)
    for match in matches:
        candidate = match.strip()
        if is_valid_python(candidate):
            return ExtractionResult(
                code=candidate, status="success", method="generic_block"
            )

    # === Metoda 3: cała odpowiedź jako kod ===
    candidate = response.strip()
    if is_valid_python(candidate):
        return ExtractionResult(code=candidate, status="success", method="raw")

    # === Metoda 4 (NOWA): niezamknięty blok ```python (truncation) ===
    matches = PATTERN_PYTHON_BLOCK_OPEN.findall(response)
    if matches:
        code = max(matches, key=len).strip()
        if is_valid_python(code):
            return ExtractionResult(
                code=code, status="success", method="python_block_open"
            )
        fixed = try_fix_truncated(code)
        if fixed:
            return ExtractionResult(
                code=fixed, status="success", method="python_block_open_fixed"
            )

    # Fallback dla niezamkniętego generic block
    matches = PATTERN_GENERIC_BLOCK_OPEN.findall(response)
    if matches:
        code = max(matches, key=len).strip()
        if is_valid_python(code):
            return ExtractionResult(
                code=code, status="success", method="generic_block_open"
            )
        fixed = try_fix_truncated(code)
        if fixed:
            return ExtractionResult(
                code=fixed, status="success", method="generic_block_open_fixed"
            )

    # === Metoda 5 (NOWA): smart extraction z tekstu+kodu ===
    if looks_like_python(response):
        smart = smart_extract_python(response)
        if smart:
            return ExtractionResult(
                code=smart, status="success", method="smart_extract"
            )

    return ExtractionResult(code=None, status="no_code_block", method="none")


def extract_function_only(code: str, function_name: str) -> Optional[str]:
    """
    Wyodrębnia tylko definicję wskazanej funkcji z kodu.

    Przydatne, gdy model wygenerował dodatkowo testy lub przykłady użycia,
    a my chcemy tylko samą funkcję.

    Returns None, jeśli funkcja nie została znaleziona.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.unparse(node)

    return None


def extract_audit_response(response: str) -> dict:
    """
    Parsuje odpowiedź audytora w formacie JSON.

    Returns dict z polami: vulnerable, cwe_id, vulnerability_type,
    explanation, confidence. Lub {'parse_error': str} przy niepowodzeniu.
    """
    json_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    match = json_pattern.search(response)

    if match:
        json_str = match.group(1)
    else:
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"parse_error": "no_json_found", "raw": response[:500]}
        json_str = response[start:end + 1]

    try:
        parsed = json.loads(json_str)
        required = ["vulnerable"]
        for key in required:
            if key not in parsed:
                return {"parse_error": f"missing_key_{key}", "raw": response[:500]}
        return parsed
    except json.JSONDecodeError as e:
        return {"parse_error": f"json_decode: {e}", "raw": response[:500]}


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    test_cases = [
        ("Standard python block", "```python\ndef add(a, b):\n    return a + b\n```", "success"),
        ("Truncated python block", "```python\nimport base64\nimport pickle\ndef profile():\n    raw_data = request.args.get", "success"),
        ("Plain code, no markdown", "# @app.route('/profile')\ndef profile():\n    '''Get user.'''\n    return 'hi'", "success"),
        ("Text + code mixed", "Here is my solution:\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nThis works.", "success"),
        ("Empty", "", "empty"),
        ("Only text", "I cannot help with this request.", "no_code_block"),
        ("Generic block", "```\ndef bar():\n    return 42\n```", "success"),
        ("Raw code", "def baz(): return 1", "success"),
        ("Invalid syntax", "```python\ndef broken(:\n    pass\n```", "invalid_syntax"),
    ]

    print("=" * 70)
    print("TEST: extract_code")
    print("=" * 70)
    for name, inp, expected in test_cases:
        result = extract_code(inp)
        ok = "OK" if result.status == expected else "FAIL"
        print(f"[{ok}] {name}: status={result.status} (expected={expected}), method={result.method}")

    print("\n" + "=" * 70)
    print("TEST: extract_audit_response")
    print("=" * 70)
    audit_response = '''Here is my analysis:
```json
{
  "vulnerable": true,
  "cwe_id": "CWE-78",
  "vulnerability_type": "Command injection via os.system",
  "explanation": "User input is directly passed to shell.",
  "confidence": "high"
}
```'''
    parsed = extract_audit_response(audit_response)
    print(f"Parsed: {parsed}")

    print("\n" + "=" * 70)
    print("TEST: extract_function_only")
    print("=" * 70)
    code = """
def helper():
    return 1

def main(x):
    return helper() + x

def unused():
    pass
"""
    result = extract_function_only(code, "main")
    print(f"Extracted main:\n{result}")