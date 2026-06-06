"""
Analiza statyczna kodu Pythona przez:
- Radon: CC, MI, Halstead
- Ruff: PEP 8 violations
- Bandit: podatności bezpieczeństwa (CWE)

Każde narzędzie jest wywoływane jako subprocess parsujemy wyniki JSON.
"""
import subprocess
import tempfile
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QualityMetrics:
    """Metryki jakości statycznej."""
    cc_avg: Optional[float] = None
    cc_max: Optional[float] = None
    halstead_volume: Optional[float] = None
    halstead_difficulty: Optional[float] = None
    halstead_effort: Optional[float] = None
    maintainability_index: Optional[float] = None
    pep8_violations_total: int = 0
    pep8_violations_naming: int = 0
    pep8_violations_formatting: int = 0
    pep8_violations_imports: int = 0
    lines_of_code: int = 0
    n_functions: int = 0


@dataclass
class SecurityFinding:
    """Pojedyncze wykrycie podatności."""
    cwe_id: Optional[str]
    test_id: str
    severity: str
    confidence: str
    line_number: int
    code_snippet: str
    issue_text: str
    tool: str = "bandit"


@dataclass
class StaticAnalysisResult:
    """Pełny wynik analizy statycznej."""
    quality: QualityMetrics
    findings: list[SecurityFinding] = field(default_factory=list)
    analysis_errors: list[str] = field(default_factory=list)


# =============================================================================
# RADON - CC, MI, Halstead
# =============================================================================
def analyze_radon(code: str) -> dict:
    """
    Uruchamia Radon i zwraca metryki.

    Radon ma trzy podkomendy: cc, mi, hal.
    Każdą wywołujemy osobno (mogłoby być -j JSON ale prostsze parsing osobno).
    """
    metrics = {}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        # CC - cyclomatic complexity
        try:
            result = subprocess.run(
                ["radon", "cc", "-j", tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                # data jest dict {filepath: [list of functions/classes]}
                for filepath, items in data.items():
                    if items:
                        complexities = [item["complexity"] for item in items
                                       if "complexity" in item]
                        if complexities:
                            metrics["cc_avg"] = sum(complexities) / len(complexities)
                            metrics["cc_max"] = max(complexities)
                            metrics["n_functions"] = len(complexities)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            metrics["cc_error"] = str(e)

        # MI - Maintainability Index
        try:
            result = subprocess.run(
                ["radon", "mi", "-j", tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                for filepath, mi_data in data.items():
                    if isinstance(mi_data, dict) and "mi" in mi_data:
                        metrics["maintainability_index"] = mi_data["mi"]
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            metrics["mi_error"] = str(e)

        # Halstead
        try:
            result = subprocess.run(
                ["radon", "hal", "-j", tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                for filepath, hal_data in data.items():
                    total = hal_data.get("total", {})
                    if total:
                        metrics["halstead_volume"] = total.get("volume")
                        metrics["halstead_difficulty"] = total.get("difficulty")
                        metrics["halstead_effort"] = total.get("effort")
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            metrics["hal_error"] = str(e)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return metrics


# =============================================================================
# RUFF - PEP 8
# =============================================================================
def analyze_ruff(code: str) -> dict:
    """
    Uruchamia Ruff z regułami PEP 8 (E, W) i nazewnictwa (N).

    Returns dict z liczbami naruszeń per kategoria.
    """
    metrics = {
        "pep8_violations_total": 0,
        "pep8_violations_naming": 0,
        "pep8_violations_formatting": 0,
        "pep8_violations_imports": 0,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["ruff", "check", "--select", "E,W,N,I,F", "--output-format", "json",
             "--no-cache", tmp_path],
            capture_output=True, text=True, timeout=30,
        )

        if result.stdout:
            try:
                violations = json.loads(result.stdout)
                metrics["pep8_violations_total"] = len(violations)

                for v in violations:
                    code_id = v.get("code") or ""  # Bezpieczne na None i brakujący klucz
                    if not isinstance(code_id, str):
                        continue  # Skip nieprawidłowych wartości
                    if code_id.startswith("N"):
                        metrics["pep8_violations_naming"] += 1
                    elif code_id.startswith("E") or code_id.startswith("W"):
                        metrics["pep8_violations_formatting"] += 1
                    elif code_id.startswith("I"):
                        metrics["pep8_violations_imports"] += 1
            except json.JSONDecodeError:
                pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return metrics


# =============================================================================
# BANDIT - security findings
# =============================================================================
# Mapowanie Bandit test_id -> CWE
# (skrócony - pełną listę można znaleźć w dokumentacji Bandit)
BANDIT_TO_CWE = {
    "B102": "CWE-78",   # exec_used
    "B103": "CWE-732",  # set_bad_file_permissions
    "B104": "CWE-605",  # hardcoded_bind_all_interfaces
    "B105": "CWE-259",  # hardcoded_password_string
    "B106": "CWE-259",  # hardcoded_password_funcarg
    "B107": "CWE-259",  # hardcoded_password_default
    "B108": "CWE-377",  # hardcoded_tmp_directory
    "B110": "CWE-703",  # try_except_pass
    "B112": "CWE-703",  # try_except_continue
    "B201": "CWE-79",   # flask_debug_true
    "B301": "CWE-502",  # pickle
    "B302": "CWE-502",  # marshal
    "B303": "CWE-327",  # md5
    "B304": "CWE-327",  # ciphers
    "B305": "CWE-327",  # cipher_modes
    "B306": "CWE-377",  # mktemp_q
    "B307": "CWE-78",   # eval
    "B308": "CWE-79",   # mark_safe
    "B309": "CWE-295",  # httpsconnection
    "B310": "CWE-22",   # urllib_urlopen
    "B311": "CWE-330",  # random
    "B312": "CWE-330",  # telnetlib
    "B313": "CWE-20",   # xml_bad_cElementTree
    "B314": "CWE-20",   # xml_bad_ElementTree
    "B315": "CWE-20",   # xml_bad_expatreader
    "B316": "CWE-20",   # xml_bad_expatbuilder
    "B317": "CWE-20",   # xml_bad_sax
    "B318": "CWE-20",   # xml_bad_minidom
    "B319": "CWE-20",   # xml_bad_pulldom
    "B320": "CWE-20",   # xml_bad_etree
    "B321": "CWE-78",   # ftplib
    "B322": "CWE-20",   # input
    "B323": "CWE-295",  # unverified_context
    "B324": "CWE-327",  # hashlib_new_insecure_functions
    "B325": "CWE-377",  # tempnam
    "B401": "CWE-78",   # import_telnetlib
    "B402": "CWE-319",  # import_ftplib
    "B403": "CWE-502",  # import_pickle
    "B404": "CWE-78",   # import_subprocess
    "B405": "CWE-20",   # import_xml_etree
    "B406": "CWE-20",   # import_xml_sax
    "B407": "CWE-20",   # import_xml_expat
    "B408": "CWE-20",   # import_xml_minidom
    "B409": "CWE-20",   # import_xml_pulldom
    "B410": "CWE-20",   # import_lxml
    "B411": "CWE-330",  # import_xmlrpclib
    "B412": "CWE-78",   # import_httpoxy
    "B413": "CWE-327",  # import_pycrypto
    "B501": "CWE-295",  # request_with_no_cert_validation
    "B502": "CWE-327",  # ssl_with_bad_version
    "B503": "CWE-327",  # ssl_with_bad_defaults
    "B504": "CWE-327",  # ssl_with_no_version
    "B505": "CWE-326",  # weak_cryptographic_key
    "B506": "CWE-20",   # yaml_load
    "B507": "CWE-78",   # ssh_no_host_key_verification
    "B601": "CWE-78",   # paramiko_calls
    "B602": "CWE-78",   # subprocess_popen_with_shell_equals_true
    "B603": "CWE-78",   # subprocess_without_shell_equals_true
    "B604": "CWE-78",   # any_other_function_with_shell_equals_true
    "B605": "CWE-78",   # start_process_with_a_shell
    "B606": "CWE-78",   # start_process_with_no_shell
    "B607": "CWE-78",   # start_process_with_partial_path
    "B608": "CWE-89",   # hardcoded_sql_expressions
    "B609": "CWE-22",   # linux_commands_wildcard_injection
    "B610": "CWE-89",   # django_extra_used
    "B611": "CWE-89",   # django_rawsql_used
    "B701": "CWE-79",   # jinja2_autoescape_false
    "B702": "CWE-79",   # use_of_mako_templates
    "B703": "CWE-79",   # django_mark_safe
}


def analyze_bandit(code: str) -> list[SecurityFinding]:
    """
    Uruchamia Bandit i zwraca listę wykryć.
    """
    findings = []

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["bandit", "-f", "json", "-q", tmp_path],
            capture_output=True, text=True, timeout=30,
        )

        # Bandit zwraca returncode != 0, jeśli znalazł problemy - to OK
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                for issue in data.get("results", []):
                    test_id = issue.get("test_id", "")
                    finding = SecurityFinding(
                        cwe_id=BANDIT_TO_CWE.get(test_id),
                        test_id=test_id,
                        severity=issue.get("issue_severity", "UNKNOWN"),
                        confidence=issue.get("issue_confidence", "UNKNOWN"),
                        line_number=issue.get("line_number", 0),
                        code_snippet=issue.get("code", "")[:500],
                        issue_text=issue.get("issue_text", ""),
                        tool="bandit",
                    )
                    findings.append(finding)
            except json.JSONDecodeError:
                pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return findings


# =============================================================================
# MAIN: kombinacja wszystkich analiz
# =============================================================================
def analyze_code(code: str) -> StaticAnalysisResult:
    """
    Pełna analiza statyczna kodu - radon + ruff + bandit.
    """
    if not code or not code.strip():
        return StaticAnalysisResult(
            quality=QualityMetrics(),
            findings=[],
            analysis_errors=["empty_code"],
        )

    quality = QualityMetrics()
    errors = []

    # Radon
    radon_metrics = analyze_radon(code)
    quality.cc_avg = radon_metrics.get("cc_avg")
    quality.cc_max = radon_metrics.get("cc_max")
    quality.halstead_volume = radon_metrics.get("halstead_volume")
    quality.halstead_difficulty = radon_metrics.get("halstead_difficulty")
    quality.halstead_effort = radon_metrics.get("halstead_effort")
    quality.maintainability_index = radon_metrics.get("maintainability_index")
    quality.n_functions = radon_metrics.get("n_functions", 0)
    for key in ["cc_error", "mi_error", "hal_error"]:
        if key in radon_metrics:
            errors.append(f"radon_{key}: {radon_metrics[key]}")

    # Ruff
    ruff_metrics = analyze_ruff(code)
    quality.pep8_violations_total = ruff_metrics["pep8_violations_total"]
    quality.pep8_violations_naming = ruff_metrics["pep8_violations_naming"]
    quality.pep8_violations_formatting = ruff_metrics["pep8_violations_formatting"]
    quality.pep8_violations_imports = ruff_metrics["pep8_violations_imports"]

    # Bandit
    findings = analyze_bandit(code)

    # LoC
    quality.lines_of_code = sum(1 for line in code.split("\n") if line.strip())

    return StaticAnalysisResult(
        quality=quality,
        findings=findings,
        analysis_errors=errors,
    )


# =============================================================================
# Test (wymaga zainstalowanego radon, ruff, bandit)
# =============================================================================
if __name__ == "__main__":
    test_code = """
import subprocess

def run_command(user_input):
    # Podatność: command injection
    result = subprocess.call(user_input, shell=True)
    return result

def calculate_factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
"""

    result = analyze_code(test_code)
    print("=== METRYKI JAKOŚCI ===")
    print(f"CC avg: {result.quality.cc_avg}")
    print(f"CC max: {result.quality.cc_max}")
    print(f"MI: {result.quality.maintainability_index}")
    print(f"Halstead volume: {result.quality.halstead_volume}")
    print(f"PEP 8 violations: {result.quality.pep8_violations_total}")
    print(f"LoC: {result.quality.lines_of_code}")
    print(f"Functions: {result.quality.n_functions}")

    print("\n=== WYKRYTE PODATNOŚCI ===")
    for f in result.findings:
        print(f"  [{f.severity}] {f.test_id} -> {f.cwe_id}: {f.issue_text}")

    if result.analysis_errors:
        print("\n=== BŁĘDY ANALIZY ===")
        for e in result.analysis_errors:
            print(f"  {e}")