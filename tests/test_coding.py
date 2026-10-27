import threading

import pytest

from src.coding_problems import CATALOG, Problem, get_problems, new_problem
from src.python_runner import compare_outcomes, extract_code, run_solution


TWO_SUM = """from typing import List
def solve(nums: List[int], target: int) -> List[int]:
    seen = {}
    for i, value in enumerate(nums):
        if target - value in seen:
            return [i, seen[target - value]]
        seen[value] = i
"""


def test_real_python_and_order_normalization():
    result = run_solution(TWO_SUM, CATALOG[0])
    assert result["status"] == "ok", result["error"]
    assert result["passed"] == result["total"] == 4
    assert compare_outcomes(result, result)["score"] == 1


def test_wrong_but_equal_is_not_labeled_correct():
    a = run_solution("def solve(nums, target): return [0, 1]", CATALOG[0])
    assert a["passed"] < a["total"]
    assert compare_outcomes(a, a)["score"] == 1
    correct = run_solution(TWO_SUM, CATALOG[0])
    assert compare_outcomes(a, correct)["score"] == 0


@pytest.mark.parametrize("source", [
    "import os\ndef solve(*args): return os.environ",
    "def solve(*args): return ().__class__",
    "def solve(*args): return open('forbidden', 'w')",
    "def solve(*args): return eval('2+2')",
    "def solve(*args):\n    return __import__('os')",
    "def solve(:",
])
def test_restricted_execution_and_failures_are_zero(source):
    result = run_solution(source, CATALOG[0])
    assert result["status"] in ("error", "invalid"), result
    assert compare_outcomes(result, result)["score"] == 0


def test_infinite_loop_timeout():
    result = run_solution("def solve(*args):\n    while True: pass", CATALOG[0], timeout=.4)
    assert result["status"] == "timeout"


def test_memory_limit():
    result = run_solution("def solve(*args): return [0] * 1000000000", CATALOG[0])
    assert result["status"] == "error"


def test_cancellation():
    cancel = threading.Event()
    cancel.set()
    assert run_solution(TWO_SUM, CATALOG[0], cancel)["status"] == "cancelled"


def test_catalog_has_only_executable_coding_contracts():
    assert len(CATALOG) == len({p.id for p in CATALOG}) == 20
    assert all("solve(" in p.description and p.tests for p in CATALOG)
    assert all(p.category == "strings" for p in get_problems("strings"))
    added = new_problem([p.id for p in CATALOG[:10]])
    assert added.id not in {p.id for p in CATALOG[:10]}
    with pytest.raises(ValueError):
        new_problem([p.id for p in CATALOG])


def test_extract_fenced_code_and_imports():
    code = "from collections import Counter\ndef solve(nums): return len(Counter(nums)) != len(nums)"
    assert extract_code(f"```python\n{code}\n```") == code
    result = run_solution(f"```python\n{code}\n```", CATALOG[3])
    assert result["passed"] == result["total"] == 4


@pytest.mark.parametrize("fence,language,newline", [
    ("```", "python3", "\n"),
    ("```", "PYTHON", "\r\n"),
    ("```", "py", "\n"),
    ("```", "", "\n"),
    ("~~~", "python", "\n"),
    ("````", " python ", "\n"),
])
def test_extract_common_markdown_fences(fence, language, newline):
    code = "def solve(x):\n    return x"
    wrapped = f"Here is the implementation:{newline}{fence}{language}{newline}{code}{newline}{fence}{newline}Explanation follows."
    assert extract_code(wrapped) == code


def test_extract_ignores_non_python_fences():
    code = "def solve(x): return x"
    text = f'```json\n{{"example": true}}\n```\n\n```python\n{code}\n```'
    assert extract_code(text) == code
    assert extract_code(f"\ufeff```python\n{code}\n```") == code


@pytest.mark.parametrize("source,expected", [
    ("", "no Python code"),
    ("```python\n\n```", "no Python code"),
    ("Solve this algorithm problem in Python 3.", "not valid Python"),
    ("def solve(x):\n    return (", "line 2"),
])
def test_invalid_generation_has_actionable_diagnostics(source, expected, monkeypatch):
    def unexpected_process(*args, **kwargs):
        pytest.fail("Invalid source must not start a child process")
    monkeypatch.setattr("src.python_runner.subprocess.Popen", unexpected_process)
    result = run_solution(source, CATALOG[0])
    assert result["status"] == "invalid"
    assert expected in result["error"]
    assert "<unknown>" not in result["error"]
    assert "solve(...)" in result["error"]
    assert result["cases"] == [] and result["passed"] == 0
    assert compare_outcomes(result, result)["score"] == 0


def test_syntax_diagnostic_includes_offending_source():
    result = run_solution("def solve(:", CATALOG[0])
    assert "Source: 'def solve(:'" in result["error"]
    assert "line 1, column" in result["error"]


def test_context_exhaustion_never_executes_even_parseable_code(monkeypatch):
    def unexpected_process(*args, **kwargs):
        pytest.fail("Context-exhausted answers must not be executed")
    monkeypatch.setattr("src.python_runner.subprocess.Popen", unexpected_process)
    result = run_solution(TWO_SUM, CATALOG[0], truncated=True)
    assert result["status"] == "invalid"
    assert "context limit" in result["error"] and "not executed" in result["error"]
    assert result["code"] == TWO_SUM.strip()
    assert not result["cases"] and result["passed"] == 0
    assert compare_outcomes(result, result)["score"] == 0


def test_markdown_extraction_does_not_repair_code_or_relax_restrictions():
    malformed = "def solve(x):\n    return ("
    assert extract_code(f"```python3\n{malformed}\n```") == malformed
    assert run_solution(f"~~~python\n{malformed}\n~~~", CATALOG[0])["status"] == "invalid"
    forbidden = "import os\ndef solve(x): return x"
    assert run_solution(f"```python3\n{forbidden}\n```", CATALOG[0])["status"] == "invalid"


def test_zero_empty_and_null_are_real_outputs():
    p = Problem(id="null", title="Nullable", description="Implement solve(x). Return x.",
                tests=[{"args":[None], "expected":None}, {"args":[0],"expected":0}, {"args":[[]],"expected":[]}])
    result = run_solution("def solve(x): return x", p)
    assert result["passed"] == 3


def test_star_import_only_exposes_the_restricted_exports():
    result = run_solution("from typing import *\nfrom collections import *\ndef solve(nums): return len(Counter(nums)) != len(nums)", CATALOG[3])
    assert result["passed"] == result["total"] == 4
    result = run_solution("from collections import *\ndef solve(nums): return globals()", CATALOG[3])
    assert result["status"] == "error"
