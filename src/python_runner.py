"""Restricted, time/memory-bounded local Python evaluation (not a full sandbox).

No code runs in the web/GPU process. For hostile submissions use a hardened
container/VM instead; this runner is only for an explicitly consented local lab.
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .coding_problems import Problem

ALLOWED_IMPORTS = {
    "typing": ("List", "Dict", "Set", "Tuple", "Optional", "Any", "Deque", "DefaultDict"),
    "collections": ("Counter", "defaultdict", "deque"),
    "math": ("sqrt", "isqrt", "ceil", "floor", "gcd", "lcm", "inf", "factorial", "comb", "prod"),
    "heapq": ("heapify", "heappush", "heappop", "heappushpop", "heapreplace", "nlargest", "nsmallest"),
    "bisect": ("bisect_left", "bisect_right", "bisect", "insort", "insort_left", "insort_right"),
    "itertools": ("accumulate", "chain", "combinations", "permutations", "product", "groupby", "repeat", "zip_longest"),
    "functools": ("lru_cache", "cache", "reduce"),
}
SAFE_BUILTINS = (
    "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float",
    "frozenset", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "ord", "pow", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "zip", "Exception", "ValueError", "TypeError", "IndexError", "KeyError",
    "RuntimeError", "StopIteration", "ZeroDivisionError", "AssertionError",
)


def extract_code(text):
    # Only unwrap Markdown; never repair invalid Python or discard failing lines.
    text = text.strip().lstrip("\ufeff").strip()
    blocks = re.finditer(
        r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\r\n]*)\r?\n(.*?)^[ \t]*\1[ \t]*\r?$",
        text, flags=re.S | re.I | re.M,
    )
    for block in blocks:
        if block[2].strip().lower() in ("", "python", "python3", "py"):
            return block[3].strip()
    return text


def validate_code(code):
    if len(code) > 262144:
        raise ValueError("Generated source exceeds the runner's 256 KB safety limit.")
    tree = ast.parse(code, filename="<solution>")
    nodes = list(ast.walk(tree))
    if len(nodes) > 20000:
        raise ValueError("Generated program is too complex for the local runner.")
    for node in nodes:
        if isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal)):
            raise ValueError("Async/global/nonlocal operations are unavailable in the restricted runner.")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder namespace access is blocked.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private/introspection attribute access is blocked.")
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name.startswith("__") and node.name != "__init__":
            raise ValueError("Introspection/magic method definitions are blocked.")
        if isinstance(node, ast.Import):
            if any(alias.name not in ALLOWED_IMPORTS for alias in node.names):
                raise ValueError("Only the documented algorithm-library imports are allowed.")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS or any(alias.name != "*" and alias.name not in ALLOWED_IMPORTS[node.module] for alias in node.names):
                raise ValueError("This import is unavailable in the restricted runner.")
    return tree


def _windows_job(process):
    """Keep the child below 256 MiB and kill it if the parent releases the job."""
    import ctypes as c
    from ctypes import wintypes as w

    class Basic(c.Structure):
        _fields_ = [("PerProcessUserTimeLimit", c.c_longlong), ("PerJobUserTimeLimit", c.c_longlong),
                    ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", c.c_size_t),
                    ("MaximumWorkingSetSize", c.c_size_t), ("ActiveProcessLimit", w.DWORD),
                    ("Affinity", c.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]
    class IO(c.Structure):
        _fields_ = [(name, c.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
    class Extended(c.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO),
                    ("ProcessMemoryLimit", c.c_size_t), ("JobMemoryLimit", c.c_size_t),
                    ("PeakProcessMemoryUsed", c.c_size_t), ("PeakJobMemoryUsed", c.c_size_t)]
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x100 | 0x2000 | 0x8
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = 256 * 1024 * 1024
    if not handle or not kernel.SetInformationJobObject(handle, 9, c.byref(info), c.sizeof(info)) or not kernel.AssignProcessToJobObject(handle, int(process._handle)):
        if handle:
            kernel.CloseHandle(handle)
        raise OSError("Could not establish the runner memory/process limits.")
    return lambda: kernel.CloseHandle(handle)


def run_solution(text, problem: Problem, cancel=None, timeout=5, *, truncated=False):
    code = extract_code(text)
    result = {"status": "error", "code": code, "cases": [], "passed": 0, "total": len(problem.tests), "error": None}
    if truncated:
        result.update(status="invalid", error=(
            "Generation reached the model's context limit before completing the answer. "
            "The incomplete output was not executed. Try a coding model or less pruning."
        ))
        return result
    if not code:
        result.update(status="invalid", error="The model returned no Python code. Expected a complete solve(...) function.")
        return result
    try:
        validate_code(code)
    except SyntaxError as exc:
        source = (exc.text or "").strip()[:160]
        message = f"Generated output is not valid Python: {exc.msg} (line {exc.lineno}, column {exc.offset})."
        if source:
            message += f" Source: {source!r}."
        result.update(status="invalid", error=message + " Expected a complete solve(...) implementation.")
        return result
    except (ValueError, RecursionError) as exc:
        result.update(status="invalid", error=str(exc))
        return result
    payload = json.dumps({"code": code, "tests": [t.model_dump() for t in problem.tests], "normalize": problem.normalize,
                          "allowed_imports": ALLOWED_IMPORTS, "safe_builtins": SAFE_BUILTINS})
    runner = Path(__file__).with_name("python_worker.py")
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    with tempfile.TemporaryDirectory(prefix="pruning-python-") as cwd:
        # Windows venv python.exe is a launcher that spawns another process.
        # Use the base interpreter for this stdlib-only child so the one-process
        # Job Object limit cannot reject that redirection.
        executable = getattr(sys, "_base_executable", sys.executable)
        process = subprocess.Popen([executable, "-I", "-B", str(runner)], cwd=cwd, env=env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8",
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        release = lambda: None
        try:
            if os.name == "nt":
                release = _windows_job(process)
            started = time.monotonic()
            first = True
            while True:
                if cancel is not None and cancel.is_set():
                    result.update(status="cancelled", error="Execution cancelled.")
                    break
                if time.monotonic() - started > timeout:
                    result.update(status="timeout", error=f"Execution exceeded {timeout} seconds.")
                    break
                try:
                    stdout, stderr = process.communicate(input=payload if first else None, timeout=.1)
                    if process.returncode or len(stdout) > 131072:
                        detail = stderr.strip().splitlines()[-1][:300] if stderr.strip() else "memory limit or excessive output"
                        result["error"] = f"Runner terminated: {detail}"
                    else:
                        result.update(json.loads(stdout))
                    break
                except subprocess.TimeoutExpired:
                    first = False
        except (OSError, ValueError) as exc:
            result["error"] = str(exc)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()
            release()
    return result


def compare_outcomes(baseline, pruned):
    if baseline["status"] != "ok" or pruned["status"] != "ok":
        return {"score": 0, "reason": "At least one solution failed validation/execution. Errors are not matching outcomes."}
    pairs = list(zip(baseline["cases"], pruned["cases"]))
    same = len(baseline["cases"]) == len(pruned["cases"]) and bool(pairs) and all(a["output_key"] == b["output_key"] for a, b in pairs)
    return {"score": int(same), "reason": "Both programs returned the same outcomes on every test." if same else "Programs returned different outcomes on at least one test."}
