"""Private child-process entry point. Never imported by generated code."""

import builtins
import copy
import importlib
import json
import os
import sys
import types


def main():
    if os.name != "nt":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    data = json.loads(sys.stdin.read())
    # Policy is supplied by the trusted parent, never by generated source.
    ALLOWED_IMPORTS = data["allowed_imports"]
    SAFE_BUILTINS = data["safe_builtins"]
    modules = {name: types.SimpleNamespace(**{attr: getattr(importlib.import_module(name), attr) for attr in attrs})
               for name, attrs in ALLOWED_IMPORTS.items()}

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in modules or any(attr != "*" and attr not in ALLOWED_IMPORTS[name] for attr in fromlist):
            raise ImportError("Import is not allowed.")
        return modules[name]

    safe = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    safe.update({"__import__": safe_import, "__build_class__": builtins.__build_class__, "print": lambda *a, **kw: None})
    compiled = compile(data["code"], "<solution>", "exec")
    cases, passed = [], 0
    for case in data["tests"]:
        namespace = {"__builtins__": dict(safe), "__name__": "solution", **vars(modules["typing"])}
        try:
            exec(compiled, namespace)
            solve = namespace.get("solve")
            if solve is None and "Solution" in namespace:
                solve = getattr(namespace["Solution"](), "solve", None)
            if not callable(solve):
                raise ValueError("Expected a top-level function solve(*args).")
            output = solve(*copy.deepcopy(case["args"]))
            if data["normalize"] == "unordered" and isinstance(output, (list, tuple)):
                output = sorted(output)
            expected = case["expected"]
            if data["normalize"] == "unordered" and isinstance(expected, list):
                expected = sorted(expected)
            key = json.dumps(output, sort_keys=True, allow_nan=False, separators=(",", ":"))
            if len(key) > 16000:
                raise ValueError("Return value is too large.")
            correct = key == json.dumps(expected, sort_keys=True, allow_nan=False, separators=(",", ":"))
            passed += int(correct)
            cases.append({"output": output, "output_key": key, "correct": correct})
        except Exception as exc:
            print(json.dumps({"status": "error", "cases": cases, "passed": passed, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}))
            return
    print(json.dumps({"status": "ok", "cases": cases, "passed": passed, "error": None}, allow_nan=False))


if __name__ == "__main__":
    main()
