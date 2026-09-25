"""Standalone test runner for CPU testing in offline / minimal environments."""
import importlib.util
import inspect
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Provide a standard pytest mock if pytest is not installed
if "pytest" not in sys.modules:
    try:
        import pytest
    except ImportError:
        import types
        mock_pytest = types.ModuleType("pytest")
        mock_pytest.__spec__ = importlib.machinery.ModuleSpec("pytest", None)

        class _Approx:
            def __init__(self, exp, rel=1e-5, abs_tol=1e-9):
                self.exp = exp
                self.rel = rel
                self.abs_tol = abs_tol
            def __eq__(self, other):
                diff = abs(other - self.exp)
                tol = max(self.rel * abs(self.exp), self.abs_tol)
                return diff <= tol
            def __repr__(self):
                return f"approx({self.exp})"

        def approx(expected, rel=1e-5, abs_tol=1e-9):
            return _Approx(expected, rel, abs_tol)

        class mark:
            @staticmethod
            def parametrize(argnames, argvalues):
                def decorator(fn):
                    fn._pytest_parametrized = (argnames, argvalues)
                    return fn
                return decorator

        def fixture(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        mock_pytest.approx = approx
        mock_pytest.mark = mark
        mock_pytest.fixture = fixture
        sys.modules["pytest"] = mock_pytest


def run_all_tests():
    tests_dir = os.path.join(ROOT, "tests")
    test_files = sorted([f for f in os.listdir(tests_dir) if f.startswith("test_") and f.endswith(".py")])
    
    total_passed = 0
    total_failed = 0
    failed_details = []

    print(f"===================== RUNNING VOCR TESTS ({len(test_files)} files) =====================")
    start_time = time.time()

    for fname in test_files:
        fpath = os.path.join(tests_dir, fname)
        mod_name = fname[:-3]
        spec = importlib.util.spec_from_file_location(mod_name, fpath)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[{fname}] FAILED TO IMPORT: {e}")
            total_failed += 1
            failed_details.append((fname, "import", str(e)))
            continue

        test_funcs = [
            (name, fn) for name, fn in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_") and fn.__module__ == mod_name
        ]
        
        file_passed = 0
        file_failed = 0
        for name, fn in test_funcs:
            param_meta = getattr(fn, "_pytest_parametrized", None)
            if param_meta:
                argnames, argvalues = param_meta
                for val in argvalues:
                    try:
                        if isinstance(val, (tuple, list)) and not isinstance(val, str):
                            fn(*val)
                        else:
                            fn(val)
                        file_passed += 1
                        total_passed += 1
                    except Exception as e:
                        file_failed += 1
                        total_failed += 1
                        failed_details.append((fname, f"{name}({val})", str(e)))
            else:
                try:
                    # Support parameterless tests or fixture injections
                    sig = inspect.signature(fn)
                    if len(sig.parameters) == 0:
                        fn()
                    else:
                        # Simple injection if fixture named tokenizer
                        kwargs = {}
                        if "tokenizer" in sig.parameters:
                            tok_fn = getattr(mod, "get_tokenizer", None) or getattr(mod, "tokenizer", None)
                            if callable(tok_fn):
                                kwargs["tokenizer"] = tok_fn()
                        fn(**kwargs)
                    file_passed += 1
                    total_passed += 1
                except Exception as e:
                    file_failed += 1
                    total_failed += 1
                    failed_details.append((fname, name, str(e)))

        status = "PASSED" if file_failed == 0 else f"FAILED ({file_failed} failed)"
        print(f"  {fname:<25} : {file_passed} passed, {file_failed} failed -> {status}")

    elapsed = time.time() - start_time
    print(f"==========================================================================")
    print(f"SUMMARY: {total_passed} passed, {total_failed} failed in {elapsed:.2f}s")
    if total_failed > 0:
        print("\nFailures:")
        for fname, name, err in failed_details:
            print(f"  - [{fname}] {name}: {err}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
