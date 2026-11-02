"""Minimal pytest-compatible runner for offline environments.

The integration sandbox has no network access, so real pytest cannot be
installed. This shim runs pytest-style test files that use only:
  - plain `assert`
  - the `tmp_path` fixture (pathlib.Path to a fresh temp dir)
  - the `monkeypatch` fixture (setattr / delattr / setitem / setenv / undo)
  - `pytest.raises`, `pytest.approx`, `pytest.fail`, `pytest.skip`,
    `pytest.mark.parametrize`, `pytest.importorskip`, `pytest.fixture`

It is NOT a pytest replacement; it exists so the integration audit can
execute the specialists' test suites offline. Run real pytest on Windows.

Usage: python3 tools/minipytest.py tests/
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path


# --------------------------------------------------------------------------
# fake `pytest` module
# --------------------------------------------------------------------------
class Skipped(Exception):
    pass


class Failed(Exception):
    pass


class _Raises:
    def __init__(self, expected, match=None):
        self.expected = expected
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise Failed(f"DID NOT RAISE {self.expected}")
        if not issubclass(exc_type, self.expected):
            return False
        self.value = exc
        if self.match is not None:
            import re

            if not re.search(self.match, str(exc)):
                raise Failed(f"pattern {self.match!r} not found in {str(exc)!r}")
        return True


class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = 1e-6 if rel is None else rel
        self.abs = 1e-12 if abs is None else abs

    def _close(self, a, b):
        return abs(a - b) <= max(self.abs, self.rel * max(abs(a), abs(b)))

    def __eq__(self, other):
        try:
            if hasattr(self.expected, "__iter__") and not isinstance(self.expected, (str, bytes)):
                exp = list(self.expected)
                got = list(other)
                return len(exp) == len(got) and all(self._close(float(a), float(b)) for a, b in zip(got, exp))
            return self._close(float(other), float(self.expected))
        except Exception:
            return NotImplemented

    def __repr__(self):
        return f"approx({self.expected!r})"


class _Mark:
    def parametrize(self, argnames, argvalues, **kw):
        names = [n.strip() for n in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

        def deco(fn):
            cases = getattr(fn, "_mini_params", [])
            cases.append((names, list(argvalues)))
            fn._mini_params = cases
            return fn

        return deco

    def __getattr__(self, _name):
        def deco(*a, **k):
            if len(a) == 1 and callable(a[0]) and not k:
                return a[0]
            return lambda fn: fn

        return deco


def _fixture(*a, **k):
    if len(a) == 1 and callable(a[0]) and not k:
        a[0]._mini_fixture = True
        return a[0]

    def deco(fn):
        fn._mini_fixture = True
        return fn

    return deco


def _importorskip(name, *a, **k):
    try:
        return importlib.import_module(name)
    except Exception:
        raise Skipped(f"module {name} not available")


import importlib

pytest_stub = types.ModuleType("pytest")
pytest_stub.raises = _Raises
pytest_stub.approx = _Approx
pytest_stub.mark = _Mark()
pytest_stub.fixture = _fixture
pytest_stub.skip = lambda msg="", **k: (_ for _ in ()).throw(Skipped(msg))
pytest_stub.fail = lambda msg="", **k: (_ for _ in ()).throw(Failed(msg))
pytest_stub.importorskip = _importorskip
pytest_stub.warns = lambda *a, **k: _Raises(Warning)
pytest_stub.Skipped = Skipped
sys.modules.setdefault("pytest", pytest_stub)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
class MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=None, raising=True):
        if isinstance(target, str):
            mod_name, _, attr = target.rpartition(".")
            target, name, value = importlib.import_module(mod_name), attr, name
        old = getattr(target, name, _MISSING)
        self._undo.append((target, name, old))
        setattr(target, name, value)

    def delattr(self, target, name=None, raising=True):
        old = getattr(target, name, _MISSING)
        self._undo.append((target, name, old))
        if old is not _MISSING:
            delattr(target, name)

    def setitem(self, dic, key, value):
        old = dic.get(key, _MISSING)
        self._undo.append((dic, key, old))
        dic[key] = value

    def delitem(self, dic, key, raising=True):
        old = dic.get(key, _MISSING)
        self._undo.append((dic, key, old))
        dic.pop(key, None)

    def setenv(self, key, value, prepend=None):
        self.setitem(os.environ, key, str(value))

    def delenv(self, key, raising=True):
        self.delitem(os.environ, key)

    def chdir(self, path):
        cwd = os.getcwd()
        self._undo.append(("__chdir__", None, cwd))
        os.chdir(path)

    def undo(self):
        for target, name, old in reversed(self._undo):
            if target == "__chdir__":
                os.chdir(old)
            elif isinstance(target, dict):
                if old is _MISSING:
                    target.pop(name, None)
                else:
                    target[name] = old
            else:
                if old is _MISSING:
                    try:
                        delattr(target, name)
                    except Exception:
                        pass
                else:
                    setattr(target, name, old)
        self._undo.clear()


class _Missing:
    pass


_MISSING = _Missing()


def _build_args(fn, module, tmp_root, mps):
    """Resolve fixture arguments for a test function."""
    kwargs = {}
    for pname in inspect.signature(fn).parameters:
        if pname == "tmp_path":
            d = Path(tempfile.mkdtemp(dir=tmp_root))
            kwargs[pname] = d
        elif pname == "tmp_path_factory":
            kwargs[pname] = _TmpFactory(tmp_root)
        elif pname == "monkeypatch":
            mp = MonkeyPatch()
            mps.append(mp)
            kwargs[pname] = mp
        elif pname == "capsys":
            kwargs[pname] = _CapSys()
        else:
            fixture = getattr(module, pname, None)
            if fixture is not None and getattr(fixture, "_mini_fixture", False):
                sub = _build_args(fixture, module, tmp_root, mps)
                val = fixture(**sub)
                if inspect.isgenerator(val):
                    val = next(val)
                kwargs[pname] = val
            else:
                raise Skipped(f"unsupported fixture {pname!r}")
    return kwargs


class _TmpFactory:
    def __init__(self, root):
        self.root = root

    def mktemp(self, name, numbered=True):
        return Path(tempfile.mkdtemp(prefix=str(name), dir=self.root))


class _CapSys:
    def readouterr(self):
        return types.SimpleNamespace(out="", err="")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def run(paths):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(p.rglob("test_*.py")))
        else:
            files.append(p)

    tmp_root = Path(tempfile.mkdtemp(prefix="minipytest-"))
    passed = failed = skipped = 0
    failures = []
    collect_errors = []

    for f in files:
        try:
            mod = _load_module(f)
        except Skipped as e:
            print(f"SKIP MODULE {f}: {e}")
            continue
        except Exception:
            collect_errors.append((str(f), traceback.format_exc()))
            print(f"COLLECT ERROR {f}")
            continue

        for name in sorted(vars(mod)):
            fn = getattr(mod, name)
            if not (name.startswith("test_") and callable(fn)):
                continue
            param_sets = getattr(fn, "_mini_params", None)
            cases = [({}, "")]
            if param_sets:
                cases = []
                names, values = param_sets[0]
                for v in values:
                    v = v if isinstance(v, (tuple, list)) else (v,)
                    cases.append((dict(zip(names, v)), f"[{','.join(map(str, v))}]"))
            for extra, label in cases:
                mps = []
                try:
                    kwargs = _build_args(fn, mod, tmp_root, mps)
                    kwargs.update(extra)
                    fn(**kwargs)
                    passed += 1
                except Skipped as e:
                    skipped += 1
                    print(f"SKIP {f.name}::{name}{label}: {e}")
                except Exception:
                    failed += 1
                    failures.append((f"{f.name}::{name}{label}", traceback.format_exc()))
                    print(f"FAIL {f.name}::{name}{label}")
                finally:
                    for mp in mps:
                        mp.undo()

    shutil.rmtree(tmp_root, ignore_errors=True)

    print("\n" + "=" * 70)
    for label, tb in failures:
        print(f"--- {label} ---")
        print(tb.strip()[-2000:])
        print()
    for label, tb in collect_errors:
        print(f"--- COLLECT {label} ---")
        print(tb.strip()[-1500:])
        print()
    print(f"passed={passed} failed={failed} skipped={skipped} collect_errors={len(collect_errors)}")
    return 1 if (failed or collect_errors) else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(run(sys.argv[1:] or ["tests"]))
