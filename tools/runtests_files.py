"""Which test files does test/runtests.jl run? (shared by the tools/check_*.py guards)

runtests.jl no longer lists test files by hand: it auto-discovers test/test_*.jl,
minus the hand-maintained _TEST_EXCLUDE list, plus _TEST_ORDER (which runs first).
The check_*.py guards used to grep runtests.jl for include("...") calls; that
now finds almost nothing, so they call wired_test_files() instead.
"""
import os
import re


def _list(src, name):
    m = re.search(r"const " + name + r" = \[(.*?)\n\]", src, re.S)
    if not m:
        return []
    body = "\n".join(l.split("#", 1)[0] for l in m.group(1).splitlines())
    return re.findall(r'"([^"]+)"', body)


def wired_test_files(test_dir="test"):
    """Basenames run through the default suite: _TEST_ORDER + discovered test_*.jl
    (minus _TEST_EXCLUDE) + any literal include("...") left in runtests.jl
    (the gated JET / parity files)."""
    rt = os.path.join(test_dir, "runtests.jl")
    src = open(rt, encoding="utf-8").read()
    order, excl = _list(src, "_TEST_ORDER"), set(_list(src, "_TEST_EXCLUDE"))
    disc = [f for f in os.listdir(test_dir)
            if re.match(r"^test_.*\.jl$", f) and f not in order and f not in excl]
    code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
    literal = [os.path.basename(m) for m in re.findall(r'(?<![_\w])include\(\s*"([^"]+)"\s*\)', code)]
    return set(order) | set(disc) | set(literal)
