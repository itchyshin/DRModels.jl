#!/usr/bin/env python3
"""Path-portable source manifests for runner receipts.

Runner receipts record absolute paths on the machine that produced them (for
example /home/snakagaw/claude-606b/after/src/DRModels.jl). The receipt bytes are
evidence and stay as written. Integrity rests on the per-file sha256, so the
validators re-key both the recorded and the local manifests by
(root label, repo-relative path) and compare those. The recorded Julia root is
the directory that holds the loaded src/DRModels.jl; the recorded R root is the
directory that holds the recorded NAMESPACE.
"""
from pathlib import PurePosixPath


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def loaded_root(source):
    """Root of the recorded loaded Julia source; it must be <root>/src/DRModels.jl."""
    _require(isinstance(source, str), 'loaded Julia source')
    path = PurePosixPath(source)
    _require(path.is_absolute() and path.parts[-2:] == ('src', 'DRModels.jl'), 'loaded Julia source')
    return str(path.parent.parent)


def recorded(manifest, julia_root):
    """Re-key a recorded absolute-path manifest as 'julia:<rel>' / 'R:<rel>'."""
    _require(isinstance(manifest, dict) and manifest, 'source manifest')
    jprefix = julia_root + '/'
    names = [k for k in manifest if PurePosixPath(k).name == 'NAMESPACE' and not k.startswith(jprefix)]
    _require(len(names) == 1, 'recorded R root')
    rprefix = str(PurePosixPath(names[0]).parent) + '/'
    out = {}
    for key, value in manifest.items():
        if key.startswith(jprefix):
            out['julia:' + key[len(jprefix):]] = value
        elif key.startswith(rprefix):
            out['R:' + key[len(rprefix):]] = value
        else:
            _require(False, 'source path outside recorded roots')
    _require(len(out) == len(manifest), 'source manifest keys')
    return out


def local(rfiles, rroot, jfiles, jroot, digest):
    """The same re-keyed manifest computed from the local R and Julia checkouts."""
    out = {'R:' + p.relative_to(rroot).as_posix(): digest(p) for p in rfiles}
    out.update({'julia:' + p.relative_to(jroot).as_posix(): digest(p) for p in jfiles})
    return out
