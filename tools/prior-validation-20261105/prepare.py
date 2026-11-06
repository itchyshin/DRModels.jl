"""Prepare a disposable timed runner without changing test assertions or order."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]); output=Path(sys.argv[2])
original=(root/'test/runtests.jl').read_text()
needle='''        include(path)
        _check_blas_restored(path)'''
replacement='''        println(stderr, "PRIOR_TEST_START\\t", time(), "\\t", path)
        flush(stderr)
        _prior_elapsed = @elapsed include(path)
        _check_blas_restored(path)
        println(stderr, "PRIOR_TEST_DONE\\t", time(), "\\t", path, "\\t", _prior_elapsed)
        flush(stderr)'''
assert original.count(needle)==1
output.mkdir(parents=True,exist_ok=True)
(output/'runtests-timed.jl').write_text(original.replace(needle,replacement))
files=sorted([p for base in ['src','test'] for p in (root/base).rglob('*') if p.is_file()]+[root/'Project.toml'],key=lambda p:str(p.relative_to(root)))
rows={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(output/'candidate-manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
(output/'runner-receipt.json').write_text(json.dumps({'original_sha256':hashlib.sha256(original.encode()).hexdigest(),'instrumented_sha256':hashlib.sha256((output/'runtests-timed.jl').read_bytes()).hexdigest(),'replacement_count':1,'changes':'stderr start/end markers and @elapsed around the existing include(path); assertion bytes and selection logic unchanged'},indent=2)+'\n')
