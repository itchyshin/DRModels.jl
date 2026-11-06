"""Recheck code, auxiliary inputs, runtime and raw logs after test execution."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(sys.argv[1]);root=base/'julia-validation8';evidence=base/'evidence/validation8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest_map(rows):return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
actual={}
for name in json.loads((base/'candidate-manifest.json').read_text()):
 p=root/name
 if name=='test/runtests.jl':p=root/'test/runtests-original.jl'
 actual[name]=sha(p)
aux={name:sha(root/name) for name in json.loads((base/'auxiliary-manifest.json').read_text())}
logs=sorted(evidence.glob('shard-*.log'))
raw={str(p):subprocess.check_output(['sha256sum',str(p)],text=True).split()[0] for p in logs}
receipt={'candidate_aggregate_sha256':digest_map(actual),'auxiliary_aggregate_sha256':digest_map(aux),'timed_runner_sha256':sha(root/'test/runtests.jl'),'root_manifest_sha256':sha(root/'Manifest.toml'),'julia_executable_sha256':sha(Path('/home/snakagaw/hsq_work/julia-1.10.10/bin/julia')),'raw_log_sha256':raw}
(evidence/'post-run-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('POST_RUN_RECEIPT',len(actual),len(aux),len(raw))
