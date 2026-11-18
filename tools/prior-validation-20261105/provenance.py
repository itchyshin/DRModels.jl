"""Capture the active Pkg.test environments and executable for this run."""
from pathlib import Path
import hashlib,json,re,shutil,sys
base=Path(sys.argv[1]); project=base/'julia-validation8'; evidence=base/'evidence/validation8'
rows={}
for p in [project/'Manifest.toml',project/'Project.toml',Path('/home/snakagaw/hsq_work/julia-1.10.10/bin/julia')]:
 rows[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
for k in range(1,9):
 text=(evidence/f'shard-{k}.log').read_text(errors='replace')
 m=re.search(r'Status `([^`]+/Project.toml)`',text)
 if m:
  folder=Path(m.group(1)).parent
  for name in ['Project.toml','Manifest.toml']:
   source=folder/name
   if source.exists():
    destination=evidence/f'shard-{k}-{name}'
    shutil.copyfile(source,destination)
    rows[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
(evidence/'environment-provenance.json').write_text(json.dumps(rows,indent=2)+'\n')
print('ENVIRONMENT_RECEIPTS',len(rows))
