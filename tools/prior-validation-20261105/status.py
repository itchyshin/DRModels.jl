"""Read bounded shard progress without interpreting partial output as a pass."""
from pathlib import Path
import json,re,sys,time
root=Path(sys.argv[1]); out=[]
for k in range(1,9):
 p=root/f'shard-{k}.log'; text=p.read_text(errors='replace') if p.exists() else ''
 starts=re.findall(r'PRIOR_TEST_START\t([0-9.eE+-]+)\t([^\n]+)',text,re.M)
 done=re.findall(r'PRIOR_TEST_DONE\t([0-9.eE+-]+)\t([^\t\n]+)\t([\d.eE+-]+)',text,re.M)
 exitfile=root/f'shard-{k}.exit'
 completed={r[1] for r in done}; active=[r for r in starts if r[1] not in completed]
 out.append({'shard':k,'completed_files':len(done),'exit':int(exitfile.read_text()) if exitfile.exists() else None,'active_file':active[-1][1] if active else None,'active_seconds':round(time.time()-float(active[-1][0])) if active else None,'recent_done':done[-1][1] if done else None})
print(json.dumps(out,indent=2))
