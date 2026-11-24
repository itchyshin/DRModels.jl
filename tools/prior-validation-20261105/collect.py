"""Retain compact completion transcripts with hashes of the complete raw logs."""
from pathlib import Path
import csv,hashlib,json,re,sys
base=Path(sys.argv[1]);out=base/'collected';out.mkdir(exist_ok=True)
receipts=[];timings=[]
for k in range(1,9):
 exitfile=base/f'shard-{k}.exit'
 if not exitfile.exists():continue
 log=base/f'shard-{k}.log';raw=log.read_bytes();text=raw.decode(errors='replace')
 selected=[l.rstrip(" \t") for l in text.splitlines() if any(s in l for s in ['PRIOR_TEST_START\t','PRIOR_TEST_DONE\t','DRModels tests passed','DRModels tests failed','R-parity suite skipped','Q-gate: JET type-stability','JET not in this environment','Test Failed','Error During Test'])]
 transcript=out/f'shard-{k}.txt';transcript.write_text('\n'.join(selected)+'\n')
 (out/f'shard-{k}.exit').write_bytes(exitfile.read_bytes())
 start=(base/f'shard-{k}.started').read_text().strip();end=(base/f'shard-{k}.ended').read_text().strip()
 receipts.append({'shard':k,'exit':int(exitfile.read_text()),'started_utc':start,'ended_utc':end,'raw_log_path':str(log),'raw_log_sha256':hashlib.sha256(raw).hexdigest(),'raw_bytes':len(raw),'transcript':transcript.name,'transcript_sha256':hashlib.sha256(transcript.read_bytes()).hexdigest(),'filter':'include markers, package verdict, JET/parity status and failure labels; trailing horizontal whitespace trimmed; raw log retained separately'})
 for stamp,name,seconds in re.findall(r'PRIOR_TEST_DONE\t([0-9.eE+-]+)\t([^\t\n]+)\t([0-9.eE+-]+)',text):
  timings.append({'shard':k,'file':name,'elapsed_seconds':float(seconds),'completed_epoch':float(stamp)})
(out/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
with (out/'file-timings.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['shard','file','elapsed_seconds','completed_epoch'],lineterminator='\n');writer.writeheader();writer.writerows(timings)
print('COLLECTED',len(receipts),'shards',len(timings),'completed files')
