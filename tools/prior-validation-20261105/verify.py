"""Verify full default-suite receipts; missing or partial runs fail closed."""
from pathlib import Path
import hashlib,json,re,sys
START=re.compile(r'PRIOR_TEST_START\t([0-9.eE+-]+)\t([^\n]+)')
DONE=re.compile(r'PRIOR_TEST_DONE\t([0-9.eE+-]+)\t([^\t\n]+)\t([0-9.eE+-]+)')

def check_shard(expected,text,code):
    assert code==0, f'exit {code}'
    starts=[m[1] for m in START.findall(text)]
    done=[m[1] for m in DONE.findall(text)]
    assert starts==expected, ('starts',starts,expected)
    assert done==expected, ('completions',done,expected)
    assert 'DRModels tests passed' in text, 'Pkg.test success absent'
    assert 'R-parity suite skipped' in text, 'gated parity status absent'
    assert 'Q-gate: JET type-stability' in text, 'JET gate status absent'
    assert 'JET not in this environment' not in text, 'JET did not run'
    assert not re.search(r'Test Failed|Error During Test|Testing DRModels tests failed',text)
    return len(done)

def check_payload(receipt,raw,code):
    assert code==receipt['exit'], 'exit receipt mismatch'
    assert hashlib.sha256(raw).hexdigest()==receipt['transcript_sha256'], 'transcript hash mismatch'

def digest_map(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def check_provenance(evidence):
    home=evidence.parent
    root=Path(__file__).resolve().parents[2]
    candidate=json.loads((home/'candidate-manifest.json').read_text())
    auxiliary=json.loads((home/'auxiliary-manifest.json').read_text())
    for name,digest in (candidate|auxiliary).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, name
    names=sorted(k for k in candidate if k=='Project.toml' or (k.startswith('src/') and k.endswith('.jl')))
    source=hashlib.sha256(''.join(candidate[k]+'  '+k+'\n' for k in names).encode()).hexdigest()
    assert source=='a30e347285f925af2e128ed6350fafe4d6b9035668444606bfa2fcea13e26653'
    post=json.loads((home/'post-run-provenance.json').read_text())
    assert post['candidate_aggregate_sha256']==digest_map(candidate)
    assert post['auxiliary_aggregate_sha256']==digest_map(auxiliary)
    runner=json.loads((home/'runner-receipt.json').read_text())
    assert post['timed_runner_sha256']==runner['instrumented_sha256']
    assert hashlib.sha256((root/'test/runtests.jl').read_bytes()).hexdigest()==runner['original_sha256']
    env=json.loads((home/'environment-provenance.json').read_text())
    one=lambda suffix: next(v for k,v in env.items() if k.endswith(suffix) and not k.startswith('/tmp/'))
    assert post['root_manifest_sha256']==one('/julia-validation8/Manifest.toml')
    assert hashlib.sha256((home/'runtime-root-manifest.toml').read_bytes()).hexdigest()==post['root_manifest_sha256']
    assert post['julia_executable_sha256']==one('/bin/julia')
    for kind in ['Project','Manifest']:
        entries=[v for k,v in env.items() if k.startswith('/tmp/') and k.endswith('/'+kind+'.toml')]
        assert len(entries)==8, 'incomplete sandbox provenance'
        expected=set(entries)
        assert len(expected)==1
        assert hashlib.sha256((home/('runtime-test-'+kind.lower()+'.toml')).read_bytes()).hexdigest() in expected
    return post

def negative_controls():
    f=['test_a.jl','test_b.jl']
    good=''.join(f'PRIOR_TEST_START\t1.7e9\t{x}\nPRIOR_TEST_DONE\t1.7e9\t{x}\t1.0\n' for x in f)+'DRModels tests passed\nR-parity suite skipped\nQ-gate: JET type-stability\n'
    assert check_shard(f,good,0)==2
    bad=[(good,124),(good.replace('PRIOR_TEST_DONE\t1.7e9\ttest_b.jl\t1.0\n',''),0),(good.replace('test_b.jl','test_a.jl'),0),(good.replace('DRModels tests passed',''),0)]
    for text,code in bad:
        try:check_shard(f,text,code)
        except AssertionError:pass
        else:raise AssertionError('negative control accepted')
    raw=good.encode();receipt={'exit':0,'transcript_sha256':hashlib.sha256(raw).hexdigest()}
    check_payload(receipt,raw,0)
    for mutant,code in [(raw+b' ',0),(raw,124)]:
        try:check_payload(receipt,mutant,code)
        except AssertionError:pass
        else:raise AssertionError('hash/exit negative control accepted')
    print('NEGATIVE_CONTROLS_PASS 6/6; scientific-notation timestamps accepted')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--self-test':negative_controls();raise SystemExit(0)
    evidence=Path(sys.argv[1]); plan=json.loads(Path(sys.argv[2]).read_text())
    allfiles=sum(plan.values(),[])
    assert len(allfiles)==len(set(allfiles))==554
    post=check_provenance(evidence)
    receipts=json.loads((evidence/'receipts.json').read_text())
    assert len(receipts)==8 and {r['shard'] for r in receipts}==set(range(1,9))
    for k in range(1,9):
        receipt=next(r for r in receipts if r['shard']==k)
        assert receipt['transcript']==f'shard-{k}.txt'
        assert post['raw_log_sha256'][receipt['raw_log_path']]==receipt['raw_log_sha256']
        raw=(evidence/f'shard-{k}.txt').read_bytes()
        code=int((evidence/f'shard-{k}.exit').read_text())
        check_payload(receipt,raw,code)
        check_shard(plan[str(k)],raw.decode(errors='replace'),code)
    negative_controls()
    print('FULL_DEFAULT_SUITE_PASS 8/8 shards; 554/554 selected files; JET executed; opt-in parity off')
