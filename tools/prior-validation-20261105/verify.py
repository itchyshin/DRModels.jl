"""Verify full default-suite receipts; missing or partial runs fail closed."""
from pathlib import Path
import json,re,sys
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

def negative_controls():
    f=['test_a.jl','test_b.jl']
    good=''.join(f'PRIOR_TEST_START\t1.7e9\t{x}\nPRIOR_TEST_DONE\t1.7e9\t{x}\t1.0\n' for x in f)+'DRModels tests passed\nR-parity suite skipped\nQ-gate: JET type-stability\n'
    assert check_shard(f,good,0)==2
    bad=[(good,124),(good.replace('PRIOR_TEST_DONE\t1.7e9\ttest_b.jl\t1.0\n',''),0),(good.replace('test_b.jl','test_a.jl'),0),(good.replace('DRModels tests passed',''),0)]
    for text,code in bad:
        try:check_shard(f,text,code)
        except AssertionError:pass
        else:raise AssertionError('negative control accepted')
    print('NEGATIVE_CONTROLS_PASS 4/4; scientific-notation timestamps accepted')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--self-test':negative_controls();raise SystemExit(0)
    evidence=Path(sys.argv[1]); plan=json.loads(Path(sys.argv[2]).read_text())
    allfiles=sum(plan.values(),[])
    assert len(allfiles)==len(set(allfiles))==554
    for k in range(1,9):
        text=(evidence/f'shard-{k}.log').read_text(errors='replace')
        code=int((evidence/f'shard-{k}.exit').read_text())
        check_shard(plan[str(k)],text,code)
    negative_controls()
    print('FULL_DEFAULT_SUITE_PASS 8/8 shards; 554/554 selected files; JET executed; opt-in parity off')
