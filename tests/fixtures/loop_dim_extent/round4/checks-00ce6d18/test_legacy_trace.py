import hashlib,json,os,pathlib,shutil
import pytest,yaml
from solar.einsum_dsl.cli_einsum import translate_with_einsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
R=pathlib.Path(__file__).parent
OLD=R.parent/'checks-cf6251bc'

@pytest.mark.parametrize('name',['keyword_order','last_wins_swapped'])
def test_legacy_saved_trace_is_correct_or_refused(name):
    src=OLD/'runs/order-head'/name
    dest=R/'runs'/('legacy-'+os.environ['REVIEW_ROLE'])/name
    dest.mkdir(parents=True,exist_ok=True)
    shutil.copytree(src/'dsl',dest/'dsl')
    oldrecord=json.loads((src/'record.json').read_text())
    metadata=yaml.safe_load(next((dest/'dsl').glob('*.yaml')).read_text())
    oldmanifest=json.loads((OLD/'SHA256SUMS.json').read_text())
    verified={}
    for p in (src/'dsl').iterdir():
        if p.is_file():
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            assert h==oldmanifest[str(p.relative_to(OLD))]
            verified[str(p)]=h
    graphpath=translate_with_einsum(dest/'dsl',dest/'einsum')
    graph=yaml.safe_load(graphpath.read_text())
    analysis=EinsumGraphAnalyzer().analyze_graph(str(graphpath),str(dest/'analysis'),precision='fp32',copy_graph=False)
    rows=[(k,v) for k,v in graph['layers'].items() if k.startswith('k#')]
    record={'source':str(src),'verified_source_hashes':verified,'metadata':metadata,'expected_runtime_counts':oldrecord['runtime_statement_counts'],'graph':graph,'analysis':analysis}
    (dest/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    assert len(rows)==len(record['expected_runtime_counts'])
    for (k,v),n in zip(rows,record['expected_runtime_counts']):
        assert v.get('loop_iterations_complete') is False or v.get('loop_iterations')==n,record
