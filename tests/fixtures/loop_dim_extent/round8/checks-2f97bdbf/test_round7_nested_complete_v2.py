import pytest
import test_round5_edges as edge
from test_round7_edges import CASES
SPECS={}
for old in ['r7_header_parallel_nested_same','r7_header_parallel_nested_other','r7_standalone_nested_overwrite']:
 name=old.replace('r7_','r7_complete_');s=CASES[old]['code'].replace('Y[0]+=X[0]','Y[0]+=Z[0]');SPECS[name]=s
 SPECS[name+'_bool']=s.replace('parallel=Dim("T",size=3)','parallel=bool(Dim("T",size=3))').replace('parallel=Dim("U",size=3)','parallel=bool(Dim("U",size=3))').replace('parallel=Dim("U",size=4)','parallel=bool(Dim("U",size=4))')
SPECS['r7_complete_literal_parallel_control']=edge.prog('    Y=Tensor(O=1)\n    for t in Dim("T",parallel=True):\n        Y[0]+=Z[0]\n    return Y\n')
SPECS={k+'_v2':s for k,s in SPECS.items()}
edge.CASES.update({k:{'code':s} for k,s in SPECS.items()})
@pytest.mark.parametrize('name',SPECS)
def test_nested_registration_count_or_refusal(name):
 r=edge.observe(name);assert 'conversion_error' not in r,r
 l=r['graph']['layers']['k#0'];a=r['analysis']['layers']['k#0']
 if a.get('status')=='incomplete':return
 assert l['loop_iterations_complete'] is True and l['loop_iterations']==r['runtime_count'],r
 assert a['other_ops']==r['runtime_count'],r
