import pytest
import test_round5_edges as edge
BASE='    Y=Tensor(O=1)\n    for t in CALL:\n        Y[0]+=X[0]\n    return Y\n'
def code(call):return edge.prog(BASE.replace('CALL',call))
CASES={
 'r7_keyword_order':{'code':code('Dim("T",size=3,parallel=True)'),'want':'count'},
 'r7_all_keywords':{'code':code('Dim(name="T",parallel=False,size=3)'),'want':'count'},
 'r7_positional_parallel_scalar':{'code':code('Dim("T",True,n)'),'want':'count'},
 'r7_empty_kwargs':{'code':code('Dim("T",**{})'),'want':'refuse'},
 'r7_star_name':{'code':code('Dim(*("T",False,3))'),'want':'refuse'},
 'r7_star_suffix':{'code':code('Dim("T",*(False,3))'),'want':'refuse'},
 'r7_standalone_star_name':{'code':edge.prog('    Y=Tensor(O=1)\n    D=Dim(*("U",False,3))\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'),'want':'refuse'},
 'r7_size_nested_call':{'code':edge.prog(BASE.replace('CALL','Dim("T",size=choose(Dim("U",size=3)))'),helper='def choose(x):\n    return 3\n'),'want':'refuse'},
 'r7_header_parallel_nested_same':{'code':code('Dim("T",parallel=Dim("T",size=3))'),'want':'count_or_refuse'},
 'r7_header_parallel_nested_other':{'code':edge.prog(BASE.replace('CALL','Dim("T",parallel=Dim("U",size=3))').replace('    return Y','    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y')),'want':'count_or_refuse'},
 'r7_standalone_nested_overwrite':{'code':edge.prog('    Y=Tensor(O=1)\n    D=Dim("U",parallel=Dim("U",size=4),size=3)\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'),'want':'count_or_refuse'},
 'r7_local_dim_then_size':{'code':edge.prog(BASE.replace('    Y=','    q=Dim("U",size=3)\n    n=choose(q)\n    Y=').replace('CALL','Dim("T",size=n)'),helper='def choose(x):\n    return 3\n'),'want':'refuse'},
}
edge.CASES.update({k:{a:b for a,b in v.items() if a!='want'} for k,v in CASES.items()})
@pytest.mark.parametrize('name',CASES)
def test_size_or_explicit_refusal(name):
 r=edge.observe(name);assert 'conversion_error' not in r,r
 l=r['graph']['layers']['k#0'];want=CASES[name]['want']
 if l.get('loop_iterations_complete') is False:
  assert want!='count',r
  assert 'loop_iterations' not in l and r['analysis']['layers']['k#0']['status']=='incomplete',r
 else:
  assert want!='refuse',r
  assert l.get('loop_iterations')==r['runtime_count'],r
