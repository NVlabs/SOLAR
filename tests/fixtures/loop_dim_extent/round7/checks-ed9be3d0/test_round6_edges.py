import pytest
import test_round5_edges as edge

BASE='    Y=Tensor(O=1)\n    for t in Dim("T",size=SIZE):\n        Y[0]+=X[0]\n    return Y\n'
def expr(value,**kwargs):return edge.prog(BASE.replace('SIZE',value),**kwargs)
SPECS={
 'r6_ternary':{'code':expr('n if flag else m',sig=edge.K.replace('n:int=3','n:int=3,m:int=4,flag:bool=True')),'want':'refuse'},
 'r6_max':{'code':expr('max(2,n)'),'want':'refuse'},
 'r6_two_tensor_shapes':{'code':expr('X.shape[0]+Z.shape[0]'),'want':'count'},
 'r6_two_tensor_shapes_legacy':{'code':expr('X.shape[0]+Z.shape[0]'),'strip':['input_order'],'want':'refuse'},
 'r6_two_local_chain':{'code':edge.prog('    a=X.shape[0]\n    b=a+1\n    Y=Tensor(O=1)\n    for t in Dim("T",size=b):\n        Y[0]+=X[0]\n    return Y\n'),'want':'count'},
 'r6_two_local_chain_legacy':{'code':edge.prog('    a=X.shape[0]\n    b=a+1\n    Y=Tensor(O=1)\n    for t in Dim("T",size=b):\n        Y[0]+=X[0]\n    return Y\n'),'strip':['input_order'],'want':'refuse'},
 'r6_posonly_tensor':{'code':edge.SHAPE.replace('Z:Tensor("B"))','Z:Tensor("B"),/)').replace('nb=X.shape[1]','nb=X.shape[1]//1'),'shape':True,'want':'count'},
 'r6_keyword_reordered_tensor':{'code':edge.SHAPE.replace('k(x,z).data','k(Z=z,X=x).data').replace('nb=X.shape[1]','nb=X.shape[1]//1'),'shape':True,'want':'count'},
 'r6_coincidental_registry_equality':{'code':edge.SHAPE.replace('nb=X.shape[1]','nb=X.shape[1]//3'),'shape':True,'strip':['input_order'],'want':'refuse'},
 'r6_mixed_headers':{'code':edge.SHAPE.replace('nb=X.shape[1]','nb=X.shape[1]//1').replace('    for t in Dim("T"):', '    for b in Dim("B",size=2):\n        Y[0]+=X[0,0]\n    for t in Dim("T"):'),'shape':True,'strip':['input_order'],'want':'refuse'},
 'r6_len_shape':{'code':expr('len(X.shape)'),'want':'refuse'},
 'r6_standalone_opaque':{'code':edge.prog('    Y=Tensor(O=1)\n    D=Dim("U",size=max(2,n))\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'),'want':'refuse'},
 'r6_positional_header_size':{'code':edge.prog(BASE.replace('Dim("T",size=SIZE)','Dim("T",False,3)')),'want':'count_or_refuse'},
 'r6_keyword_dict_header_size':{'code':edge.prog(BASE.replace('Dim("T",size=SIZE)','Dim("T",**{"size":3})')),'want':'count_or_refuse'},
 'r6_positional_standalone_size':{'code':edge.prog('    Y=Tensor(O=1)\n    D=Dim("U",False,3)\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'),'want':'count_or_refuse'},
 'r6_keyword_dict_standalone_size':{'code':edge.prog('    Y=Tensor(O=1)\n    D=Dim("U",**{"size":3})\n    for u in Dim("U"):\n        Y[0]+=X[0]\n    return Y\n'),'want':'count_or_refuse'},
 'r6_keyword_name':{'code':edge.prog(BASE.replace('Dim("T",size=SIZE)','Dim(name="T",size=3)')),'want':'count_or_refuse'},
}
edge.CASES.update({k:{a:b for a,b in v.items() if a!='want'} for k,v in SPECS.items()})
@pytest.mark.parametrize('name',SPECS)
def test_explicit_size_contract(name):
 r=edge.observe(name);assert 'conversion_error' not in r,r
 layer=r['graph']['layers']['k#0'];want=SPECS[name]['want'];incomplete=layer.get('loop_iterations_complete') is False
 if incomplete:
  assert want!='count',r
  assert 'loop_iterations' not in layer and r['analysis']['layers']['k#0']['status']=='incomplete',r
  assert any(k.startswith('loop_') and ('unresolved' in k or 'mismatch' in k or 'unknown' in k or 'omitted' in k) for k in layer),r
 else:
  assert want!='refuse',r
  assert layer.get('loop_iterations')==r['runtime_count'],r
