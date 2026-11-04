import pytest
import test_round5_edges as edge
for name,legacy in [('header_binop_fresh',False),('header_binop_legacy',True)]:
 edge.CASES[name]={'code':edge.SHAPE.replace('nb=X.shape[1]','nb=X.shape[1]//1'),'shape':True,'strip':['input_order'] if legacy else []}
@pytest.mark.parametrize('name',['header_binop_fresh','header_binop_legacy'])
def test_header_binop_count_or_refusal(name):
 edge.test_count_matches_runtime_or_typed_refusal(name)
