# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import copy,json,numbers
from decimal import Decimal
import numpy as np,pytest,yaml
from solar.analysis.graph_analyzer import dsl_loop_count_evidence,EinsumGraphAnalyzer
from solar.perf import EinsumGraphPerfModel
T=Path(__file__).parent
class IntSubclass(int): pass
VALUES=[('fractional_float',7.9),('infinity',float('inf')),('numeric_string','24'),('integral_float',24.0),('nan',float('nan')),('negative',-1),('bool',True),('none',None),('zero',0),('complex',24+0j),('list',[24]),('negative_inf',float('-inf')),('numpy_float',np.float64(24)),('numpy_bool',np.bool_(True)),('decimal',Decimal('24')),('int',24),('numpy_int64',np.int64(24)),('numpy_int32',np.int32(24)),('numpy_uint64',np.uint64(2**64-1)),('huge_int',2**80+37),('int_subclass',IntSubclass(24))]
@pytest.mark.parametrize('name,value',VALUES,ids=[x[0] for x in VALUES])
def test_gate_contract(name,value,tmp_path):
 ev=dsl_loop_count_evidence({'loop_iterations':value,'loop_iterations_complete':True})
 trusted=isinstance(value,numbers.Integral) and not isinstance(value,bool) and value>0
 (tmp_path/'observation.json').write_text(json.dumps({'case':name,'repr':repr(value),'type':type(value).__name__,'integral':isinstance(value,numbers.Integral),'expected_trusted':bool(trusted),'evidence':ev._asdict()},indent=2)+'\n')
 assert ev.present
 assert ev.trusted is bool(trusted)
 assert ev.count==(int(value) if trusted else 0)
 assert ev.reasons==([] if trusted else ['loop_count_malformed:'+repr(value)])
 assert ev._fields==('present','count','trusted','reasons','equation_faithful')

SERIALIZABLE=[p for p in VALUES if p[0] in {'fractional_float','infinity','numeric_string','integral_float','nan','negative','bool','none','zero','list','negative_inf','int','huge_int'}]
@pytest.mark.parametrize('name,value',SERIALIZABLE,ids=[x[0] for x in SERIALIZABLE])
def test_layer_total_perf_contract(name,value,tmp_path):
 graph=yaml.safe_load((T/'round4-dense-graph.yaml').read_text())  # shipped beside this test
 node=graph['layers']['linear#0'];node['loop_iterations']=value;node['loop_iterations_complete']=True;node.pop('loop_iterations_omitted',None)
 p=tmp_path/'graph.yaml';p.write_text(yaml.safe_dump(graph))
 a=EinsumGraphAnalyzer().analyze_graph(p,tmp_path/'analysis')
 perf=EinsumGraphPerfModel().predict(tmp_path/'analysis/analysis.yaml',tmp_path/'perf',arch_config='H100_PCIe')  # the checkout's configs/arch/H100_PCIe.yaml
 l=a['layers']['linear#0'];trusted=type(value) is int and value>0
 obs={'name':name,'raw_repr':repr(value),'gate':dsl_loop_count_evidence(node)._asdict(),'layer':{k:l.get(k) for k in ('macs','other_ops','status','incomplete_reason','loop_iterations_omitted')},'total_status':a['total'].get('status'),'perf_status':perf['workload'].get('status'),'perf_macs':perf['workload'].get('total_macs')}
 (tmp_path/'observation.json').write_text(json.dumps(obs,indent=2)+'\n')
 assert l['macs']==(value if trusted else 24)
 if not trusted:
  assert [l.get('status'),a['total'].get('status'),perf['workload'].get('status')]==['incomplete']*3
  assert l['incomplete_reason']=='loop_metadata_incomplete'
  assert l['loop_iterations_omitted']==['loop_count_malformed:'+repr(value)]
 else:assert not any(x.get('status')=='incomplete' for x in (l,a['total'],perf['workload']))

def test_genuinely_absent_remains_absent():
 ev=dsl_loop_count_evidence({})
 assert ev.present is False and ev.count==0 and ev.trusted is False and ev.reasons==[]

def test_omissions_and_completeness_still_win():
 for layer in [{'loop_iterations':24},{'loop_iterations':24,'loop_iterations_complete':False},{'loop_iterations':24,'loop_iterations_complete':True,'loop_iterations_omitted':['range']}]:
  ev=dsl_loop_count_evidence(layer)
  assert ev.present and not ev.trusted and ev.count==0 and ev.reasons
