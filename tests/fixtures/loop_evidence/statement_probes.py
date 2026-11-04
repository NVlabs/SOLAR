import pathlib,json,sys,copy,traceback
import numpy as np,yaml
from solar.einsum_lib.verify import execute_translated
from solar.einsum_lib import build_metadata,get_trace
from solar.einsum_lib.metadata import write_metadata_yaml
from solar.einsum_dsl.cli_einsum import translate_with_einsum
from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
R=pathlib.Path(__file__).parent
label=sys.argv[1]
header='from solar.einsum_lib import Tensor,Dim,kernel,clear_dims\n'
prefix='''@kernel
def probe(X:Tensor("T,H"))->Tensor("T,H"):
 Y=Tensor(T=X.shape[0],H=X.shape[1])
 for t in Dim("T",parallel=True):
  for h in Dim("H"):
'''
suffix=''' return Y
def run_model(inputs):
 clear_dims()
 x=Tensor.from_numpy(inputs["x"],"T,H")
 return {"out":probe(x).data}
'''
cases={
"range7":(header+prefix+'   for j in range(7):\n    Y[t,h] += X[t,h]\n'+suffix,56,7),
"outer_statements":(header+prefix+'   s=X[t,h]+1\n   for j in Dim("J",size=3):\n    s=s+2\n   Y[t,h]=s\n'+suffix,40,8),
"two_statements":(header+prefix+'   Y[t,h]=X[t,h]*2\n   Y[t,h]=Y[t,h]+1\n'+suffix,16,3),
}
records={}
def analyze(graph,name):
 d=R/"probes"/label/name;d.mkdir(parents=True,exist_ok=True)
 path=d/"graph.yaml";path.write_text(yaml.safe_dump(graph,sort_keys=False))
 return EinsumGraphAnalyzer().analyze_graph(str(path),str(d/"analysis"))["layers"]["probe#0"]
for name,(src,expected,value) in cases.items():
 inp={"x":np.ones((2,4))}
 outputs,_=execute_translated(src,{k:v.shape for k,v in inp.items()},all_inputs=inp,trace=True,shape_only=False,fast=False)
 assert np.all(outputs["out"]==value)
 md=build_metadata(get_trace())
 d=R/"probes"/label/name/"dsl";d.mkdir(parents=True,exist_ok=True)
 (d/"probe_einsum_lib.py").write_text(src);write_metadata_yaml(md,str(d/"probe_einsum_lib.yaml"))
 gpath=translate_with_einsum(d,d.parent/"einsum");g=yaml.safe_load(pathlib.Path(gpath).read_text())
 a=analyze(g,name);node=g["layers"]["probe#0"]
 records[name]={"numerical_values_checked":True,"expected_executed_assignments":expected,"graph_fields":{k:node.get(k) for k in ("loop_iterations","loop_iterations_complete","loop_iterations_omitted","equation_faithful","is_real_einsum")},"analysis":{k:a.get(k) for k in ("other_ops","cost_basis","status","incomplete_reason","compute_density")},"exact_count_pass":a["other_ops"]==expected}
 if name=="two_statements":
  bad=copy.deepcopy(g);bad["layers"]["probe#0"].update(loop_iterations=1,loop_iterations_complete=False,loop_iterations_omitted=["range(7) unrepresented"],equation_faithful=False)
  b=analyze(bad,"explicit_incomplete")
  records["explicit_incomplete"]={"supplied_loop_count":1,"complete":False,"omitted":["range(7) unrepresented"],"analysis":{k:b.get(k) for k in ("other_ops","cost_basis","status","incomplete_reason")},"gate_rejects_incomplete_count":b["other_ops"]!=1}
masksrc=header+'''@kernel
def probe(X:Tensor("T,H"),M:Tensor("T,H"))->Tensor("T,H"):
 Y=Tensor(T=X.shape[0],H=X.shape[1])
 for t in Dim("T",parallel=True):
  for h in Dim("H"):
   Y[t,h]=X[t,h]*M[t,h]
 return Y
def run_model(inputs):
 clear_dims()
 x=Tensor.from_numpy(inputs["x"],"T,H")
 m=Tensor.from_numpy(inputs["m"],"T,H",sparsity=.25)
 return {"out":probe(x,m).data}
'''
mask=np.zeros((2,4));mask[:,0]=1
inp={"x":np.ones((2,4)),"m":mask}
execute_translated(masksrc,{k:v.shape for k,v in inp.items()},all_inputs=inp,trace=True,shape_only=False,fast=False)
md=build_metadata(get_trace());d=R/"probes"/label/"mask"/"dsl";d.mkdir(parents=True,exist_ok=True)
(d/"probe_einsum_lib.py").write_text(masksrc);write_metadata_yaml(md,str(d/"probe_einsum_lib.yaml"))
gp=translate_with_einsum(d,d.parent/"einsum");g=yaml.safe_load(pathlib.Path(gp).read_text());a=analyze(g,"mask")
records["mask"]={"expected_dense":8,"expected_discounted":2,"graph_loop_iterations":g["layers"]["probe#0"].get("loop_iterations"),"analysis":{k:a.get(k) for k in ("other_ops","cost_basis","compute_density","status","incomplete_reason")},"discount_pass":a["other_ops"]==2}
(R/"evidence"/("probes-"+label+".json")).write_text(json.dumps(records,indent=2)+"\n")
print(json.dumps(records,indent=2),flush=True)

