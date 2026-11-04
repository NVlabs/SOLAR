"""Private independent round9 literal boundary, identity, and seeded-state controls."""
import ast,json,os,pathlib,random,hashlib,pickle,inspect
import numpy as np
import pytest
import solar.einsum_lib.numba_accel as na
from solar.einsum_lib.verify import execute_translated
from solar.einsum_lib import clear_dims,stop_trace
R=pathlib.Path(__file__).parent
OUT=R/'runs/review4';OUT.mkdir(exist_ok=True)
H='from solar.einsum_lib import Tensor, kernel, clear_dims\nimport numpy as np\n'
K='@kernel\ndef k(A:Tensor("M,N",dtype="fp32"))->Tensor("M,N"):\n    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype="bf16",fill=1.003)\n    return Y\n'
RUN='\ndef run_model(inputs):\n    clear_dims()\n    a=Tensor.from_numpy(inputs["a"],"M,N",dtype="fp32")\n    return {"output":k(a).data}\n'
WRAP='def wrapper(f):\n    return real_kernel(f)\n'
PREFIXES={'function_replacement':'real_kernel=kernel\ndef kernel(f):\n    return real_kernel(f)\n', 'module_rebinding':'real_kernel=kernel\n'+WRAP+'kernel=wrapper\n', 'forwarding_attributes':'real_kernel=kernel\n'+WRAP+'wrapper.__dict__.update(kernel.__dict__)\nkernel=wrapper\n', 'reflective_replacement':'real_kernel=kernel\n'+WRAP+'exec("kernel=wrapper")\n'}
def record(name,v):
 (OUT/(name+'.json')).write_text(json.dumps(v,indent=2)+'\n')
@pytest.mark.parametrize('name',PREFIXES)
@pytest.mark.parametrize('fp64',[False,True])
def test_identity_decline_literal(name,fp64):
 code=H+PREFIXES[name]+K+RUN; reasons=[];fallback=[];inputs={'a':np.ones((1,2))}
 clear_dims();stop_trace();low=na.numba_lower(code,use_fp64=fp64,on_decline=reasons.append)
 slow=execute_translated(code,{},all_inputs=inputs,fast=False)[0]['output']; direct=na.execute_fast(code,inputs,use_fp64=fp64,on_fallback=fallback.append);out=execute_translated(code,{},all_inputs=inputs,fast=True,use_fp64=fp64)[0]['output']
 v={'code':code,'lowered':low is not None,'reasons':reasons,'fallback':fallback,'slow':slow.tolist(),'public':out.tolist(),'direct_is_none':direct is None};record('identity-'+name+'-'+str(fp64),v)
 assert direct is None and any('the @kernel decorator' in x for x in reasons+fallback),v
 np.testing.assert_array_equal(out,slow)

def digest_state():return hashlib.sha256(pickle.dumps((np.random.get_state(),random.getstate()),protocol=4)).hexdigest()
@pytest.mark.parametrize('fast,kind',[(False,'numpy'),(True,'numpy'),(False,'python')])
def test_loop_rng_output_count_and_parent_state(fast,kind):
 n=7
 if kind=='numpy':code='import numpy as np\nV=[]\nfor i in range(7):\n    V.append(np.random.uniform(1,2))\ndef run_model(inputs):\n    return {"output":np.array(V+[np.random.uniform(1,2)])}\n'
 else:code='import numpy as np\nimport random\nV=[]\nfor i in range(7):\n    V.append(random.random())\ndef run_model(inputs):\n    return {"output":np.array(V+[random.random()])}\n'
 orig=(np.random.get_state(),random.getstate()); rec={'code':code,'draws':n+1}
 try:
  for label,timeout in [('inprocess',None),('isolated',30)]:
   np.random.seed(12345);random.seed(777);before=digest_state()
   out=execute_translated(code,{},all_inputs={},fast=fast,timeout=timeout)[0]['output']
   after=digest_state();rec[label]={'out':out.tolist(),'before':before,'after':after,'next_numpy':np.random.uniform(1,2),'next_python':random.random()}
  np.random.seed(12345);random.seed(777)
  ref=[np.random.uniform(1,2) if kind=='numpy' else random.random() for _ in range(n+1)]
  rec['expected']=ref;rec['caller_states_equal_between_modes']=rec['inprocess']['after']==rec['isolated']['after'];record('loop-rng-'+os.getenv('VARIANT','head')+'-'+kind+'-'+str(fast),rec)
  assert rec['inprocess']['out']==rec['isolated']['out']==ref,rec
  assert rec['isolated']['before']==rec['isolated']['after'],rec
  assert rec['inprocess']['before']!=rec['inprocess']['after'],rec
 finally:np.random.set_state(orig[0]);random.setstate(orig[1])

@pytest.mark.parametrize('fp64',[False,True])
def test_nonreturned_dtype_expression_declines(fp64):
 code=H+'COUNT=0\ndef choose_dtype():\n    global COUNT\n    COUNT+=1\n    return "bf16"\n'+K.replace('    Y=Tensor','    T=Tensor(M=A.shape[0],N=A.shape[1],dtype=choose_dtype(),fill=0)\n    Y=Tensor').replace('fill=1.003','fill=COUNT+1.003')+RUN
 reasons=[];fallback=[];inputs={'a':np.ones((1,2))};clear_dims();stop_trace()
 low=na.numba_lower(code,use_fp64=fp64,**({'on_decline':reasons.append} if 'on_decline' in inspect.signature(na.numba_lower).parameters else {}));slow=execute_translated(code,{},all_inputs=inputs,fast=False)[0]['output'];raw=na.execute_fast(code,inputs,use_fp64=fp64,on_fallback=fallback.append);out=execute_translated(code,{},all_inputs=inputs,fast=True,use_fp64=fp64)[0]['output']
 rec={'code':code,'lowered':low is not None,'reasons':reasons,'fallback':fallback,'slow':slow.tolist(),'public':out.tolist()};record('nonreturned-dtype-'+os.getenv('VARIANT','head')+'-'+str(fp64),rec)
 np.testing.assert_array_equal(slow,out,err_msg=str(rec))
 assert low is None and any('kernel_output_dtype_unestablished' in s for s in reasons+fallback),rec
