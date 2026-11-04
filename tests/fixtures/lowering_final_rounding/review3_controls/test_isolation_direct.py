import json,pathlib,os
import numpy as np,pytest
from solar.einsum_lib.verify import execute_translated,_run_isolated
R=pathlib.Path(__file__).parent
def _sigkill():
 import signal
 os.kill(os.getpid(),signal.SIGKILL)
def test_sigkill_maps_to_memoryerror():
 with pytest.raises(MemoryError,match='SIGKILLed'):_run_isolated(_sigkill,30,10**9)
@pytest.mark.parametrize('fast',[False,True])
def test_seeded_numpy_state_is_preserved_by_isolation(fast):
 code='import numpy as np\nR = np.random.uniform(1,2)\ndef run_model(inputs):\n    return {"output":np.array([R])}\n'
 record={}
 original=np.random.get_state()
 try:
  for label,timeout in [('inprocess',None),('isolated',30)]:
   np.random.seed(12345)
   record[label]=execute_translated(code,{},all_inputs={},fast=fast,timeout=timeout)[0]['output'].tolist()
 finally:np.random.set_state(original)
 p=R/'runs/isolation';p.mkdir(parents=True,exist_ok=True)
 (p/('rng-'+str(fast)+'.json')).write_text(json.dumps(record,indent=2)+'\n')
 assert record['inprocess']==record['isolated'],record

