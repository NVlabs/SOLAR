from pathlib import Path as P
import json,time,signal,sys,os,traceback
import torch
from solar.einsum.pytorch_to_einsum import _returned_leaf_region,_returned_group_footprint
OUT=P(os.environ['REVIEW_RUN_OUT']);N=(1<<20)+1
base=torch.empty(2*N,dtype=torch.float32)
def region(t,path):
    d={'shape':list(t.shape),'stride':list(t.stride()),'storage_offset':t.storage_offset(),'element_size':t.element_size(),'storage_nbytes':t.untyped_storage().nbytes()}
    r=_returned_leaf_region(d,{});r['path']=path;return d,r
bd,b=region(base[::2],'bound');rows=[];failed=False
def alarm(sig,frame):raise TimeoutError('reviewer 30-second computation-only timer')
signal.signal(signal.SIGALRM,alarm)
for k in [256,512,1024,32768]:
    ud,u=region(base[1:2*k:2],'unbound')
    record={'k':k,'bound_identity':bd,'unbound_identity':ud,'expected_unbound_bytes':4*k,'bound_method':b['method'],'unbound_method':u['method'],'unbound_intervals':len(u['intervals'] or []),'same_real_storage':base[::2].untyped_storage()._cdata==base[1:2*k:2].untyped_storage()._cdata}
    t=time.monotonic();signal.setitimer(signal.ITIMER_REAL,30)
    try:
        fp=_returned_group_footprint([u],[b],base.untyped_storage().nbytes())
        record.update(state='COMPLETED',result=fp)
        assert fp['union_bytes']==4*k and fp['covered_bytes']==0 and fp['period']==8
    except TimeoutError:
        record.update(state='TIMEOUT',traceback=traceback.format_exc());failed=True
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    record['seconds']=time.monotonic()-t;rows.append(record)
    (OUT/'RESULT.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(record),flush=True)
    if failed:break
raise SystemExit(1 if failed else 0)
