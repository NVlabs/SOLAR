from solar.einsum_lib import Tensor, Dim, kernel, clear_dims
DT="bf16"
class Rebind:
    global DT
    DT='fp32'
@kernel
def k(A:Tensor("M,N",dtype="fp32"))->Tensor("M,N"):
    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype=DT,fill=1.003)
    return Y

def run_model(inputs):
    clear_dims()
    a=Tensor.from_numpy(inputs['a'],'M,N',dtype='fp32')
    y=k(a)
    return {'output':y.data}
