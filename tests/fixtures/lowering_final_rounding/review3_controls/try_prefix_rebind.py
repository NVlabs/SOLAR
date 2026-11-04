from solar.einsum_lib import Tensor, Dim, kernel, clear_dims
DT="bf16"
@kernel
def k(A:Tensor("M,N",dtype="fp32"))->Tensor("M,N"):
    Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='fp32',fill=1.003)
    try:
        Y=Tensor(M=A.shape[0],N=A.shape[1],dtype='bf16',fill=1.003)
        raise ValueError('trigger')
    except Exception:
        pass
    return Y

def run_model(inputs):
    clear_dims()
    a=Tensor.from_numpy(inputs['a'],'M,N',dtype='fp32')
    y=k(a)
    return {'output':y.data}
