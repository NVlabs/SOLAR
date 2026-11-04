from solar.einsum_lib import Tensor, Dim, kernel, clear_dims
@kernel
def k(X:Tensor("T"), Z:Tensor("U")) -> Tensor("O"):
    Y=Tensor(O=1)
    for t in Dim("T"):
        Y[0]+=X[0]
    return Y

def run_model(inputs):
    clear_dims()
    x=Tensor.from_numpy(inputs["x"],"T")
    z=Tensor.from_numpy(inputs["z"],"T")
    return {"a":k(x,z).data,"b":k(z,x).data}
