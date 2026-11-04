from solar.einsum_lib import Tensor, Dim, kernel
@kernel
def gather(X: Tensor("N,H", dtype="fp32"), Idx: Tensor("I", dtype="int64")) -> Tensor("I,H"):
    Y = Tensor(I=Idx.shape[0], H=X.shape[1], dtype="fp32")
    for i in Dim("I"):
        for h in Dim("H"):
            Y[i,h] = X[i,h] * 2.0
    return Y
