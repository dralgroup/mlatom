using LinearAlgebra

function solveSysLinEqs(AA,bb)
    # println(typeof(AA))
    # println(size(AA))
    # print(AA[begin:5,begin:5])
    # print(bb[begin:5])
    sAA = cholesky(Symmetric(AA))
    xx = sAA\bb 
    return xx 
end