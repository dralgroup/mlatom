import deepmd.DeepPot as DP
import numpy as np
import sys

path, model, enest, gradest = sys.argv[1:5]
dp = DP(model)
coord = np.load(path)
cell = np.repeat(np.diag(64 * np.ones(3)).reshape([1, -1]),len(coord), axis=0)
with open('type.raw','r') as ff:
    atype=[int(i) for i in ff.readline().split()]

e, f, v = dp.eval(coord, cell, atype)

line="%d\n\n" % int(coord.shape[1]/3)
with open(enest,'wb') as ff:
    np.savetxt(ff, e, fmt='%20.12f', delimiter=" ")
with open(gradest,'wb') as ff:
    for force in f:
        ff.write(line.encode('utf-8'))
        np.savetxt(ff, -1*force, fmt='%20.12f', delimiter=" ")
