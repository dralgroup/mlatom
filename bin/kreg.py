import numpy as np 
from sklearn.linear_model import Ridge
try:
    from . import data
except:
    import data

class kreg():
    def __init__(self):
        self.model=None
        pass 

    def train(self,X,Y,alpha):
        self.model = Ridge(alpha=alpha)
        self.model.fit(X,Y)


    def predict(self,X):
        self.model.predict(X)
        

def gaussian_kernel(xi,xj,sigma):
    return np.exp(-np.sum((xi-xj)**2)/(2*sigma**2))

def RE_descriptor(Req,coord):
    Natoms = len(coord)
    descriptor = []
    icount = 0
    for iatom in range(Natoms):
        for jatom in range(iatom+1,Natoms):
            descriptor.append(Req[icount]/distance(coord[iatom],coord[jatom]))
    return np.array(descriptor)

def distance(xia,xib):
    return np.sqrt(np.sum((xia-xib)**2))

