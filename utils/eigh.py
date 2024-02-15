'''
PyTorch has its own implementation of the backward function for
the symmetric eigensolver, however it is slow and not necessarily stable.
We can reimplement it here to use a scipy forward function... (not yet done)
'''

import numpy as np 
import torch
import time
import scipy.linalg

class EigenSolver(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        w, v = torch.symeig(A, eigenvectors=True)

        self.save_for_backward(w, v)
        return w, v

    @staticmethod
    def backward(self, dw, dv):
        w, v = self.saved_tensors
        dtype, device = w.dtype, w.device
        N = v.shape[0]

        F = w - w[:,None]
        # in case of degenerate eigenvalues, replace the following two lines with a safe inverse
        F.diagonal().fill_(np.inf);
        F = 1./F  

        vt = v.t()
        vdv = vt@dv

        return v@(torch.diag(dw) + F*(vdv-vdv.t())/2) @vt

def test_eigs():
    M = 2
    torch.manual_seed(42)
    A = torch.rand(M, M, dtype=torch.float64)
    A = torch.nn.Parameter(A+A.t())
    assert(torch.autograd.gradcheck(DominantEigensolver.apply, A, eps=1e-6, atol=1e-4))
    print("Test Pass!")

if __name__=='__main__':
    re = torch.rand([1500,1500])
    im = torch.rand([1500,1500])
    mat = torch.complex(re, im)
    matn = mat.numpy()
    time0 = time.time()
    w, v = torch.linalg.eigh(mat)
    time1 = time.time()
    vals, vecs = scipy.linalg.eigh(matn, subset_by_index=[0,32], driver='evr')
    time2 = time.time()

    print(f"time torch eigh: {time1-time0}")
    print(f"time scipy eigh subset: {time2-time1}")

