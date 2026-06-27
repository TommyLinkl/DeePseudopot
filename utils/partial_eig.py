"""
Partial Hermitian eigensolver with a custom autograd backward.

Motivation: the band-structure loss needs only the lowest ~nBands eigenvalues of
the (2*nbv)-dimensional Hamiltonian, but torch.linalg.eigvalsh computes ALL of
them. When nBands << 2*nbv (here typically ~1%), computing only the requested
subset with a direct LAPACK partial driver (scipy.linalg.eigh with
subset_by_index, i.e. ?heevr / ?heevx) is far cheaper in both time and memory.

Autograd: the solver itself need not be differentiable. For a Hermitian H with
eigenpair (lambda_i, v_i), the eigenvalue derivative is Hellmann-Feynman,

        d lambda_i = v_i^H (dH) v_i,

so for an upstream cotangent g_i = dL/d(lambda_i) the reverse-mode cotangent on H
is the low-rank Hermitian matrix

        H_bar = sum_i g_i v_i v_i^H  =  V diag(g) V^H.

This needs ONLY the eigenvectors the solver already returns, has NO 1/(l_i - l_j)
terms, and is therefore degeneracy-safe (unlike full-eigh eigenVECTOR autograd).
It is invariant to the (arbitrary) phase scipy assigns to each v_i, so the phase
mismatch between scipy and torch eigenvectors is irrelevant.

The gradient convention is matched to torch.linalg.eigvalsh EXACTLY (verified
numerically; run `python -m utils.partial_eig` for the self-check): for a real
loss L(lambda), autograd.grad through partial_eigvalsh(H, k) equals
autograd.grad through (torch.linalg.eigvalsh(H)[:k]) elementwise on H.

Only the eigenvalues are differentiated here; the coupling code path that needs
eigenVECTORS still uses torch.linalg.eigh (with its degenerate-subspace handling)
and must NOT use this solver.
"""
import numpy as np
import torch
from scipy.linalg import eigh as _scipy_eigh


class _PartialEigvalsh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, k, lower, driver):
        n = H.shape[-1]
        k = int(k)
        if k < 1 or k > n:
            raise ValueError(f"partial_eigvalsh: k={k} out of range [1, {n}]")
        # Direct LAPACK subset driver (heevr/heevx for complex, syevr/syevx for
        # real). Reads the same triangle (lower by default) as torch.linalg.eigvalsh
        # so the two agree even if H carries tiny non-Hermitian rounding noise.
        Hnp = H.detach().contiguous().cpu().numpy()
        w, V = _scipy_eigh(Hnp, lower=lower, subset_by_index=[0, k - 1],
                           eigvals_only=False, driver=driver)
        w_t = torch.as_tensor(w, dtype=torch.float64, device=H.device)
        V_t = torch.as_tensor(np.ascontiguousarray(V), dtype=H.dtype, device=H.device)
        ctx.save_for_backward(V_t)
        return w_t

    @staticmethod
    def backward(ctx, grad_w):
        (V,) = ctx.saved_tensors                       # (n, k) eigenvectors as columns
        # H_bar = V diag(grad_w) V^H = sum_j grad_w[j] v_j v_j^H  (Hermitian, rank<=k).
        gV = V * grad_w.to(V.dtype).unsqueeze(0)        # scale column j by grad_w[j]
        grad_H = gV @ V.conj().transpose(-2, -1)
        return grad_H, None, None, None


def partial_eigvalsh(H, k, lower=True, driver="evr"):
    """Lowest-k eigenvalues (ascending) of Hermitian/symmetric H, differentiable in H.

    H      : (n, n) complex or real Hermitian torch tensor.
    k      : number of lowest eigenvalues to return.
    lower  : read the lower triangle of H (matches torch.linalg.eigvalsh UPLO='L').
    driver : LAPACK subset driver -- 'evr' (heevr/syevr, robust, default) or
             'evx' (heevx/syevx, often faster for very small subsets).
    Returns a length-k real (float64) tensor.
    """
    return _PartialEigvalsh.apply(H, k, lower, driver)


def _self_check():
    """Numerically verify forward values and backward gradients against
    torch.linalg.eigvalsh (the convention we must match). Returns nothing; raises
    AssertionError on mismatch. Run with `python -m utils.partial_eig`."""
    torch.manual_seed(0)
    for dtype in (torch.complex128, torch.float64):
        for n, k in ((40, 1), (40, 4), (64, 7), (16, 16)):
            A = torch.randn(n, n, dtype=dtype)
            Hbase = A + A.conj().transpose(-2, -1)        # Hermitian
            g = torch.randn(k, dtype=torch.float64)        # arbitrary upstream cotangent

            # reference: full eigvalsh, slice to k
            Hr = Hbase.clone().detach().requires_grad_(True)
            lam_full = torch.linalg.eigvalsh(Hr)[:k]
            (g * lam_full).sum().backward()
            grad_ref = Hr.grad

            # partial solver
            Hp = Hbase.clone().detach().requires_grad_(True)
            lam_part = partial_eigvalsh(Hp, k)
            (g * lam_part).sum().backward()
            grad_part = Hp.grad

            val_err = (lam_full.detach() - lam_part.detach()).abs().max().item()
            grad_err = (grad_ref - grad_part).abs().max().item()
            print(f"dtype={str(dtype):>16}  n={n:>3} k={k:>3}  "
                  f"max|dlam|={val_err:.2e}  max|dgrad|={grad_err:.2e}")
            assert val_err < 1e-9, f"eigenvalue mismatch {val_err}"
            assert grad_err < 1e-9, f"gradient mismatch {grad_err}"
    print("partial_eigvalsh self-check PASSED (values + gradients match eigvalsh).")


if __name__ == "__main__":
    _self_check()
