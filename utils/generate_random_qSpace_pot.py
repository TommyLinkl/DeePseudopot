"""
Generate smooth random functions on an interval [xmin, xmax].

Run like: 
$ python generate_random_qSpace_pot.py ./output_files -e H P Al --seed 12345

Methods:
  - 'fourier': Random low-frequency Fourier series (fast, no SciPy required).
  - 'gp':      Gaussian Process with RBF kernel sampled on a grid (smooth, flexible).

Returns a list of callables f_i(x) you can evaluate on arbitrary x in [xmin, xmax].
If SciPy is available, uses CubicSpline for smooth interpolation; otherwise uses linear.

Example:
    funcs = generate_random_smooth_functions(
        n_funcs=5, xmin=0.0, xmax=10.0, method='fourier',
        fourier_max_freq=4, fourier_decay=1.5, seed=42
    )
    # Evaluate and plot
    import numpy as np, matplotlib.pyplot as plt
    xs = np.linspace(0, 10, 500)
    for f in funcs:
        plt.plot(xs, f(xs), lw=2, alpha=0.9)
    plt.xlabel("x"); plt.ylabel("f(x)"); plt.title("Random smooth functions")
    plt.show()
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
import argparse

try:
    from scipy.interpolate import CubicSpline
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _smooth_interpolant(xg: np.ndarray, yg: np.ndarray):
    """Return a callable interpolant (CubicSpline if available, else linear)."""
    if _HAS_SCIPY:
        return CubicSpline(xg, yg, bc_type="natural")
    else:
        # Linear fallback; still continuous, less smooth between grid points
        def f(x):
            x = np.asarray(x)
            return np.interp(x, xg, yg)
        return f


def _randn(rng, *shape):
    return rng.standard_normal(size=shape)


def _generate_fourier(rng, xg, max_freq=4, decay=1.5, zero_mean=True):
    """
    Random low-frequency Fourier series:
        f(x) = a0 + Σ_k [A_k cos(2π k u) + B_k sin(2π k u)] / k^decay
    where u maps x in [xmin, xmax] to [0,1]. Larger 'decay' -> smoother.
    """
    u = (xg - xg[0]) / (xg[-1] - xg[0])
    y = np.zeros_like(u)
    a0 = _randn(rng) * (1.0 if not zero_mean else 0.0)
    y += a0
    for k in range(1, max_freq + 1):
        scale = k ** (-decay)
        A_k = _randn(rng)
        B_k = _randn(rng)
        y += scale * (A_k * np.cos(2 * np.pi * k * u) + B_k * np.sin(2 * np.pi * k * u))
    return y


def _rbf_kernel(x, y, lengthscale, variance):
    """Squared-exponential (RBF) kernel."""
    x = x[:, None]
    y = y[None, :]
    d2 = (x - y) ** 2
    return variance * np.exp(-0.5 * d2 / (lengthscale ** 2))


def _generate_gp(rng, xg, lengthscale=1.5, variance=1.0, jitter=1e-9):
    """
    Sample from a zero-mean GP with RBF kernel on grid xg, then return the values.
    """
    K = _rbf_kernel(xg, xg, lengthscale=lengthscale, variance=variance)
    # Add tiny jitter for numerical stability
    K.flat[::K.shape[0] + 1] += jitter
    # Cholesky and sample
    L = np.linalg.cholesky(K)
    z = _randn(rng, xg.size)
    return L @ z


def generate_random_smooth_functions(
    n_funcs: int,
    xmin: float = 0.0,
    xmax: float = 10.0,
    method: str = "fourier",
    n_grid: int = 512,
    # Fourier-specific
    fourier_max_freq: int = 5,
    fourier_decay: float = 1.5,
    # GP-specific
    gp_lengthscale: float = 1.5,
    gp_variance: float = 1.0,
    # Common
    normalize: bool = True,
    amplitude: float = 1.0,
    vertical_shift: float = 0.0,
    seed: int | None = None,
):
    """
    Create n_funcs smooth random functions on [xmin, xmax].

    Parameters
    ----------
    method : {'fourier', 'gp'}
        'fourier' → random band-limited Fourier series (fast).
        'gp'      → Gaussian process sample with RBF kernel (flexible).
    n_grid : int
        Internal grid resolution for constructing the interpolant.
    fourier_max_freq : int
        Highest harmonic in Fourier method. Smaller => smoother.
    fourier_decay : float
        Spectral decay exponent. Larger => smoother.
    gp_lengthscale : float
        RBF kernel lengthscale. Larger => smoother.
    gp_variance : float
        Output variance for GP samples.
    normalize : bool
        If True, each function is standardized (zero mean, unit std) before
        scaling by 'amplitude' and shifting by 'vertical_shift'.
    amplitude : float
        Overall vertical scale after (optional) normalization.
    vertical_shift : float
        Constant added at the end.
    seed : int | None
        Seed for reproducibility.

    Returns
    -------
    list of callables f(x)
    """
    rng = np.random.default_rng(seed)
    xg = np.linspace(xmin, xmax, n_grid)

    funcs = []
    for _ in range(n_funcs):
        if method == "fourier":
            yg = _generate_fourier(rng, xg, max_freq=fourier_max_freq, decay=fourier_decay)
        elif method == "gp":
            yg = _generate_gp(rng, xg, lengthscale=gp_lengthscale, variance=gp_variance)
        else:
            raise ValueError("method must be 'fourier' or 'gp'.")

        if normalize:
            m, s = yg.mean(), yg.std()
            s = s if s > 1e-12 else 1.0
            yg = (yg - m) / s

        yg = amplitude * yg + vertical_shift
        funcs.append(_smooth_interpolant(xg, yg))

    return funcs


if __name__ == "__main__":
    # ---------------------------
    # Command line arguments
    # ---------------------------
    parser = argparse.ArgumentParser(
        description="Generate random smooth functions and save them to files."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="./output_files",
        help="Directory where the output files will be saved (default: ./output_files)",
    )
    parser.add_argument(
        "-e",
        "--elements",
        nargs="+",
        default=["H"],
        help="List of element symbols (one random function per element).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducibility (default: 12345).",
    )
    args = parser.parse_args()

    # ---------------------------
    # Setup
    # ---------------------------
    elements = sorted(args.elements)
    if not elements:
        raise ValueError("At least one element symbol must be provided.")

    xs = np.linspace(0, 30, 4096)
    sigma = 2.0
    seed = args.seed  # Seed for reproducibility
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Generate functions
    # ---------------------------
    f_fourier = generate_random_smooth_functions(
        n_funcs=len(elements),
        xmin=0,
        xmax=10,
        method="fourier",
        fourier_max_freq=4,
        fourier_decay=1.6,
        amplitude=40.0,
        seed=seed,
    )
    f_gp = generate_random_smooth_functions(
        n_funcs=len(elements),
        xmin=0,
        xmax=10,
        method="gp",
        gp_lengthscale=1.2,
        gp_variance=1.0,
        amplitude=40.0,
        seed=seed,
    )

    # Pairwise sum of Fourier and GP components for each element
    funcs = [lambda x, f1=f1, f2=f2: f1(x) + f2(x) for f1, f2 in zip(f_fourier, f_gp)]

    # ---------------------------
    # Save output
    # ---------------------------
    envelope = np.exp(-xs**2 / (2 * sigma**2))
    values = [f(xs) * envelope for f in funcs]
    data = np.column_stack([xs] + values)

    filename = output_dir / "rand_gen_qSpace_pot.dat"
    header_cols = ["# q"] + [f"v(q)_{el}" for el in elements]
    header = "        ".join(header_cols)
    np.savetxt(filename, data, header=header, comments="", fmt="%.8f")
    print(f"Saved {filename}")
