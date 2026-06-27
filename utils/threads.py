"""
Central CPU / thread budget for the fitting code.

Two kinds of parallelism coexist and must NOT oversubscribe the node:

  1. k-point parallelism   -- num_cores worker *processes* (an mp.Pool over the
     k-points; the primary, most efficient axis).
  2. linear-algebra threads -- the eigensolve (torch.linalg.eigh / eigvalsh) and
     BLAS, controlled per-process by torch.set_num_threads / OMP_NUM_THREADS.

The single invariant enforced everywhere is

        num_cores * blas_threads_per_worker  <=  available_cpus

so the two layers TILE the node instead of fighting over it. When num_cores==0
(serial k-point loop) a single process gets the linear-algebra threads.

Why a thread CAP (and not just available//num_cores)? Measured on a Perlmutter
CPU node (AMD EPYC 7763, complex128):
  * eigh WITH eigenvectors (the training/backward path): ~4x at 8 threads,
    ~5x at 16, then flattens -- decent up to ~16.
  * eigvalsh (eigenvalues only, the no-grad eval path): peaks ~2x at 4-8 threads
    and gets SLOWER beyond ~16.
A single eigensolve therefore never benefits from the whole node; the efficient
way to fill 256 logical CPUs is many k-point processes, each with a MODEST
thread count. The auto policy below caps per-worker threads at LINALG_THREAD_CAP
to stay in the productive region; raise num_cores (toward the k-point count) to
use more of the node. A user can override with num_threads>0 (still clamped so
the invariant holds).

The SO/NL integral initialization is a separate, one-shot phase that runs in the
main process BEFORE any k-point worker pool exists, so it may use all available
CPUs as plain Python threads -- the numpy/scipy integral kernels release the GIL
and parallelize cleanly over k-points (see init_thread_count()).
"""
import os

# Per-worker linear-algebra thread cap for the AUTO policy. Past this, the
# Hermitian eigensolve stops scaling (see module docstring); extra node capacity
# is better spent on more k-point processes (num_cores).
LINALG_THREAD_CAP = 16


def available_cpus():
    """CPUs actually usable by this process.

    Uses sched_getaffinity so a SLURM/cgroup cpuset binding (a node shared
    between jobs) is respected rather than the raw hardware count.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def plan_blas_threads(num_cores, num_threads=0, available=None):
    """Linear-algebra threads per *process*, chosen so the k-point process layer
    and the BLAS/eigensolve thread layer don't oversubscribe the node.

    num_cores   : k-point worker processes (0 == serial, one process).
    num_threads : 0/<=0 means AUTO (fill evenly, capped at LINALG_THREAD_CAP);
                  a positive value is honored but clamped so num_cores*threads
                  never exceeds `available`.
    """
    if available is None:
        available = available_cpus()
    ncore = max(1, num_cores)                 # num_cores==0 -> single process
    cap = max(1, available // ncore)          # hard ceiling: no oversubscription
    if num_threads and num_threads > 0:
        return max(1, min(num_threads, cap))
    return max(1, min(cap, LINALG_THREAD_CAP))


def init_thread_count(num_threads=0, available=None):
    """Thread count for the one-shot SO/NL integral initialization.

    This phase owns the whole node (no worker pool is live yet) and the integral
    kernels release the GIL, so it scales over k-points with all CPUs. Honors an
    explicit positive num_threads, otherwise uses every available CPU.
    """
    if available is None:
        available = available_cpus()
    if num_threads and num_threads > 0:
        return max(1, min(num_threads, available))
    return available


def set_process_threads(n):
    """Set the linear-algebra thread count for THIS process.

    torch.set_num_threads is runtime-safe and is what the eigensolve (ATen/MKL)
    actually honors. The env vars are best-effort for already-imported BLAS and,
    importantly, are inherited by freshly spawned/forked workers.
    """
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
    return n


def pool_worker_init(n):
    """mp.Pool initializer: pin each worker to `n` linear-algebra threads.

    This is the piece that finally lets the per-k-point eigensolve use >1 thread
    while keeping the workers, collectively, within the node budget. Must stay a
    top-level function so it is picklable by the spawn start method.
    """
    set_process_threads(n)
