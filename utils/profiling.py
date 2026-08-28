"""
Lightweight aggregating profiler for timing and memory, gated on the NNConfig
'runtime_flag' (timing) and 'memory_flag' (memory) switches.

Purpose: answer "where is the bottleneck?" by accumulating per-section timing
across the thousands of (k-point x epoch) calls in the training inner loop and
printing ONE sorted table with an explicit Hamiltonian-build vs diagonalization
diagnosis. The old approach (a `print(... elapsed ...)` per call) produced
thousands of un-aggregated lines that could not reveal where time actually went.

Overhead when disabled: a single boolean check per section. When enabled: one
perf_counter() pair + a dict update per section (microseconds) -- negligible
next to an eigensolve.

Multiprocessing: every worker process imports this module and gets its own
module-level PROF. In the separateKptGrad mp path the heavy build+diagonalize
work runs in the workers, so each worker returns PROF.snapshot() and the parent
calls PROF.merge() to fold that time into the aggregated report.
"""
import time
import os
import sys
from contextlib import contextmanager

try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False

try:
    import resource
    _HAVE_RESOURCE = True
except Exception:
    _HAVE_RESOURCE = False


class Profiler:
    # Inner-loop "leaf" sections that PARTITION the per-k-point work (mutually
    # exclusive, no nesting), so their accumulated times sum to ~the inner-loop
    # wall time and the percentages are meaningful.
    BUILD_KEYS = ("Htot_kinetic", "Htot_Vloc", "Htot_SO", "Htot_NL")
    DIAG_KEYS = ("diag",)

    def __init__(self):
        self.time_on = False
        self.mem_on = False
        self._t = {}   # name -> [count, total_s, min_s, max_s]

    def configure(self, runtime_flag=False, memory_flag=False):
        self.time_on = bool(runtime_flag)
        self.mem_on = bool(memory_flag)

    def reset(self):
        self._t = {}

    def record(self, name, dt):
        e = self._t.get(name)
        if e is None:
            self._t[name] = [1, dt, dt, dt]
        else:
            e[0] += 1
            e[1] += dt
            if dt < e[2]:
                e[2] = dt
            if dt > e[3]:
                e[3] = dt

    @contextmanager
    def time(self, name):
        """Context manager: `with PROF.time("diag"): ...`. No-op when timing off."""
        if not self.time_on:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - t0)

    # ----- memory -----------------------------------------------------------
    def current_rss_gb(self):
        if _HAVE_PSUTIL:
            return psutil.Process().memory_info().rss / (1024 ** 3)
        # Dependency-free fallback (Linux): RSS pages from /proc/self/statm.
        try:
            with open("/proc/self/statm") as f:
                rss_pages = int(f.read().split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return rss_pages * page_size / (1024 ** 3)
        except Exception:
            return float('nan')

    def peak_rss_gb(self):
        """OS-tracked high-water mark (free; no sampling needed)."""
        if _HAVE_RESOURCE:
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is kB on Linux, bytes on macOS.
            if sys.platform == "darwin":
                return ru / (1024 ** 3)
            return ru / (1024 ** 2)
        return self.current_rss_gb()

    def mem_checkpoint(self, label):
        """Print current RSS and the process high-water peak at a named point."""
        if not self.mem_on:
            return
        rss = self.current_rss_gb()
        peak = self.peak_rss_gb()
        extra = ""
        try:
            import torch
            if torch.cuda.is_available():
                extra = (f"   cuda_alloc={torch.cuda.memory_allocated()/(1024**3):6.3f} GB"
                         f"   cuda_peak={torch.cuda.max_memory_allocated()/(1024**3):6.3f} GB")
        except Exception:
            pass
        print(f"[mem] {label:<34s} RSS={rss:7.3f} GB   peak={peak:7.3f} GB{extra}", flush=True)

    # ----- snapshot / merge for multiprocessing -----------------------------
    def snapshot(self):
        """Picklable copy of the accumulated timers (for return from mp workers)."""
        if not self.time_on:
            return None
        return {k: list(v) for k, v in self._t.items()}

    def merge(self, snap):
        """Fold another profiler's snapshot (e.g. from an mp worker) into this one."""
        if not snap:
            return
        for name, e in snap.items():
            cur = self._t.get(name)
            if cur is None:
                self._t[name] = list(e)
            else:
                cur[0] += e[0]
                cur[1] += e[1]
                if e[2] < cur[2]:
                    cur[2] = e[2]
                if e[3] > cur[3]:
                    cur[3] = e[3]

    # ----- report -----------------------------------------------------------
    def report(self, title="RUNTIME PROFILE", reset=False):
        if not self.time_on or not self._t:
            return
        try:
            import torch
            nthreads = torch.get_num_threads()
        except Exception:
            nthreads = os.environ.get("OMP_NUM_THREADS", "?")

        rows = sorted(self._t.items(), key=lambda kv: kv[1][1], reverse=True)
        grand = sum(e[1] for _, e in rows)

        print(f"\n{'=' * 84}")
        print(f"{title}   (torch/BLAS threads = {nthreads})")
        print(f"{'-' * 84}")
        print(f"{'section':<24}{'count':>9}{'total(s)':>11}{'mean(ms)':>11}"
              f"{'min(ms)':>10}{'max(ms)':>10}{'%tot':>8}")
        for name, (c, tot, mn, mx) in rows:
            pct = 100.0 * tot / grand if grand > 0 else 0.0
            print(f"{name:<24}{c:>9d}{tot:>11.3f}{1e3 * tot / c:>11.3f}"
                  f"{1e3 * mn:>10.3f}{1e3 * mx:>10.3f}{pct:>7.1f}%")
        print(f"{'-' * 84}")

        build = sum(self._t[k][1] for k in self.BUILD_KEYS if k in self._t)
        diag = sum(self._t[k][1] for k in self.DIAG_KEYS if k in self._t)
        bd = build + diag
        if bd > 0:
            print("DIAGNOSIS  (Hamiltonian construction vs diagonalization):")
            print(f"  build (Htot_kinetic+Vloc+SO+NL) : {build:9.3f} s  "
                  f"({100 * build / bd:5.1f}% of build+diag)")
            print(f"  diag  (eigvalsh / eigh)         : {diag:9.3f} s  "
                  f"({100 * diag / bd:5.1f}% of build+diag)")
            if diag >= build:
                print("  -> DIAGONALIZATION dominates. Multi-threaded BLAS should help:")
                print("     raise num_threads (>1) and re-check the 'diag' mean(ms), or call")
                print("     utils.profiling.benchmark_eigensolve(ndim) to see the speedup curve.")
            else:
                print("  -> HAMILTONIAN CONSTRUCTION dominates. Threading the eigensolve will")
                print("     help little; focus on buildHtot (Htot_Vloc, the NN form factor, is")
                print("     usually the largest piece). Keeping BLAS single-threaded is fine.")
        print(f"{'=' * 84}\n", flush=True)
        if reset:
            self.reset()


# Module-level singleton (mirrors the existing utils/memory.py global pattern).
PROF = Profiler()


# ---------------------------------------------------------------------------
# Heuristic (a-priori) memory estimator.
#
# Motivation: the RSS-based memory_flag checkpoints (mem_checkpoint above) read
# the OS resident-set size, which is hard to interpret once the run fans out
# into many k-point worker processes -- shared-memory SO/NL segments are counted
# per-process, private per-worker graphs come and go, and the "peak" any single
# process sees is not the node total. The dominant costs, however, are known a
# priori: they are fixed by the plane-wave basis size, the numerical precision,
# the number of matrix groups / k-points, and the NN depth+width. This function
# totals them from those numbers alone -- no RAM sensing -- and prints a table.
# It runs unconditionally (independent of memory_flag) so every run leaves a
# record of its expected peak footprint.
#
# The matrices themselves (SO/NL/Vloc/Htot) are allocated as complex128 in
# ham.py regardless of the torch default dtype; the NN forward/graph runs in the
# torch default real dtype. Both are reported in the header.
# ---------------------------------------------------------------------------

_GB = 1024.0 ** 3


def _fmt_bytes(nbytes):
    """Human-readable GB with a MB fallback for small terms."""
    gb = nbytes / _GB
    if gb >= 0.01 or nbytes == 0:
        return f"{gb:8.3f} GB"
    return f"{nbytes / (1024.0 ** 2):8.2f} MB"


def _linear_layer_widths(model):
    """
    (in_features, out_features) for every nn.Linear reachable from `model`,
    in module-registration order. Works across all the Net_* wrappers because
    it walks .modules() rather than assuming a particular attribute name.
    Returns [] if torch/model is unavailable.
    """
    try:
        import torch.nn as nn
    except Exception:
        return []
    if model is None:
        return []
    widths = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            widths.append((m.in_features, m.out_features))
    return widths


def estimate_peak_memory(hams, NNConfig, PPmodel=None, spinModel=None,
                         LSDmodels=None, title="ESTIMATED PEAK MEMORY (heuristic)"):
    """
    Print a table of the estimated peak RAM footprint of a DeePseudopot run,
    computed analytically from the basis size, numerical precision, k-point
    count, matrix-group count, and NN depth/width -- WITHOUT calling any
    RAM-sensing utility. Always safe to call (guards every optional field);
    returns the estimated peak in bytes (or None if it cannot introspect hams).

    Memory model (per BulkSystem, with nbv = basis size, spinor doubling the
    Hamiltonian dimension to ndimH = 2*nbv):

        SO cache   : nkp * nMatGroups * (2*nbv)^2       * cbytes   [SObool]
        NL cache   : nkp * nMatGroups * 2 * ndimH^2     * cbytes   [NLbool]
        Vloc mat   : ndimH^2 * cbytes  +  gdiff/q (nbv^2*(3+1)*fbytes)
        Htot       : ndimH^2 * cbytes
        eigensolve : ~2 * ndimH^2 * cbytes  (LAPACK copy + eigenvectors that
                     torch.linalg.eigvalsh saves for the autograd backward)
        NN graph   : nbv^2 * 2*sum(layer_widths) * fbytes  (retained activations;
                     ~0 when checkpoint is on, since they are recomputed)
        NN params  : 4 * n_params * fbytes  (weights + grad + 2 Adam moments)

    The SO/NL caches are built once and held in shared memory for the whole run
    (summed across systems, counted once); everything else is transient per
    k-point worker and is multiplied by the number of concurrent workers
    (max(1, num_cores)). With low_mem/disk_cache on, the SO/NL caches live on
    disk and only one k-point per worker is resident, so they move into the
    per-worker column instead.
    """
    if not hams:
        print(f"[mem-est] no Hamiltonians available; skipping memory estimate.")
        return None

    # Numerical precision. The stored matrices are complex128 (ham.py hardcodes
    # np.complex128 / torch.complex128); the NN runs in the torch default real
    # dtype. Derive both so the header states exactly what was assumed.
    cbytes = 16  # complex128
    fbytes = 8   # float64 default
    default_dtype = "float64"
    try:
        import torch
        fbytes = torch.finfo(torch.get_default_dtype()).bits // 8
        default_dtype = str(torch.get_default_dtype()).replace("torch.", "")
    except Exception:
        pass

    checkpoint = bool(NNConfig.get('checkpoint', False))
    low_mem = bool(NNConfig.get('low_mem', False))
    num_cores = int(NNConfig.get('num_cores', 0) or 0)
    n_workers = max(1, num_cores)

    # NN graph + parameter cost is shared across systems (one PPmodel), but the
    # retained-activation batch is nbv^2, which is system-dependent. Collect the
    # per-model width sum and param count once here.
    nn_widths = _linear_layer_widths(PPmodel)
    width_sum = sum(a + b for a, b in nn_widths)   # sum over linear layers of (in+out)
    n_models = 0
    n_params = 0
    try:
        for mdl in [PPmodel, spinModel]:
            if mdl is not None:
                n_models += 1
                n_params += sum(p.numel() for p in mdl.parameters())
        if LSDmodels:
            for mdl in LSDmodels.values():
                if mdl is not None:
                    n_models += 1
                    n_params += sum(p.numel() for p in mdl.parameters())
    except Exception:
        pass
    # A spin model doubles the retained-activation graph (a second forward of the
    # same nbv^2 batch); LSD models add per-atom graphs but are architecture- and
    # geometry-specific, so we fold only the spin factor into the graph estimate.
    graph_forward_factor = 2 if (spinModel is not None) else 1

    # Parameter/optimizer memory (weights + grad + 2 Adam moments). Small, but
    # resident in the parent AND copied into each spawned worker.
    params_bytes = 4 * n_params * fbytes

    shared_bytes = 0          # allocated once for the whole run
    per_worker_bytes = 0      # transient; multiplied by n_workers below
    per_worker_sys = -1

    W = 84
    print(f"\n{'=' * W}")
    print(f"{title}")
    print(f"{'-' * W}")
    print(f"matrices: complex128 ({cbytes} B/elem)   NN/graph: {default_dtype} "
          f"({fbytes} B/elem)")
    print(f"workers (max(1,num_cores)) = {n_workers}   checkpoint = "
          f"{'ON' if checkpoint else 'OFF'}   low_mem = {'ON' if low_mem else 'OFF'}")
    if nn_widths:
        shp = " -> ".join([str(nn_widths[0][0])] + [str(b) for _, b in nn_widths])
        print(f"PPmodel: {len(nn_widths)} linear layers [{shp}], "
              f"{n_params:,} params across {n_models} model(s)")
    print(f"{'-' * W}")

    for iSys, ham in enumerate(hams):
        try:
            nbv = int(ham.basis.shape[0])
            nkp = int(ham.system.getNKpts())
            spinor = bool(ham.spinor)
            nMG = int(getattr(ham, 'nMatGroups', 1))
            SObool = bool(getattr(ham, 'SObool', False))
            NLbool = bool(getattr(ham, 'NLbool', False)) and bool(getattr(ham, 'checknl', True))
        except Exception as e:
            print(f"  system {iSys}: cannot introspect ham ({e}); skipped.")
            continue

        ndimH = 2 * nbv if spinor else nbv

        so_cache = nkp * nMG * (2 * nbv) ** 2 * cbytes if SObool else 0
        nl_cache = nkp * nMG * 2 * ndimH ** 2 * cbytes if NLbool else 0
        # per-k-point (single kpt) versions, used when disk_cache holds the full
        # stack on disk and only one k-point is resident per worker.
        so_1k = nMG * (2 * nbv) ** 2 * cbytes if SObool else 0
        nl_1k = nMG * 2 * ndimH ** 2 * cbytes if NLbool else 0

        vloc = ndimH ** 2 * cbytes + nbv ** 2 * (3 + 1) * fbytes
        htot = ndimH ** 2 * cbytes
        eigsolve = 2 * ndimH ** 2 * cbytes
        nn_graph = 0 if checkpoint else nbv ** 2 * 2 * width_sum * fbytes * graph_forward_factor

        # per-worker transient for THIS system
        worker_sys = vloc + htot + eigsolve + nn_graph + params_bytes
        if low_mem:
            worker_sys += so_1k + nl_1k
        else:
            shared_bytes += so_cache + nl_cache

        if worker_sys > per_worker_bytes:
            per_worker_bytes = worker_sys
            per_worker_sys = iSys

        print(f"  system {iSys}: nbv={nbv}  nkpt={nkp}  ndimH={ndimH}  "
              f"nMatGroups={nMG}  spinor={spinor}  SO={SObool}  NL={NLbool}")
        if not low_mem:
            if SObool:
                print(f"      SO cache   (shared)        = {_fmt_bytes(so_cache)}"
                      f"    [{nkp}*{nMG}*(2*{nbv})^2*{cbytes}]")
            if NLbool:
                print(f"      NL cache   (shared)        = {_fmt_bytes(nl_cache)}"
                      f"    [{nkp}*{nMG}*2*{ndimH}^2*{cbytes}]")
        else:
            if SObool:
                print(f"      SO 1-kpt   (per worker)    = {_fmt_bytes(so_1k)}   [disk_cache]")
            if NLbool:
                print(f"      NL 1-kpt   (per worker)    = {_fmt_bytes(nl_1k)}   [disk_cache]")
        print(f"      Vloc + gdiff/q (per worker)= {_fmt_bytes(vloc)}")
        print(f"      Htot           (per worker)= {_fmt_bytes(htot)}")
        print(f"      eigensolve     (per worker)= {_fmt_bytes(eigsolve)}"
              f"    [copy + saved eigenvectors]")
        if checkpoint:
            print(f"      NN graph       (per worker)= {_fmt_bytes(0)}   "
                  f"[checkpoint ON -> activations recomputed]")
        else:
            print(f"      NN graph       (per worker)= {_fmt_bytes(nn_graph)}"
                  f"    [{nbv}^2*2*{width_sum}*{fbytes}"
                  f"{'*2(spin)' if graph_forward_factor == 2 else ''}]")

    per_worker_total = n_workers * per_worker_bytes
    peak = shared_bytes + per_worker_total

    print(f"{'-' * W}")
    print(f"  NN params + Adam state (per model set)   = {_fmt_bytes(params_bytes)}")
    print(f"  Shared caches (SO+NL, all systems, once) = {_fmt_bytes(shared_bytes)}")
    print(f"  Per-worker transient (worst system"
          f"{'' if per_worker_sys < 0 else f' #{per_worker_sys}'})     "
          f"= {_fmt_bytes(per_worker_bytes)}")
    print(f"  Per-worker x {n_workers} worker(s)"
          f"{' ' * max(1, 20 - len(str(n_workers)))}= {_fmt_bytes(per_worker_total)}")
    print(f"{'-' * W}")
    print(f"  ESTIMATED PEAK TOTAL                     = {_fmt_bytes(peak)}")
    print(f"{'=' * W}")
    print("  Heuristic upper-ish bound: sums the dominant matrices + NN graph; "
          "excludes")
    print("  interpreter/library baseline (~0.3-1 GB/process) and transient "
          "scratch.")
    print(f"{'=' * W}\n", flush=True)
    return peak


def benchmark_eigensolve(ndim, dtype=None, thread_counts=(1, 2, 4, 8, 16), repeats=3):
    """
    Time a Hermitian eigenvalue solve (torch.linalg.eigvalsh) on a representative
    ndim x ndim matrix across BLAS/torch thread counts. This directly answers
    "would OpenMP-style multi-threaded diagonalization help on MY matrices?"
    without needing a full training run. Prints a small speedup table and
    restores the previous thread count afterwards.
    """
    import torch
    if dtype is None:
        dtype = torch.complex128

    try:
        ncpu = os.cpu_count() or 1
    except Exception:
        ncpu = 1
    thread_counts = sorted({nt for nt in thread_counts if 1 <= nt <= ncpu})
    if not thread_counts:
        thread_counts = [1]

    # One representative Hermitian matrix, reused across thread counts.
    A = torch.randn(ndim, ndim, dtype=dtype)
    H = A + A.conj().T

    print(f"\n{'=' * 60}")
    print(f"EIGENSOLVE BENCHMARK   ndim={ndim}, dtype={dtype}, cpus={ncpu}")
    print(f"{'-' * 60}")
    print(f"{'threads':>8}{'mean(s)':>13}{'speedup':>11}{'efficiency':>12}")
    base = None
    prev = torch.get_num_threads()
    try:
        for nt in thread_counts:
            torch.set_num_threads(nt)
            torch.linalg.eigvalsh(H)  # warmup at this thread count
            t0 = time.perf_counter()
            for _ in range(repeats):
                torch.linalg.eigvalsh(H)
            dt = (time.perf_counter() - t0) / repeats
            if base is None:
                base = dt
            eff = (base / dt) / nt
            print(f"{nt:>8d}{dt:>13.4f}{base / dt:>10.2f}x{100 * eff:>10.0f}%")
    finally:
        torch.set_num_threads(prev)
    print(f"{'-' * 60}")
    print("speedup = t(1 thread)/t(n);  efficiency = speedup/n.")
    print("If speedup stays near 1.0x, diagonalization does NOT benefit from more")
    print("threads at this size -> keep BLAS single-threaded and parallelize over")
    print("k-points instead. If it scales, set num_threads accordingly.")
    print(f"{'=' * 60}\n", flush=True)
