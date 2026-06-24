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
