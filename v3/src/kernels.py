"""
CPU/GPU kernels for the tick hot path.

Backends:
  numpy  — vectorized NumPy (always available)
  numba  — LLVM JIT on CPU (fallback if requested numba missing)
  cuda   — Numba CUDA (fallback to numba, then numpy, if no GPU)

Call `resolve_and_set(requested)` once per engine. Kernels are deterministic
for drain / popcount / nearest_food / age_curves given the same inputs.
Move RNG stays on the host (NumPy Generator) so seeds still match on CPU.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

_BACKEND = "numpy"
_NUMBA_OK = False
_CUDA_OK = False

try:
    from numba import njit

    _NUMBA_OK = True
except ImportError:  # pragma: no cover
    def njit(*_a, **_k):  # type: ignore
        def deco(fn):
            return fn
        if _a and callable(_a[0]):
            return _a[0]
        return deco


def cuda_available() -> bool:
    try:
        from numba import cuda
        return bool(cuda.is_available())
    except Exception:
        return False


def numba_available() -> bool:
    return _NUMBA_OK


def resolve_backend(requested: str) -> str:
    req = (requested or "numpy").lower()
    if req in ("cuda", "numba_cuda"):
        if cuda_available():
            return "cuda"
        if _NUMBA_OK:
            return "numba"
        return "numpy"
    if req == "numba":
        return "numba" if _NUMBA_OK else "numpy"
    return "numpy"


def set_backend(name: str) -> str:
    global _BACKEND, _CUDA_OK
    _BACKEND = resolve_backend(name)
    _CUDA_OK = _BACKEND == "cuda"
    return _BACKEND


def current_backend() -> str:
    return _BACKEND


def resolve_and_set(requested: str) -> str:
    return set_backend(requested)


# ---------------------------------------------------------------------------
# popcount
# ---------------------------------------------------------------------------

@njit(cache=True)
def _popcount32_njit(x: NDArray[np.uint32]) -> NDArray[np.int32]:
    n = x.shape[0]
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        v = np.uint32(x[i])
        c = np.int32(0)
        for _b in range(32):
            c += np.int32(v & np.uint32(1))
            v = v >> np.uint32(1)
        out[i] = c
    return out


def popcount32_numpy(x: NDArray[np.uint32]) -> NDArray[np.int32]:
    vals = np.asarray(x, dtype=np.uint32).copy()
    n = np.zeros(vals.shape, dtype=np.int32)
    for _ in range(32):
        n += (vals & np.uint32(1)).astype(np.int32)
        vals >>= np.uint32(1)
    return n


_CUDA_POPCOUNT = None
_CUDA_NEAREST = None
_CUDA_DRAIN = None


def _cuda_popcount_kernel():
    global _CUDA_POPCOUNT
    if _CUDA_POPCOUNT is None:
        from numba import cuda

        @cuda.jit
        def kernel(src, dst):
            i = cuda.grid(1)
            if i < src.size:
                v = src[i]
                c = 0
                for _b in range(32):
                    c += int(v & np.uint32(1))
                    v = v >> np.uint32(1)
                dst[i] = c

        _CUDA_POPCOUNT = kernel
    return _CUDA_POPCOUNT


def _popcount32_cuda(x: NDArray[np.uint32]) -> NDArray[np.int32]:
    from numba import cuda

    x_d = cuda.to_device(np.ascontiguousarray(x, dtype=np.uint32))
    out_d = cuda.device_array(x.shape[0], dtype=np.int32)
    threads = 128
    blocks = max(1, (x.shape[0] + threads - 1) // threads)
    _cuda_popcount_kernel()[blocks, threads](x_d, out_d)
    return out_d.copy_to_host()


def popcount32(x: NDArray[np.uint32]) -> NDArray[np.int32]:
    x = np.asarray(x, dtype=np.uint32)
    if x.size == 0:
        return np.zeros(0, dtype=np.int32)
    if _BACKEND == "cuda" and x.size >= 256:
        try:
            return _popcount32_cuda(x)
        except Exception:
            pass
    if _BACKEND in ("numba", "cuda") and _NUMBA_OK:
        return _popcount32_njit(x)
    return popcount32_numpy(x)


# ---------------------------------------------------------------------------
# nearest food
# ---------------------------------------------------------------------------

@njit(cache=True)
def _nearest_food_njit(ax, ay, fx, fy, radius, width, height):
    n = ax.shape[0]
    m = fx.shape[0]
    in_range = np.zeros(n, dtype=np.bool_)
    tx = ax.copy()
    ty = ay.copy()
    r2 = radius * radius
    half_w = width // 2
    half_h = height // 2
    for i in range(n):
        best = r2 + 1
        bx = ax[i]
        by = ay[i]
        axi = ax[i]
        ayi = ay[i]
        for j in range(m):
            dx = (fx[j] - axi) % width
            if dx > half_w:
                dx -= width
            dy = (fy[j] - ayi) % height
            if dy > half_h:
                dy -= height
            d2 = dx * dx + dy * dy
            if d2 < best:
                best = d2
                bx = fx[j]
                by = fy[j]
        if best <= r2:
            in_range[i] = True
            tx[i] = bx
            ty[i] = by
    return in_range, tx, ty


def nearest_food_numpy(ax, ay, fx, fy, radius, width, height):
    n = ax.shape[0]
    tx = ax.copy()
    ty = ay.copy()
    if n == 0 or fx.size == 0:
        return np.zeros(n, dtype=bool), tx, ty
    raw_x = np.mod(fx[None, :] - ax[:, None], width)
    dx = np.where(raw_x > width // 2, raw_x - width, raw_x)
    raw_y = np.mod(fy[None, :] - ay[:, None], height)
    dy = np.where(raw_y > height // 2, raw_y - height, raw_y)
    d2 = dx * dx + dy * dy
    nearest = np.argmin(d2, axis=1)
    rows = np.arange(n)
    best = d2[rows, nearest]
    in_range = best <= radius * radius
    tx = ax.copy()
    ty = ay.copy()
    tx[in_range] = fx[nearest][in_range].astype(np.int32)
    ty[in_range] = fy[nearest][in_range].astype(np.int32)
    return in_range, tx, ty


def _cuda_nearest_kernel():
    global _CUDA_NEAREST
    if _CUDA_NEAREST is None:
        from numba import cuda

        @cuda.jit
        def kernel(ax_, ay_, fx_, fy_, radius_, width_, height_, in_r, tx, ty):
            i = cuda.grid(1)
            if i >= ax_.size:
                return
            r2 = radius_ * radius_
            half_w = width_ // 2
            half_h = height_ // 2
            best = r2 + 1
            bx = ax_[i]
            by = ay_[i]
            axi = ax_[i]
            ayi = ay_[i]
            for j in range(fx_.size):
                dx = (fx_[j] - axi) % width_
                if dx > half_w:
                    dx -= width_
                dy = (fy_[j] - ayi) % height_
                if dy > half_h:
                    dy -= height_
                d2 = dx * dx + dy * dy
                if d2 < best:
                    best = d2
                    bx = fx_[j]
                    by = fy_[j]
            if best <= r2:
                in_r[i] = np.uint8(1)
                tx[i] = bx
                ty[i] = by
            else:
                in_r[i] = np.uint8(0)
                tx[i] = axi
                ty[i] = ayi

        _CUDA_NEAREST = kernel
    return _CUDA_NEAREST


def _nearest_food_cuda(ax, ay, fx, fy, radius, width, height):
    from numba import cuda

    n = ax.shape[0]
    ax_d = cuda.to_device(np.ascontiguousarray(ax, dtype=np.int32))
    ay_d = cuda.to_device(np.ascontiguousarray(ay, dtype=np.int32))
    fx_d = cuda.to_device(np.ascontiguousarray(fx, dtype=np.int32))
    fy_d = cuda.to_device(np.ascontiguousarray(fy, dtype=np.int32))
    in_d = cuda.device_array(n, dtype=np.uint8)
    tx_d = cuda.device_array(n, dtype=np.int32)
    ty_d = cuda.device_array(n, dtype=np.int32)
    threads = 128
    blocks = max(1, (n + threads - 1) // threads)
    _cuda_nearest_kernel()[blocks, threads](
        ax_d, ay_d, fx_d, fy_d,
        int(radius), int(width), int(height),
        in_d, tx_d, ty_d,
    )
    return in_d.copy_to_host().astype(bool), tx_d.copy_to_host(), ty_d.copy_to_host()


def nearest_food(ax, ay, fx, fy, radius, width, height):
    ax = np.asarray(ax, dtype=np.int32)
    ay = np.asarray(ay, dtype=np.int32)
    fx = np.asarray(fx, dtype=np.int32)
    fy = np.asarray(fy, dtype=np.int32)
    n = ax.shape[0]
    if n == 0 or fx.size == 0:
        return np.zeros(n, dtype=bool), ax.copy(), ay.copy()
    if _BACKEND == "cuda" and n >= 256:
        try:
            return _nearest_food_cuda(ax, ay, fx, fy, radius, width, height)
        except Exception:
            pass
    if _BACKEND in ("numba", "cuda") and _NUMBA_OK:
        return _nearest_food_njit(ax, ay, fx, fy, int(radius), int(width), int(height))
    return nearest_food_numpy(ax, ay, fx, fy, radius, width, height)


# ---------------------------------------------------------------------------
# energy drain
# ---------------------------------------------------------------------------

def apply_drain(
    energy: NDArray[np.float32],
    weight: NDArray[np.float32],
    speed: NDArray[np.float32],
    base: float,
    k_ws: float,
    defense: Optional[NDArray[np.uint32]] = None,
    k_def: float = 0.0,
    defense_cost: bool = False,
) -> None:
    """In-place energy drain, clamped to [0, 1]."""
    n = energy.shape[0]
    drain = np.float32(base) + np.float32(k_ws) * weight * speed
    if defense_cost and defense is not None:
        drain = drain + np.float32(k_def) * popcount32(defense).astype(np.float32)
    if _BACKEND == "cuda" and n >= 256:
        try:
            _apply_drain_cuda(energy, drain)
            return
        except Exception:
            pass
    np.clip(energy - drain.astype(np.float32), 0.0, 1.0, out=energy)


def _cuda_drain_kernel():
    global _CUDA_DRAIN
    if _CUDA_DRAIN is None:
        from numba import cuda

        @cuda.jit
        def kernel(e, d):
            i = cuda.grid(1)
            if i < e.size:
                v = e[i] - d[i]
                if v < 0.0:
                    v = 0.0
                if v > 1.0:
                    v = 1.0
                e[i] = v

        _CUDA_DRAIN = kernel
    return _CUDA_DRAIN


def _apply_drain_cuda(energy: NDArray[np.float32], drain: NDArray[np.float32]) -> None:
    from numba import cuda

    host = np.ascontiguousarray(energy, dtype=np.float32)
    e_d = cuda.to_device(host)
    d_d = cuda.to_device(np.ascontiguousarray(drain, dtype=np.float32))
    threads = 128
    blocks = max(1, (energy.size + threads - 1) // threads)
    _cuda_drain_kernel()[blocks, threads](e_d, d_d)
    e_d.copy_to_host(host)
    energy[:] = host


# ---------------------------------------------------------------------------
# packed DNA helpers (init / extract — not every tick)
# ---------------------------------------------------------------------------

@njit(cache=True)
def pack_bits_njit(bits: NDArray[np.uint8]) -> NDArray[np.uint64]:
    n, length = bits.shape
    words = (length + 63) // 64
    out = np.zeros((n, words), dtype=np.uint64)
    for i in range(n):
        for b in range(length):
            if bits[i, b]:
                w = b // 64
                off = b % 64
                out[i, w] |= np.uint64(1) << np.uint64(off)
    return out


def warmup() -> None:
    """Compile Numba kernels so the first timed tick is not a JIT stall."""
    if not _NUMBA_OK:
        return
    x = np.array([1, 3, 7], dtype=np.uint32)
    _popcount32_njit(x)
    ax = np.array([0, 1], dtype=np.int32)
    ay = np.array([0, 1], dtype=np.int32)
    fx = np.array([0], dtype=np.int32)
    fy = np.array([0], dtype=np.int32)
    _nearest_food_njit(ax, ay, fx, fy, 2, 8, 8)
    pack_bits_njit(np.ones((2, 64), dtype=np.uint8))
    if cuda_available():
        try:
            _popcount32_cuda(np.arange(256, dtype=np.uint32))
            _nearest_food_cuda(
                np.zeros(256, dtype=np.int32),
                np.zeros(256, dtype=np.int32),
                np.zeros(4, dtype=np.int32),
                np.zeros(4, dtype=np.int32),
                4, 16, 16,
            )
            e = np.ones(256, dtype=np.float32)
            _apply_drain_cuda(e, np.full(256, 0.1, dtype=np.float32))
        except Exception:
            pass
