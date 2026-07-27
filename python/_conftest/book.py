"""The frozen worst-case book: `WorstBook`, folded during a run."""

import pathlib
import warnings
from typing import ClassVar

import numpy as np


class WorstBook:
    """A capped CSV per (op, accuracy, dtype, fma, source) under tests/worst/."""

    # dtype -> (bit-pattern uint, float, rows kept per source file). Headerless
    # `<bits>,<err>`: col0 is decimal, the only exact form np.loadtxt parses in C.
    # f32 keeps as many rows as f64: only test_exhaustive books it, and a sweep of
    # every finite input deserves the wider cap that sampling did not.
    _SPEC: ClassVar[dict[str, tuple[type, type, int]]] = {
        "float32": (np.uint32, np.float32, 10_000),
        "float64": (np.uint64, np.float64, 10_000),
    }
    _EPS = 1e-5  # stored errors round to 6 decimals; reruns must stay idempotent

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        # key -> (uint bits ascending, |ulp| err) of the live top cases
        self._book: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._base_max: dict[str, float] = {}  # per-key max ULP as committed
        self._dirty: set[str] = set()
        self._touched: set[str] = set()  # keys this process has already measured
        self.discoveries: dict[str, dict] = {}
        self.restamped: dict[str, tuple[float, float]] = {}

    @staticmethod
    def key(
        operation: str, accuracy: str, dtype: str, have_fma: bool, source: str
    ) -> str:
        # `parse` splits on "_", so `operation` must not contain one.
        fma = "fma" if have_fma else "nonfma"
        return f"{operation}_{accuracy.lower()}_{dtype}_{fma}_{source}"

    @staticmethod
    def parse(key: str) -> tuple[str, str, str, bool, str]:
        """`key()` inverted: `(operation, accuracy, dtype, have_fma, source)`."""
        operation, accuracy, dtype, fma, source = key.split("_", 4)
        return operation, accuracy, dtype, fma == "fma", source

    def all_keys(self) -> list[str]:
        """Every book on disk, sorted; the gate parametrizes over these."""
        if not self.root.is_dir():
            return []
        return sorted(
            f"{op_dir.name}_{path.stem}"
            for op_dir in self.root.iterdir()
            if op_dir.is_dir()
            for path in op_dir.glob("*.csv")
        )

    @staticmethod
    def source(label: str, *drop: str) -> str:
        """`file.py::test_simple[SSE4-Low-_basic-float32-sin]` -> `test_simple[_basic]`.

        Each parametrization books separately, or one cap is shared by unrelated
        data sets. `drop` removes the id segments the key already carries — the
        target most of all, since every non-FMA target would file the same rows.
        """
        base, sep, bracket = label.split("::")[-1].partition("[")
        dropped = {d.lower() for d in drop}
        kept = (
            [s for s in bracket.rstrip("]").split("-") if s.lower() not in dropped]
            if sep
            else []
        )
        return f"{base}[{'-'.join(kept)}]" if kept else base

    def _spec(self, key: str) -> tuple:
        """`(uint, float, cap)` of `<op>_<accuracy>_<dtype>_<fma>_<source>`."""
        return self._SPEC[self.parse(key)[2]]

    def _path(self, key: str) -> pathlib.Path:
        op, rest = key.split("_", 1)  # the op names its dir
        return self.root / op / f"{rest}.csv"

    def _row_dtype(self, key: str) -> np.dtype:
        return np.dtype([("bits", self._spec(key)[0]), ("err", "f8")])

    @staticmethod
    def _top(bits: np.ndarray, errs: np.ndarray, cap: int) -> tuple:
        """The `cap` worst rows, bits ascending (the `_lookup` invariant)."""
        if bits.size > cap:
            keep = np.argpartition(-errs, cap)[:cap]
            bits, errs = bits[keep], errs[keep]
        order = np.argsort(bits)
        return bits[order], errs[order]

    def _load(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if key not in self._book:
            u, _, cap = self._spec(key)
            path = self._path(key)
            if path.exists():
                with warnings.catch_warnings():  # an empty file is legitimate
                    warnings.simplefilter("ignore")
                    rec = np.loadtxt(
                        path, dtype=self._row_dtype(key), delimiter=",", ndmin=1
                    )
                pair = self._top(rec["bits"], rec["err"], cap)
            else:
                pair = (np.array([], u), np.array([], np.float64))
            self._book[key] = pair
            self._base_max[key] = float(pair[1].max(initial=0.0))
        return self._book[key]

    @staticmethod
    def _member(bits: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Mask of `bits` that appear in the ascending `other`."""
        if other.size == 0:
            return np.zeros(bits.shape, bool)
        pos = np.clip(np.searchsorted(other, bits), 0, other.size - 1)
        return other[pos] == bits

    @classmethod
    def _lookup(cls, bits0: np.ndarray, err0: np.ndarray, query: np.ndarray):
        """`err0` for each `query` bit in the ascending `bits0`, else -inf."""
        if bits0.size == 0:
            return np.full(query.shape, -np.inf)
        pos = np.clip(np.searchsorted(bits0, query), 0, bits0.size - 1)
        return np.where(bits0[pos] == query, err0[pos], -np.inf)

    @staticmethod
    def _dedupe(bits: np.ndarray, errs: np.ndarray) -> tuple:
        """Worst error per input, bits ascending."""
        order = np.argsort(bits, kind="stable")
        bits, first = np.unique(bits[order], return_index=True)
        return bits, np.maximum.reduceat(errs[order], first)

    def _stale(self, key: str) -> bool:
        """True once: `key` still holds disk errors, not this run's measurements."""
        first = key not in self._touched
        self._touched.add(key)
        return first

    def fold(self, key: str, x: np.ndarray, err: np.ndarray) -> None:
        """Merge measured (x, |ulp|) into `key`'s book; `x` must be its dtype."""
        x = np.asarray(x)
        err = np.asarray(err, dtype=np.float64)
        keep = np.isfinite(x) & np.isfinite(err)
        if not keep.any():
            return
        u, _, cap = self._spec(key)
        nb, ne = self._dedupe(np.ascontiguousarray(x[keep]).view(u), err[keep])
        # bound the merge: this batch can contribute at most its own worst `cap`
        nb, ne = self._top(nb, ne, cap)

        b0, e0 = self._load(key)
        # A live measurement supersedes the stored error for the same input: a
        # kernel fix must be able to lower it. Later folds in one run take the
        # max instead, so several targets sharing a key keep the worst.
        drop = self._member(b0, nb) if self._stale(key) else np.zeros(b0.shape, bool)
        bits, errs = self._dedupe(
            np.concatenate([b0[~drop], nb]), np.concatenate([e0[~drop], ne])
        )
        bits, errs = self._top(bits, errs, cap)

        improved = errs > self._lookup(b0, e0, bits) + self._EPS
        n_new = int(improved.sum())
        self._book[key] = (bits, errs)
        # `or` short-circuits: allclose never sees a shape mismatch
        if not np.array_equal(bits, b0) or not np.allclose(
            errs, e0, rtol=0, atol=self._EPS
        ):
            self._dirty.add(key)
        if n_new:
            w = int(np.argmax(np.where(improved, errs, -np.inf)))
            d = self.discoveries.setdefault(
                key,
                {"n_new": 0, "base_max": self._base_max[key], "err": -np.inf},
            )
            d["n_new"] += n_new
            if errs[w] > d["err"]:
                d["err"], d["bits"] = float(errs[w]), int(bits[w])

    def refresh(self, key: str, x: np.ndarray, err: np.ndarray) -> None:
        """Re-stamp `key`'s stored errors from a live measurement of its inputs.

        Keyed, never slice-wide: two sources may hold the same input, and each
        book must record what its own replay measured. Without this the errors
        only ever ratchet up: a kernel fix would leave every book at its
        historical high-water mark, and `_top` would evict genuinely hard
        inputs in favour of ones that are no longer hard.
        """
        u, _, _ = self._spec(key)
        keep = np.isfinite(x) & np.isfinite(err)
        qb, qe = self._dedupe(
            np.ascontiguousarray(np.asarray(x)[keep]).view(u),
            np.asarray(err, dtype=np.float64)[keep],
        )
        bits, errs = self._load(key)
        measured = self._lookup(qb, qe, bits)
        hit = measured > -np.inf
        # first touch this run supersedes disk; later targets keep the worst
        new = np.where(
            hit if self._stale(key) else hit & (measured > errs), measured, errs
        )
        if np.allclose(new, errs, rtol=0, atol=self._EPS):
            return
        self._book[key] = (bits, new)
        self._dirty.add(key)
        was = self._base_max[key]
        self._base_max[key] = now = float(new.max(initial=0.0))
        self.restamped[key] = (self.restamped.get(key, (was, now))[0], now)

    def replay(self, key: str) -> np.ndarray:
        """`key`'s stored inputs as its own dtype, bits ascending."""
        _, f, _ = self._spec(key)
        return self._load(key)[0].view(f)

    def flush(self) -> list[str]:
        """Rewrite every changed book (worst first); return the keys written."""
        written = sorted(self._dirty)
        for key in written:
            bits, errs = self._book[key]
            order = np.argsort(-errs, kind="stable")
            rec = np.empty(order.size, dtype=self._row_dtype(key))
            rec["bits"], rec["err"] = bits[order], errs[order]
            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, rec, fmt="%d,%.6f")
        self._dirty.clear()
        return written

    def deltas(self) -> dict[str, tuple[bytes, bytes]]:
        """Raw (bits, err) buffers of every book this process changed."""
        return {
            key: (self._book[key][0].tobytes(), self._book[key][1].tobytes())
            for key in self._dirty
        }

    def absorb(
        self,
        deltas: dict[str, tuple[bytes, bytes]],
        restamped: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Fold another process's `deltas()` in; discoveries recount from disk."""
        for key, (was, now) in (restamped or {}).items():
            self._load(key)  # seeds _base_max from disk before it is superseded
            prev = self.restamped.get(key)
            self.restamped[key] = (
                (was, now) if prev is None else (prev[0], max(prev[1], now))
            )
            # a discovery on the controller ranks against the re-measured max
            self._base_max[key] = self.restamped[key][1]
        for key, (bits, errs) in deltas.items():
            u, f, _ = self._spec(key)
            self.fold(
                key,
                np.frombuffer(bits, dtype=u).view(f),
                np.frombuffer(errs, dtype=np.float64),
            )

    def report_lines(self) -> list[str]:
        """One line per book this run re-stamped or added a case to."""
        out = [
            f"{key}: re-measured, max {was:.4f} -> {now:.4f} ULP"
            for key, (was, now) in sorted(self.restamped.items())
        ]
        for key, d in sorted(self.discoveries.items()):
            u, f, _ = self._spec(key)
            x = float(np.array(d["bits"], u).view(f))
            base, now = d["base_max"], float(self._book[key][1].max(initial=0.0))
            # book max before -> after; the new case need not be the book's worst
            out.append(
                f"{key}: +{d['n_new']} case(s), max {base:.4f} -> {now:.4f} ULP "
                f"(worst new {d['err']:.4f} at x={x.hex()})"
            )
        return out
