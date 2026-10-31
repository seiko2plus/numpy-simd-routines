"""The frozen worst-case book: `WorstBook`, folded during a run."""

import pathlib
import warnings

import numpy as np
from numpy_sr import utils

# rows kept per book; f32 is also swept exhaustively
_CAP = {"float32": 1_000, "float64": 10_000}
_HEADER = "# maxulp="


def _names(dt: np.dtype) -> tuple[str, ...]:
    """`dt`'s field names; every dtype here is structured."""
    return dt.names or ()


def _pack(dt: np.dtype, cols: list) -> np.ndarray:
    """One `dt` record per row, `cols` in the order of its fields.

    Structured arrays sort and compare field by field, so the helpers below
    work the same for any number of inputs."""
    out = np.empty(len(cols[0]), dt)
    for name, col in zip(_names(dt), cols, strict=True):
        out[name] = col
    return out


def _order(bits: np.ndarray, *last: np.ndarray) -> np.ndarray:
    """Stable ascending order of `bits`, first field first, then `last`.

    A plain argsort of a structured array compares element by element, ~4x
    slower on the exhaustive f32 sweeps."""
    return np.lexsort([*last, *(bits[n] for n in reversed(_names(bits.dtype)))])


def _top(cap: int, bits: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The `cap` worst rows, bits ascending (the `_lookup` invariant)."""
    if bits.size > cap:
        keep = np.argpartition(-vals["e"], cap)[:cap]
        bits, vals = bits[keep], vals[keep]
    order = _order(bits)
    return bits[order], vals[order]


def _lookup(bits0: np.ndarray, err0: np.ndarray, query: np.ndarray) -> np.ndarray:
    """`err0` for each `query` bit in the ascending `bits0`, else -inf."""
    if bits0.size == 0:
        return np.full(query.shape, -np.inf)
    pos = np.clip(np.searchsorted(bits0, query), 0, bits0.size - 1)
    return np.where(bits0[pos] == query, err0[pos], -np.inf)


def _dedupe(bits: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The worst row per input, bits ascending."""
    if not bits.size:
        return bits, vals
    order = _order(bits, -vals["e"])  # the worst first within an input
    bits, vals = bits[order], vals[order]
    first = np.flatnonzero(np.concatenate([[True], bits[1:] != bits[:-1]]))
    return bits[first], vals[first]


def _measured(
    xdt: np.dtype, vdt: np.dtype, xs: tuple, ref: tuple, err
) -> tuple[np.ndarray, np.ndarray]:
    """Rows with finite inputs and error, as deduped (bits, vals)."""
    xa, err = [np.asarray(x) for x in xs], np.asarray(err, dtype=np.float64)
    keep = np.isfinite(err)
    for x in xa:
        keep &= np.isfinite(x)
    bits = _pack(xdt, [utils.bits(x[keep]) for x in xa])
    r, res = ref
    vals = _pack(
        vdt, [utils.bits(np.asarray(r)[keep]), np.asarray(res)[keep], err[keep]]
    )
    return _dedupe(bits, vals)


class WorstBook:
    """A capped CSV per (op, accuracy, dtype, fma, source) under `root`.

    A `# maxulp=<bound>` line, then `<bits_0>,...,<bits_n-1>,<ref>,<res>,<err>`
    rows: one bits column per input of the op (`arity`), the reference's bits,
    its residual and the |ulp| error. Bits are decimal, the only exact form
    np.loadtxt parses in C; floats are `repr`, which round-trips."""

    def __init__(self, root: pathlib.Path, arity: dict[str, int]) -> None:
        self.root, self.arity = root, arity
        # key -> (uint bits ascending, (ref, res, err)): as committed, and live
        self._disk: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._book: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # key -> maxulp: as committed, and the highest folded this run
        self._disk_bound: dict[str, float | None] = {}
        self._bound: dict[str, float] = {}

    @staticmethod
    def key(op: str, accuracy: str, dtype: str, have_fma: bool, source: str) -> str:
        # `parse` splits on "_", so `op` must not contain one.
        fma = "fma" if have_fma else "nonfma"
        return f"{op}_{accuracy.lower()}_{dtype}_{fma}_{source}"

    @staticmethod
    def parse(key: str) -> tuple[str, str, str, bool, str]:
        """`key()` inverted: `(operation, accuracy, dtype, have_fma, source)`."""
        op, accuracy, dtype, fma, source = key.split("_", 4)
        return op, accuracy, dtype, fma == "fma", source

    def all_keys(self) -> list[str]:
        """Every book on disk, sorted; the gate parametrizes over these."""
        return sorted(f"{p.parent.name}_{p.stem}" for p in self.root.glob("*/*.csv"))

    def _dtypes(self, key: str) -> tuple[np.dtype, np.dtype]:
        """`key`'s (inputs, values): one uint field per argument of its op; the
        reference's bits, its residual and the error."""
        op, _, dtype, _, _ = self.parse(key)
        uint = utils.bits_dtype(dtype)
        return (
            np.dtype([(f"x{i}", uint) for i in range(self.arity[op])]),
            np.dtype([("r", uint), ("res", "f8"), ("e", "f8")]),
        )

    def _cap(self, key: str) -> int:
        return _CAP[self.parse(key)[2]]

    def _floats(self, key: str, bits: np.ndarray) -> tuple:
        """`bits` back to one float array per input."""
        return tuple(
            utils.from_bits(np.ascontiguousarray(bits[n])) for n in _names(bits.dtype)
        )

    def _path(self, key: str) -> pathlib.Path:
        op, rest = key.split("_", 1)  # the op names its dir
        return self.root / op / f"{rest}.csv"

    def _load(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if key in self._book:
            return self._book[key]
        (xdt, vdt), path = self._dtypes(key), self._path(key)
        pair = (np.array([], xdt), np.array([], vdt))
        self._disk_bound[key] = None
        if path.exists():
            with path.open() as f:
                head = f.readline()
            if head.startswith(_HEADER):
                self._disk_bound[key] = float(head[len(_HEADER) :])
            with warnings.catch_warnings():  # an empty file is legitimate
                warnings.simplefilter("ignore")
                rec = np.loadtxt(
                    path, dtype=[*xdt.descr, *vdt.descr], delimiter=",", ndmin=1
                )
            pair = _top(
                self._cap(key),
                _pack(xdt, [rec[n] for n in _names(xdt)]),
                _pack(vdt, [rec[n] for n in _names(vdt)]),
            )
        self._disk[key] = self._book[key] = pair
        return pair

    def _stale(self, key: str) -> bool:
        """`key` still holds disk errors, not this run's measurements."""
        return self._book[key] is self._disk[key]

    def _changed(self) -> list[str]:
        """Keys whose rows or maxulp differ from disk."""
        out = []
        for key in sorted(self._book):
            (b0, v0), (bits, vals) = self._disk[key], self._book[key]
            if (
                not np.array_equal(bits, b0)
                or not np.array_equal(vals, v0)
                or self.bound(key) != self._disk_bound[key]
            ):
                out.append(key)
        return out

    def bound(self, key: str) -> float | None:
        """`key`'s maxulp: folded this run, else as committed."""
        self._load(key)
        return self._bound.get(key, self._disk_bound[key])

    def rows(self, key: str, xs: tuple, ref: tuple, err: np.ndarray) -> tuple:
        """Measured (inputs, (ref, res), |ulp|) as bytes for `fold`, deduped and
        capped; inputs and ref in `key`'s dtype."""
        bits, vals = _top(self._cap(key), *_measured(*self._dtypes(key), xs, ref, err))
        return bits.tobytes(), vals.tobytes()

    def fold(self, key: str, rows: tuple, maxulp: float | None) -> None:
        """Merge `rows()` into `key`'s book; a replay passes no `maxulp`."""
        if maxulp is not None:
            # the first fold this run supersedes disk; the highest target wins
            self._bound[key] = max(self._bound.get(key, maxulp), maxulp)
        xdt, vdt = self._dtypes(key)
        nb, nv = np.frombuffer(rows[0], xdt), np.frombuffer(rows[1], vdt)
        if not nb.size:
            return
        b0, v0 = self._load(key)
        # A live measurement supersedes the stored row for the same input: a
        # kernel fix must be able to lower it. Later folds in one run take the
        # max instead, so several targets sharing a key keep the worst.
        hit = _lookup(nb, nv["e"], b0) > -np.inf
        keep = ~hit if self._stale(key) else slice(None)
        self._book[key] = _top(
            self._cap(key),
            *_dedupe(np.concatenate([b0[keep], nb]), np.concatenate([v0[keep], nv])),
        )

    def replay(self, key: str) -> tuple:
        """`key`'s stored `(inputs, (ref, res))`; inputs and ref in its dtype."""
        bits, vals = self._load(key)
        ref = utils.from_bits(np.ascontiguousarray(vals["r"]))
        return self._floats(key, bits), (ref, np.ascontiguousarray(vals["res"]))

    def flush(self) -> None:
        """Rewrite every changed book, worst first."""
        for key in self._changed():
            bits, vals = self._book[key]
            order = np.argsort(-vals["e"], kind="stable")
            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = zip(bits[order].tolist(), vals[order].tolist(), strict=True)
            path.write_text(
                f"{_HEADER}{self.bound(key)!r}\n"
                + "".join(
                    f"{','.join(map(str, b))},{r},{res!r},{e!r}\n"
                    for b, (r, res, e) in rows
                )
            )

    def report_lines(self) -> list[str]:
        """One line per changed book: max before -> after, and its new inputs."""
        out = []
        for key in self._changed():
            (b0, v0), (bits, vals) = self._disk[key], self._book[key]
            e0, errs = v0["e"], vals["e"]
            line = (
                f"{key}: max {e0.max(initial=0.0):.4f} -> "
                f"{errs.max(initial=0.0):.4f} ULP"
            )
            new = _lookup(b0, e0, bits) == -np.inf
            if new.any():
                w = int(np.argmax(np.where(new, errs, -np.inf)))
                x = ", ".join(
                    float(c[0]).hex() for c in self._floats(key, bits[w : w + 1])
                )
                line += (
                    f", +{int(new.sum())} new input(s) (worst {errs[w]:.4f} at x={x})"
                )
            out.append(line)
        return out
