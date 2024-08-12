import argparse
import os
from typing import Optional

import numpy as np
import numpy.typing as npt
import swcgeom

EPS = 1e-6


def revolve(
    xyz: npt.NDArray,
    r: npt.NDArray,
    *,
    gap: float = 0,
    it_max: int = 1000,
    verbose: int = 0,
):
    def log(level, *args):
        if verbose >= level:
            print(*args)

    log(2, "xyz:\n", xyz, "\nr:\n", r)

    N = xyz.shape[0]
    eye_mask = np.logical_not(np.identity(N, dtype=np.bool_))
    changed = []

    d_target = r.reshape(-1, 1) + r.reshape(1, -1) + gap  # (N, N)
    log(2, f"d_target:\n", d_target)

    v, d = None, None

    def calc_overlap():
        nonlocal v, d
        v = xyz.reshape(-1, 1, 3) - xyz  # (N, N, 3)
        d = np.linalg.norm(v, axis=2, keepdims=True)  # (N, N, 1)
        if np.equal(d, 0).any(where=eye_mask.reshape(N, N, 1)):
            raise ValueError("coincide in position")

        return np.maximum(d_target - d[..., 0], 0)  # (N, N)

    for i in range(it_max):
        log(1, f"it: {i}")

        overlap = calc_overlap()
        if np.less(overlap, EPS, where=eye_mask).all():
            log(1, "non-overlap")
            break

        v_norm = np.zeros_like(v)
        v_norm = np.divide(v, d, where=d != 0, out=v_norm)  # type:ignore
        v_ab = overlap.reshape(*overlap.shape, 1) * v_norm
        log(3, "overlap:\n", overlap, "\nv_overlap:\n", v_ab)

        moving = v_ab.sum(axis=1)
        xyz += moving
        log(3, "moving:\n", moving)
        log(2, "xyz:\n", xyz)
        if verbose >= 1:
            changes = np.where(moving.sum(axis=-1) != 0)[0]
            changed.extend(changes)
            print("changed: ", changes)
    else:
        log(1, "reach max iterates")

    if verbose >= 1:
        changed = list(set(changed))
        print("changed: ", changed)

    if verbose >= 3:
        print("distance: \n", d.squeeze())  # type:ignore

    return xyz


def main(fname: str, *, output: Optional[str] = None, **kwargs):
    t = swcgeom.Tree.from_swc(fname)
    new_xyz = revolve(t.xyz(), t.r(), **kwargs)
    for i, name in enumerate([t.names.x, t.names.y, t.names.z]):
        t.ndata[name] = new_xyz[..., i]

    if output is not None:
        dest = (
            output
            if not os.path.isdir(output)
            else os.path.join(output, os.path.split(fname)[-1])
        )
        t.to_swc(dest)
    else:
        print(t.to_swc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--gap", type=float, default=0)
    parser.add_argument("--it_max", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-vv", action="store_true")
    parser.add_argument("-vvv", action="store_true")

    args = parser.parse_args()
    verbose = 3 if args.vvv else 2 if args.vv else 1 if args.verbose else 0
    main(
        args.fname,
        output=args.output,
        gap=args.gap,
        it_max=args.it_max,
        verbose=verbose,
    )
