# pyright: reportInvalidTypeForm=false, reportArgumentType=false
import argparse
import os
from typing import Optional

import numpy as np
import swcgeom
import warp as wp
from tqdm import tqdm

from resampler import IsometricResampler


def resolve(
    fname: str,
    *,
    output: Optional[str] = None,
    step: float = 0.01,
    iterates: int = 10000,
    device: str = "cpu",
    verbose: bool = False,
    **kwargs,
):
    t, xyz, r = preprocess(fname, device=device, verbose=verbose, **kwargs)
    for _ in tqdm(range(iterates)) if verbose else range(iterates):
        out = wp.zeros(r.shape[0], dtype=wp.vec3f, device=device)
        wp.launch(
            resolve_kernel,
            dim=r.shape,
            inputs=[xyz, r, step],
            outputs=[out],
            device=device,
        )
        xyz = out

    xyz = xyz.numpy()
    for i, name in enumerate([t.names.x, t.names.y, t.names.z]):
        t.ndata[name] = xyz[..., i]

    if output is None:
        print(t.to_swc())
    elif os.path.isdir(output):
        t.to_swc(os.path.join(output, os.path.split(fname)[-1]))
    else:
        t.to_swc(output)


def count(fname, *, device: str = "cpu", **kwargs) -> int:
    t, _, _ = preprocess(fname, device=device, **kwargs)
    out = wp.zeros(t.number_of_nodes(), dtype=wp.bool, device=device)
    return np.count_nonzero(out.numpy())


def preprocess(
    fname: str,
    *,
    resample: bool = False,
    gap: float = 0,
    device: str = "cpu",
    verbose: bool = False,
):
    t = swcgeom.Tree.from_swc(fname)
    if resample:
        r_min = t.r().min()
        resampler = IsometricResampler(2 * r_min)
        t = resampler(t)

    if verbose:
        print(f"n_nodes: {t.number_of_nodes()}")

    xyz, r = t.xyz(), t.r()
    r += gap / 2  # add half gap to radius

    xyz = wp.from_numpy(xyz, dtype=wp.vec3f, device=device)
    r = wp.from_numpy(r, dtype=wp.float32, device=device)
    return t, xyz, r


@wp.kernel
def resolve_kernel(
    xyz: wp.array(dtype=wp.vec3f),
    r: wp.array(dtype=wp.float32),
    step: wp.float32,
    out: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    direction = get_weighted_direction(xyz, r, i)
    out[i] = xyz[i] + step * direction


@wp.kernel
def count_kernel(
    xyz: wp.array(dtype=wp.vec3f),
    r: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    direction = get_weighted_direction(xyz, r, i)
    if wp.length(direction) > 0:
        out[i] = True


@wp.func
def get_weighted_direction(
    xyz: wp.array(dtype=wp.vec3f),
    r: wp.array(dtype=wp.float32),
    i: int,
) -> wp.vec3f:
    vec = wp.vec3f(0.0, 0.0, 0.0)
    for j in range(xyz.shape[0]):
        if i != j:
            dis = wp.length(xyz[j] - xyz[i])
            if dis < r[i] + r[j]:
                weight = r[i] / (r[i] + r[j])
                vec += (xyz[i] - xyz[j]) * weight

    l = wp.length(vec)
    if l > 0:
        vec /= l

    return vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    sub_resolve = subparsers.add_parser("resolve", help="resolve self-intersection")
    sub_resolve.add_argument("fname", type=str)
    sub_resolve.add_argument("--output", type=str)
    sub_resolve.add_argument("--gap", type=float, default=0)
    sub_resolve.add_argument("--step", type=float, default=0.01)
    sub_resolve.add_argument("--iterates", type=int, default=10000)
    sub_resolve.add_argument("--no-resample", action="store_true")
    sub_resolve.add_argument("--device", type=str, default="cpu")
    sub_resolve.add_argument("-v", "--verbose", action="store_true")

    sub_count = subparsers.add_parser("count", help="count self-intersection")
    sub_count.add_argument("fname", type=str)
    sub_count.add_argument("--gap", type=float, default=0)
    sub_count.add_argument("--no-resample", action="store_true")
    sub_count.add_argument("--device", type=str, default="cpu")
    sub_count.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    match args.command:
        case "resolve":
            resolve(
                args.fname,
                output=args.output,
                gap=args.gap,
                iterates=args.iterates,
                step=args.step,
                resample=not args.no_resample,
                device=args.device,
                verbose=args.verbose,
            )

        case "detect":
            cnt = count(
                args.fname,
                gap=args.gap,
                resample=not args.no_resample,
                device=args.device,
                verbose=args.verbose,
            )
            print(f"detect {cnt} self-intersection")

        case _ as cmd:
            raise ValueError(f"unexpected command: {cmd}")
