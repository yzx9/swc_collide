# pyright: reportInvalidTypeForm=false, reportArgumentType=false
import argparse
import os
from typing import Optional

import numpy as np
import swcgeom
import warp as wp
from tqdm import tqdm

from helper import mask_neighborhood as _mask_neighborhood
from resampler import IsometricResampler

__all__ = ["resolve", "count"]


def resolve(
    fname: str,
    *,
    output: Optional[str] = None,
    step: float = 0.1,
    iterates: int = 1000,
    device: str = "cpu",
    verbose: bool = False,
    **kwargs,
):
    with wp.ScopedDevice(device):
        t, N, xyz, r, mask = preprocess(fname, verbose=verbose, **kwargs)
        for _ in tqdm(range(iterates)) if verbose else range(iterates):
            direction = wp.zeros(N, dtype=wp.vec3f)
            wp.launch(
                resolve_compute,
                dim=(N, N - 1),
                inputs=[xyz, r, mask],
                outputs=[direction],
            )
            wp.launch(resolve_move, dim=(N,), inputs=[direction, step], outputs=[xyz])

    xyz = xyz.numpy()
    for i, name in enumerate([t.names.x, t.names.y, t.names.z]):
        t.ndata[name] = xyz[..., i]

    if output is None:
        print(t.to_swc())
    elif os.path.isdir(output):
        t.to_swc(os.path.join(output, os.path.split(fname)[-1]))
    else:
        t.to_swc(output)


def count(fname: str, *, device: str = "cpu", verbose: bool = False, **kwargs) -> int:
    with wp.ScopedDevice(device):
        _, N, xyz, r, mask = preprocess(fname, verbose=verbose, **kwargs)
        out = wp.zeros(N, dtype=wp.bool)
        wp.launch(_count, dim=(N, N - 1), inputs=[xyz, r, mask], outputs=[out])

    out = out.numpy()
    cnt = np.count_nonzero(out)
    if verbose and cnt:
        print(np.argwhere(out).flatten())

    return cnt


def preprocess(
    fname: str,
    *,
    radius: float = 0,
    resample: bool = False,
    gap: float = 0,
    mask_neighborhood: int = 15,
    verbose: bool = False,
):
    t = swcgeom.Tree.from_swc(fname)
    if radius > 0:
        t.ndata[t.names.r] = np.full_like(t.r(), fill_value=radius)

    if resample:
        r_min = t.r().min()
        resampler = IsometricResampler(2 * r_min)
        t = resampler(t)

    N = t.number_of_nodes()
    if verbose:
        print(f"n_nodes: {N}")

    xyz, r = t.xyz(), t.r()
    r += gap / 2  # add half gap to radius

    mask = _mask_neighborhood(t, mask_neighborhood)

    xyz = wp.from_numpy(xyz, dtype=wp.vec3f)
    r = wp.from_numpy(r, dtype=wp.float32)
    mask = wp.from_numpy(mask, dtype=wp.bool)
    return t, N, xyz, r, mask


@wp.kernel
def resolve_compute(
    xyz: wp.array(dtype=wp.vec3f),
    r: wp.array(dtype=wp.float32),
    mask: wp.array(ndim=2, dtype=wp.bool),
    out: wp.array(dtype=wp.vec3f),
):
    i, j = wp.tid()
    if j >= i:
        j = j + 1  # skip eye items
    direction = get_weighted_direction(xyz, r, mask, i, j)
    if wp.length(direction) > 0:
        wp.atomic_add(out, i, direction)  # type: ignore


@wp.kernel
def resolve_move(
    direction: wp.array(dtype=wp.vec3f),
    step: wp.float32,
    xyz: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    vec = wp.normalize(direction[i])
    xyz[i] = xyz[i] + step * vec


@wp.kernel
def _count(
    xyz: wp.array(dtype=wp.vec3f),
    r: wp.array(dtype=wp.float32),
    mask: wp.array(ndim=2, dtype=wp.bool),
    out: wp.array(dtype=wp.bool),
):
    i, j = wp.tid()
    if j >= i:
        j = j + 1  # skip eye items
    direction = get_weighted_direction(xyz, r, mask, i, j)
    if wp.length(direction) > 0:
        out[i] = True


@wp.func
def get_weighted_direction(
    xyz: wp.array(dtype=wp.vec3f),
    r: wp.array(dtype=wp.float32),
    mask: wp.array(ndim=2, dtype=wp.bool),
    i: int,
    j: int,
) -> wp.vec3f:
    if i == j or mask[i][j]:
        return wp.vec3f(0.0, 0.0, 0.0)

    vec = xyz[j] - xyz[i]
    dis = wp.length(vec)
    if dis >= r[i] + r[j]:
        return wp.vec3f(0.0, 0.0, 0.0)

    weight = r[j] / (r[i] + r[j])
    return wp.normalize(vec) * weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    def add_common_argument(parser):
        parser.add_argument("fname", type=str)
        parser.add_argument("--gap", type=float, default=0)
        parser.add_argument("--radius", type=float, default=0)
        parser.add_argument("--no-resample", action="store_true")
        parser.add_argument("--mask_neighborhood", type=int, default=15)
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("-v", "--verbose", action="store_true")

    def extract_common_args(args):
        return {
            "fname": args.fname,
            "gap": args.gap,
            "radius": args.radius,
            "resample": not args.no_resample,
            "mask_neighborhood": args.mask_neighborhood,
            "device": args.device,
            "verbose": args.verbose,
        }

    sub_resolve = subparsers.add_parser("resolve", help="resolve self-intersection")
    add_common_argument(sub_resolve)
    sub_resolve.add_argument("--output", type=str)
    sub_resolve.add_argument("--step", type=float, default=0.1)
    sub_resolve.add_argument("--iterates", type=int, default=1000)

    sub_count = subparsers.add_parser("count", help="count self-intersection")
    add_common_argument(sub_count)

    args = parser.parse_args()
    match args.command:
        case "resolve":
            resolve(
                output=args.output,
                iterates=args.iterates,
                step=args.step,
                **extract_common_args(args),
            )

        case "count":
            cnt = count(**extract_common_args(args))
            print(f"Detected intersections at {cnt} nodes.")

        case _ as cmd:
            raise ValueError(f"unexpected command: {cmd}")
