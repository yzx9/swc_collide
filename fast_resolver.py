# pyright: reportInvalidTypeForm=false, reportArgumentType=false
import os
from typing import Optional

import numpy as np
import swcgeom
import warp as wp
from tqdm import tqdm

from helper import mask_neighborhood as _mask_neighborhood
from resampler import IsometricResampler

__all__ = ["resolve", "count"]

Vec3fArr = wp.array(dtype=wp.vec3f)


def resolve(
    fname: str,
    *,
    step: float = 0.1,
    iterates: int = 1000,
    verbose: bool = False,
    output: Optional[str] = None,
    **kwargs,
):
    t, N, scene = preprocess(fname, verbose=verbose, **kwargs)
    for _ in tqdm(range(iterates)) if verbose else range(iterates):
        direction = wp.zeros(N, dtype=wp.vec3f)
        wp.launch(resolve_compute, dim=(N, N - 1), inputs=[scene], outputs=[direction])
        wp.launch(resolve_move, dim=(N,), inputs=[direction, step], outputs=[scene.xyz])

    xyz = scene.xyz.numpy()
    for i, name in enumerate([t.names.x, t.names.y, t.names.z]):
        t.ndata[name] = xyz[..., i]

    if output is None:
        print(t.to_swc())
    elif os.path.isdir(output):
        t.to_swc(os.path.join(output, os.path.split(fname)[-1]))
    else:
        t.to_swc(output)


def count(
    fname: str, *, verbose: bool = False, output: Optional[str] = None, **kwargs
) -> int:
    t, N, scene = preprocess(fname, verbose=verbose, **kwargs)
    out = wp.zeros(N, dtype=wp.bool)
    wp.launch(_count, dim=(N, N - 1), inputs=[scene], outputs=[out])

    out = out.numpy()
    cnt = np.count_nonzero(out)
    if verbose and cnt:
        print(np.argwhere(out).flatten())

    if output is not None:
        t.ndata[t.names.type] = np.full_like(t.type(), fill_value=1)
        t.ndata[t.names.type][out] = 2
        t.to_swc(output)

    return cnt


@wp.struct
class Scene:
    xyz: Vec3fArr
    r: wp.array(dtype=wp.float32)
    mask: wp.array(ndim=2, dtype=wp.bool)


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

    xyz, r = t.xyz().copy(), t.r().copy()
    r += gap / 2  # add half gap to radius

    mask = _mask_neighborhood(t, mask_neighborhood)

    scene = Scene()
    scene.xyz = wp.from_numpy(xyz, dtype=wp.vec3f)
    scene.r = wp.from_numpy(r, dtype=wp.float32)
    scene.mask = wp.from_numpy(mask, dtype=wp.bool)
    return t, N, scene


@wp.kernel
def resolve_compute(scene: Scene, out: Vec3fArr):
    i, j = wp.tid()
    if j >= i:
        j = j + 1  # skip eye items

    direction = get_weighted_direction(scene, i, j)
    if wp.length(direction) > 0:
        wp.atomic_add(out, i, direction)  # type: ignore


@wp.kernel
def resolve_move(direction: Vec3fArr, step: wp.float32, xyz: Vec3fArr):
    i = wp.tid()
    vec = wp.normalize(direction[i])
    xyz[i] = xyz[i] + step * vec


@wp.kernel
def _count(scene: Scene, out: wp.array(dtype=wp.bool)):
    i, j = wp.tid()
    if j >= i:
        j = j + 1  # skip eye items

    direction = get_weighted_direction(scene, i, j)
    if wp.length(direction) > 0:
        out[i] = True


@wp.func
def get_weighted_direction(scene: Scene, i: int, j: int) -> wp.vec3f:
    if i == j or scene.mask[i][j]:
        return wp.vec3f(0.0, 0.0, 0.0)

    vec = scene.xyz[i] - scene.xyz[j]
    if wp.length(vec) >= scene.r[i] + scene.r[j]:
        return wp.vec3f(0.0, 0.0, 0.0)

    weight = scene.r[j] / (scene.r[i] + scene.r[j])
    return wp.normalize(vec) * weight


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    def add_common_argument(parser):
        parser.add_argument("fname", type=str)
        parser.add_argument("--gap", type=float, default=0)
        parser.add_argument("--radius", type=float, default=0)
        parser.add_argument("--no-resample", action="store_true")
        parser.add_argument("--mask_neighborhood", type=int, default=15)
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("-o", "--output", type=str, required=False)
        parser.add_argument("-v", "--verbose", action="store_true")

    def extract_common_args(args):
        return {
            "fname": args.fname,
            "gap": args.gap,
            "radius": args.radius,
            "resample": not args.no_resample,
            "mask_neighborhood": args.mask_neighborhood,
            "output": args.output,
            "verbose": args.verbose,
            # "device": args.device, # consumed
        }

    sub_resolve = subparsers.add_parser("resolve", help="resolve self-intersection")
    add_common_argument(sub_resolve)
    sub_resolve.add_argument("--step", type=float, default=0.1)
    sub_resolve.add_argument("--iterates", type=int, default=1000)

    sub_count = subparsers.add_parser("count", help="count self-intersection")
    add_common_argument(sub_count)

    args = parser.parse_args()
    wp.config.quiet = not args.verbose
    with wp.ScopedDevice(args.device):
        match args.command:
            case "resolve":
                resolve(
                    iterates=args.iterates, step=args.step, **extract_common_args(args)
                )

            case "count":
                cnt = count(**extract_common_args(args))
                print(f"Detected intersections at {cnt} nodes.")

            case _ as cmd:
                raise ValueError(f"unexpected command: {cmd}")
