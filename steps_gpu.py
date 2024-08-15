# type: ignore
import argparse
from typing import Optional

import numpy.typing as npt
import swcgeom
import warp as wp
from tqdm import tqdm

from resampler import IsometricResampler


def main(
    fname: str,
    *,
    output: Optional[str] = None,
    resample: bool = False,
    verbose: bool = False,
    **kwargs,
):
    t = swcgeom.Tree.from_swc(fname)
    if resample:
        r_min = t.r().min()
        resampler = IsometricResampler(2 * r_min)
        t = resampler(t)

    if verbose:
        print(f"n_nodes: {t.number_of_nodes()}")

    new_xyz = run(t.xyz(), t.r(), verbose=verbose, **kwargs)
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


def run(
    xyz: npt.NDArray,
    r: npt.NDArray,
    *,
    step: float = 0.01,
    gap: float = 0,
    iterates: int = 10000,
    device: str = "cpu",
    verbose: bool = False,
) -> int:
    d_target = r.reshape(-1, 1) + r.reshape(1, -1) + gap
    weight = wp.ones(d_target.shape, dtype=wp.float32, device=device)

    xyz = wp.from_numpy(xyz, dtype=wp.vec3f, device=device)
    d_target = wp.from_numpy(d_target, dtype=wp.float32, device=device)

    for _ in tqdm(range(iterates)) if verbose else range(iterates):
        out = wp.zeros(r.shape[0], dtype=wp.vec3f, device=device)
        wp.launch(
            run_step,
            dim=r.shape,
            inputs=[xyz, weight, d_target, step],
            outputs=[out],
            device=device,
        )
        xyz = out

    return xyz.numpy()


@wp.kernel
def run_step(
    points: wp.array(dtype=wp.vec3f),
    weight: wp.array(dtype=wp.float32, ndim=2),
    d_target: wp.array(dtype=wp.float32, ndim=2),
    step: wp.float32,
    out: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    direction = get_direction(points, i, weight, d_target)
    out[i] = points[i] + step * direction


@wp.func
def get_direction(
    points: wp.array(dtype=wp.vec3f),
    i: int,
    weight: wp.array(dtype=wp.float32, ndim=2),
    d_target: wp.array(dtype=wp.float32, ndim=2),
) -> wp.vec3f:
    vec = wp.vec3f(0.0, 0.0, 0.0)
    for j in range(points.shape[0]):
        if i != j:
            dis = distance(points[j], points[i])
            if dis < d_target[j][i]:
                vec += (points[i] - points[j]) * weight[i][j]

    l = wp.length(vec)
    if l > 0:
        vec /= l

    return vec


@wp.func
def distance(pa: wp.vec3f, pb: wp.vec3f) -> wp.float32:
    return wp.length(pa - pb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--gap", type=float, default=0)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--iterates", type=int, default=10000)
    parser.add_argument("--no-resample", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    main(
        args.fname,
        output=args.output,
        gap=args.gap,
        iterates=args.iterates,
        step=args.step,
        resample=not args.no_resample,
        device=args.device,
        verbose=args.verbose,
    )
