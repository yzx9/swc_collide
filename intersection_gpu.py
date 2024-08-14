# type: ignore
import argparse

import numpy as np
import swcgeom
import warp as wp

from resampler import IsometricResampler


def count(fname: str, *, gap: float = 0) -> int:
    t = swcgeom.Tree.from_swc(fname)

    r_min = t.r().min()
    resampler = IsometricResampler(2 * r_min)

    t_resampled = resampler(t)
    N = t_resampled.number_of_nodes()

    r = np.array([n.r for n in t_resampled])
    d_target = r.reshape(N, 1) + r.reshape(1, N) + gap

    xyz = np.array([n.xyz() for n in t_resampled])
    weight = wp.ones((N, N), dtype=wp.float32)

    result = wp.zeros((N,), dtype=wp.uint32)
    wp.launch(step, dim=(N,), inputs=[xyz, weight, d_target], outputs=[result])
    return np.sum(result)


@wp.kernel
def step(
    points: wp.array(dtype=wp.vec3f),
    weight: wp.array(dtype=wp.float32, ndim=2),
    d_target: wp.array(dtype=wp.float32, ndim=2),
    out: wp.array(dtype=wp.uint32),
):
    pi = wp.tid()
    direction = get_direction(pi, points, weight, d_target)
    out[pi] = wp.uint32(wp.length(direction) > 0)


@wp.func
def get_direction(
    pi: int,
    points: wp.array(dtype=wp.vec3f),
    weight: wp.array(dtype=wp.float32, ndim=2),
    d_target: wp.array(dtype=wp.float32, ndim=2),
) -> wp.vec3f:
    vec = wp.vec3f(0.0, 0.0, 0.0)
    for j in range(points.shape[0]):
        if pi != j:
            dis = distance(points[j], points[pi])
            if dis < d_target[j][pi]:
                vec += (points[j] - points[pi]) * weight[pi][j]

    l = wp.length(vec)
    if l > 0:
        vec /= l

    return vec


@wp.func
def distance(pa: wp.vec3f, pb: wp.vec3f) -> wp.float32:
    return wp.length(pa - pb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_detect = subparsers.add_parser("count", help="detect self-intersection")
    parser_detect.add_argument("fname", type=str)
    parser_detect.add_argument("--radius", type=float, default=0)

    parser_plot_trends = subparsers.add_parser(
        "plot_trends", help="plot self-intersection trends of changes"
    )
    parser_plot_trends.add_argument("fname", type=str)
    parser_plot_trends.add_argument("--rmin", type=float, default=0.1)
    parser_plot_trends.add_argument("--rmax", type=float, default=10)
    parser_plot_trends.add_argument("--step", type=float, default=0.1)
    parser_plot_trends.add_argument("--output", type=str, required=False)
    parser_plot_trends.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    match args.command:
        case "count":
            cnt = count(args.fname)
            print(f"detect {cnt} self-intersection")
