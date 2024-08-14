import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import swcgeom
from tqdm import tqdm

from helper import mask_neighborhood
from resampler import IsometricResampler


def detect(t: swcgeom.Tree, *, mask_neighbor: int = 10) -> int:
    # assert all of the r is the same
    r_min = t.r().min()
    resampler = IsometricResampler(2 * r_min)

    t_resampled = resampler(t)
    N = t_resampled.number_of_nodes()

    xyz = np.array([n.xyz() for n in t_resampled])
    v = xyz.reshape(N, 1, 3) - xyz.reshape(1, N, 3)
    d = np.linalg.norm(v, axis=2)

    r = np.array([n.r for n in t_resampled])
    d_target = r.reshape(N, 1) + r.reshape(1, N)

    valid = np.triu(np.ones((N, N), dtype=np.bool_), k=1)
    mask_neighborhood(t_resampled, mask_neighbor, out=valid, mask_value=False)

    cnt = np.count_nonzero(np.less(d - d_target, 0, out=np.zeros_like(d), where=valid))
    return cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_detect = subparsers.add_parser("detect", help="detect self-intersection")
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
        case "detect":
            t = swcgeom.Tree.from_swc(args.fname)
            if args.radius > 0:
                t.ndata[t.names.r] = np.full_like(t.r(), args.radius)

            cnt = detect(t)
            print(f"detect {cnt} self-intersection")

        case "plot_trends":
            radii = np.arange(args.rmin, args.rmax, args.step)
            radii_it = tqdm(radii) if args.verbose else radii

            if os.path.isdir(args.fname):
                fnames = os.listdir(args.fname)

                def calc(fname: str, radius: float):
                    t = swcgeom.Tree.from_swc(os.path.join(args.fname, fname))
                    t.ndata[t.names.r] = np.full_like(t.r(), radius)
                    return detect(t) > 0

                data = [sum(calc(fname, r) for fname in fnames) for r in radii_it]

                plt.figure()
                plt.plot(radii, data, marker="o")
                plt.xlabel("Radius")
                plt.ylabel("Number of files with self-intersections")
                plt.title("Number of files with self-intersections over radius")
            else:
                t = swcgeom.Tree.from_swc(args.fname)
                data = []

                for r in radii_it:
                    t.ndata[t.names.r] = np.full_like(t.r(), r)
                    data.append(detect(t))

                plt.figure()
                plt.plot(radii, data, marker="o")
                plt.xlabel("Radius")
                plt.ylabel("Number of self-intersections")
                plt.title("Self-intersections over radius")

            if args.output:
                plt.savefig(args.output)
            else:
                plt.show()

        case _ as cmd:
            raise ValueError(f"unexpected command: {cmd}")
