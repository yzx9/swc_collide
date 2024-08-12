import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import swcgeom
from tqdm import tqdm

from resampler import BranchIsometricResampler, Resampler


def detect(t: swcgeom.Tree, *, mask_neighbor: int = 10) -> int:
    # assert all of the r is the same
    r_min = t.r().min()
    resampler = Resampler(BranchIsometricResampler(2 * r_min))

    t_new = resampler(t)
    N = t_new.number_of_nodes()

    xyz = np.array([n.xyz() for n in t_new])
    v = xyz.reshape(N, 1, 3) - xyz.reshape(1, N, 3)
    d = np.linalg.norm(v, axis=2)

    r = np.array([n.r for n in t_new])
    d_target = r.reshape(N, 1) + r.reshape(1, N)

    valid = np.triu(np.ones((N, N), dtype=np.bool_), k=1)
    # TODO following code would be very slow
    for n in t_new:
        s = [(n, 0)]
        s = [(c, 1) for c in n.children()] + [(n.parent(), 1)]
        while len(s):
            neighbor, era = s.pop()
            if neighbor is None or (
                not valid[n.id][neighbor.id] and not valid[neighbor.id][n.id]
            ):
                continue

            valid[n.id][neighbor.id] = False
            valid[neighbor.id][n.id] = False
            if era < mask_neighbor:
                s.extend((c, era + 1) for c in neighbor.children())
                if (p := neighbor.parent()) is not None:
                    s.append((p, era + 1))

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

        case "plot_trends" if os.path.isdir(args.fname):
            fnames = os.listdir(args.fname)

            def calc(fname: str, radius: float):
                t = swcgeom.Tree.from_swc(os.path.join(args.fname, fname))
                t.ndata[t.names.r] = np.full_like(t.r(), radius)
                return detect(t) > 0

            radii = np.arange(args.rmin, args.rmax, args.step)
            data = [
                sum(calc(fname, r) for fname in fnames)
                for r in (tqdm(radii) if args.verbose else radii)
            ]

            plt.figure()
            plt.plot(radii, data, marker="o")
            plt.xlabel("Radius")
            plt.ylabel("Number of files with self-intersections")
            plt.title("Number of files with self-intersections over radius")

            if args.output:
                plt.savefig(args.output)
            else:
                plt.show()

        case "plot_trends":
            t = swcgeom.Tree.from_swc(args.fname)
            radii = np.arange(args.rmin, args.rmax, args.step)
            data = []

            for r in tqdm(radii) if args.verbose else radii:
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
