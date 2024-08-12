import argparse
import os
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import swcgeom
from swcgeom.core import Branch, BranchTree, Node, Tree
from swcgeom.transforms import Transform
from swcgeom.transforms.branch import _BranchResampler
from tqdm import tqdm


def detect(t: swcgeom.Tree) -> int:
    # assert all of the r is the same
    r_min = t.r().min()
    resampler = TreeResampler(BranchIsometricResampler(2 * r_min))

    t_new = resampler(t)
    N = t_new.number_of_nodes()

    xyz = np.array([n.xyz() for n in t_new])
    v = xyz.reshape(N, 1, 3) - xyz.reshape(1, N, 3)
    d = np.linalg.norm(v, axis=2)

    r = np.array([n.r for n in t_new])
    d_target = r.reshape(N, 1) + r.reshape(1, N)

    valid = np.triu(np.ones((N, N), dtype=np.bool_), k=1)
    cnt = np.count_nonzero(np.less(d - d_target, 0, out=np.zeros_like(d), where=valid))
    return cnt


class BranchIsometricResampler(_BranchResampler):
    def __init__(self, distance: float, *, adjust_last_gap: bool = True) -> None:
        super().__init__()
        self.distance = distance
        self.adjust_last_gap = adjust_last_gap

    def resample(self, xyzr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Resampling by isometric interpolation, DO NOT keep original node.

        Parameters
        ----------
        xyzr : np.ndarray[np.float32]
            The array of shape (N, 4).

        Returns
        -------
        new_xyzr : ~numpy.NDArray[float32]
            An array of shape (n_nodes, 4).
        """

        # Compute the cumulative distances between consecutive points
        diffs = np.diff(xyzr[:, :3], axis=0)
        distances = np.sqrt((diffs**2).sum(axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])

        total_length = cumulative_distances[-1]
        n_nodes = int(np.ceil(total_length / self.distance)) + 1

        # Determine the new distances
        if self.adjust_last_gap and n_nodes > 1:
            new_distances = np.linspace(0, total_length, n_nodes)
        else:
            new_distances = np.arange(0, total_length, self.distance)
            # keep endpoint
            new_distances = np.concatenate([new_distances, total_length])

        # Interpolate the new points
        new_xyzr = np.zeros((n_nodes, 4), dtype=np.float32)
        new_xyzr[:, :3] = np.array(
            [
                np.interp(new_distances, cumulative_distances, xyzr[:, i])
                for i in range(3)
            ]
        ).T
        new_xyzr[:, 3] = np.interp(new_distances, cumulative_distances, xyzr[:, 3])
        return new_xyzr

    def extra_repr(self) -> str:
        return f"distance={self.distance},adjust_last_gap={self.adjust_last_gap}"


class BranchTreeAssembler(Transform[BranchTree, Tree]):
    EPS = 1e-6

    def __call__(self, x: BranchTree) -> Tree:
        nodes = [x.soma().detach()]
        stack = [(x.soma(), 0)]  # n_orig, id_new
        while len(stack):
            n_orig, pid_new = stack.pop()
            children = n_orig.children()

            for br, c in self.pair(x.branches.get(n_orig.id, []), children):
                s = 1 if np.linalg.norm(br[0].xyz() - n_orig.xyz()) < self.EPS else 0
                e = -2 if np.linalg.norm(br[-1].xyz() - c.xyz()) < self.EPS else -1

                br_nodes = [n.detach() for n in br[s:e]] + [c.detach()]
                for i, n in enumerate(br_nodes):
                    # reindex
                    n.id = len(nodes) + i
                    n.pid = len(nodes) + i - 1

                br_nodes[0].pid = pid_new
                nodes.extend(br_nodes)
                stack.append((c, br_nodes[-1].id))

        return swcgeom.Tree(
            len(nodes),
            source=f"Assemble from Branch Tree {x.source}",
            comments=x.comments,
            names=x.names,
            **{
                k: np.array([n.__getattribute__(k) for n in nodes])
                for k in x.names.cols()
            },
        )

    def pair(
        self, branches: list[Branch], endpoints: list[Node]
    ) -> Iterable[tuple[Branch, Node]]:
        assert len(branches) == len(endpoints)
        xyz1 = [br[-1].xyz() for br in branches]
        xyz2 = [n.xyz() for n in endpoints]
        v = np.reshape(xyz1, (-1, 1, 3)) - np.reshape(xyz2, (1, -1, 3))
        dis = np.linalg.norm(v, axis=-1)

        # greedy algorithm
        pairs = []
        for _ in range(len(branches)):
            # find minimal
            min_idx = np.argmin(dis)
            min_branch_idx, min_endpoint_idx = np.unravel_index(min_idx, dis.shape)
            pairs.append((branches[min_branch_idx], endpoints[min_endpoint_idx]))

            # remove current node
            dis[min_branch_idx, :] = np.inf
            dis[:, min_endpoint_idx] = np.inf

        return pairs


class TreeResampler(Transform[Tree, Tree]):
    def __init__(self, branch_resampler: Transform[Branch, Branch]) -> None:
        super().__init__()
        self.resampler = branch_resampler
        self.assembler = BranchTreeAssembler()

    def __call__(self, x: Tree) -> Tree:
        t = BranchTree.from_tree(x)
        t.branches = {
            k: [self.resampler(br) for br in brs] for k, brs in t.branches.items()
        }
        return self.assembler(t)


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
