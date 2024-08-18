from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import swcgeom
from swcgeom.core import Branch, BranchTree, Node, Tree
from swcgeom.transforms import Transform
from swcgeom.transforms.branch import _BranchResampler


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
            source=x.source,
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


class Resampler(Transform[Tree, Tree]):
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


class IsometricResampler(Resampler):
    def __init__(
        self, distance: float, *, adjust_last_gap: bool = True, **kwargs
    ) -> None:
        branch_resampler = BranchIsometricResampler(
            distance, adjust_last_gap=adjust_last_gap, **kwargs
        )
        super().__init__(branch_resampler)
