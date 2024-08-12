from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import swcgeom


def mask_neighborhood(
    t: swcgeom.Tree,
    rank: int = 5,
    *,
    out: Optional[npt.NDArray] = None,
    fill_value: Any = False,
    mask_value: Any = True,
) -> npt.NDArray:
    N = t.number_of_nodes()
    if out is None:
        dtype = np.bool_ if isinstance(fill_value, bool) else np.float32
        out = np.full((N, N), fill_value=fill_value, dtype=dtype)
    else:
        assert out.shape == (N, N)

    for n in t:
        s = [(c, 1) for c in n.children()]  # node, rank
        if (p := n.parent()) is not None:
            s.append((p, 1))

        visited = set([n.id] + [nn.id for nn, _ in s])
        while len(s):
            neighbor, r = s.pop()
            if neighbor is None:
                continue

            visited.add(neighbor.id)
            out[n.id][neighbor.id] = mask_value
            out[neighbor.id][n.id] = mask_value
            if r < rank:
                for c in neighbor.children():  # TODO this is vary slow
                    if c.id not in visited:
                        s.append((c, r + 1))
                        visited.add(c.id)

                if (p := neighbor.parent()) is not None and p.id not in visited:
                    s.append((p, r + 1))
                    visited.add(p.id)

    return out
