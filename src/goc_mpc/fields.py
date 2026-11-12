"""Per-agent `ntfields` fields for goc-mpc's solvers.

An `AgentField` is an `ntfields` `Field` (e.g. a `TrainedField`) for one
agent, with the map from the agent's row into the field's coordinates.
`TracedTimingMPC` plans each agent's segments through its field, and
`GCSWaypointMPC`/`SCIPWaypointMPC` price each agent's arcs by the field's
travel time (`fields=`), through `FieldTravelTimeModel`.

`ntfields` is an optional dependency (the `ntfields` extra): torch is imported
only when a field is queried.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._ext.configuration_spline import BlockType
from ._ext.goc_mpc import ArcTravelTimeModel, WaypointObjective

#: The waypoint objectives that price arcs by travel time.
TRAVEL_TIME_OBJECTIVES = (WaypointObjective.kMinMaxTravelTime, WaypointObjective.kAvgTravelTime)


@dataclasses.dataclass(frozen=True)
class AgentField:
    """An `ntfields` `Field` for one agent, and the map from the agent's row
    into the field's coordinates: `field_q = (q[:dim] - offset) / scale`, with
    `dim = len(offset)`. The agent's trailing columns are not part of the
    field."""

    field: Any
    offset: Sequence[float]
    scale: float = 1.0


class FieldTravelTimeModel(ArcTravelTimeModel):
    """An `AgentField` as an `ArcTravelTimeModel`.

    The domain is the field's box in the agent's coordinates, and times are in
    the agent's units (the field's time times `scale`), so the gradients with
    respect to the agent's coordinates are the field's own. `max_travel_time`
    is `headroom` times the largest time over `n_samples` random pairs in the
    domain.

    Arcs reach the field in batches of exactly `batch_size` pairs (padded,
    or split), so a field compiled per batch size (`ntfields`' "compile"
    backend) compiles once, here at construction, not once per arc count
    the solves happen to produce."""

    def __init__(self, agent_field: AgentField, *, n_samples: int = 4096, headroom: float = 2.0, seed: int = 0,
                 batch_size: int = 64):
        import torch

        field = agent_field.field
        offset = np.asarray(agent_field.offset, dtype=float)
        scale = float(agent_field.scale)
        bounds = field.problem.env.bounds.detach().cpu().double()  # (dim, 2), field coordinates
        g = torch.Generator().manual_seed(seed)
        pairs = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.rand(
            n_samples, 2, offset.shape[0], generator=g, dtype=torch.float64)
        pairs = pairs.to(device=field.device, dtype=torch.float32)
        t_max = float(field.travel_time(pairs[:, 0], pairs[:, 1]).max()) * scale
        super().__init__(bounds[:, 0].numpy() * scale + offset, bounds[:, 1].numpy() * scale + offset,
                         headroom * t_max)
        self._field, self._offset, self._scale, self._batch_size = field, offset, scale, batch_size
        mid = (self.lower + self.upper) / 2.0
        self.linearize(mid[None], mid[None])

    def linearize(self, X, Y):
        import torch

        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        n, b = X.shape[0], self._batch_size
        if n == 0:
            return np.zeros(0), np.zeros((0, self.dim())), np.zeros((0, self.dim()))
        n_pad = -(-n // b) * b
        rows = np.minimum(np.arange(n_pad), n - 1)  # padding repeats the last pair
        device = self._field.device
        to_field = lambda q: torch.as_tensor((q[rows] - self._offset) / self._scale,
                                             dtype=torch.float32, device=device)
        q_s, q_g = to_field(X), to_field(Y)
        t, g_s, g_g = [], [], []
        for k in range(0, n_pad, b):
            t.append(self._field.travel_time(q_s[k:k + b], q_g[k:k + b]))
            gs, gg = self._field.gradient(q_s[k:k + b], q_g[k:k + b])
            g_s.append(gs)
            g_g.append(gg)
        as_np = lambda parts: torch.cat(parts)[:n].detach().cpu().double().numpy()
        return as_np(t) * self._scale, as_np(g_s), as_np(g_g)


class L2TravelTimeModel(ArcTravelTimeModel):
    """`T(x, y) = |y - x|` on the box `[lower, upper]`."""

    def __init__(self, lower, upper):
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        super().__init__(lower, upper, float(np.linalg.norm(upper - lower)))

    def linearize(self, X, Y):
        d = np.asarray(Y, dtype=float) - np.asarray(X, dtype=float)
        t = np.linalg.norm(d, axis=1)
        u = d / np.maximum(t, 1e-12)[:, None]
        return t, -u, u


def travel_time_models(graph, fields: Sequence[AgentField | None]) -> list:
    """One `ArcTravelTimeModel` per agent of `graph`: its field's
    `FieldTravelTimeModel`, or, for an agent whose entry is None, the L2
    distance over its leading `Block.R` columns within the graph's state
    box."""
    if len(fields) != graph.num_agents:
        raise ValueError(f"fields has {len(fields)} entries for {graph.num_agents} agents")
    models = []
    for agent, agent_field in enumerate(fields):
        if agent_field is not None:
            models.append(FieldTravelTimeModel(agent_field))
            continue
        dim = 0
        for block in graph._robot_specs[agent]:
            if block.type != BlockType.R:
                break
            dim += block.size
        lo = graph.agent_col_offsets[agent]
        models.append(L2TravelTimeModel(np.asarray(graph._global_x_lb)[lo:lo + dim],
                                        np.asarray(graph._global_x_ub)[lo:lo + dim]))
    return models


def field_solver_kwargs(graph, fields, kwargs: dict) -> dict:
    """`kwargs` for a `GCSWaypointMPC`/`SCIPWaypointMPC`/`VRPWaypointMPC` constructor with
    `fields` turned into `travel_time_models`. The objective must be a
    travel-time one; it defaults to `kMinMaxTravelTime`."""
    if "travel_time_models" in kwargs:
        raise ValueError("pass either fields= or travel_time_models=, not both")
    objective = kwargs.setdefault("objective", WaypointObjective.kMinMaxTravelTime)
    if objective not in TRAVEL_TIME_OBJECTIVES:
        raise ValueError(f"fields= needs a travel-time objective, got {objective}")
    kwargs["travel_time_models"] = travel_time_models(graph, list(fields))
    return kwargs
