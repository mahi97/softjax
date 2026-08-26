"""Relaxation policy: mode, method, softness, straight-through, numerics."""

from __future__ import annotations

import contextvars
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any


_POLICY_STACK: contextvars.ContextVar[tuple[RelaxationPolicy, ...]] = (
    contextvars.ContextVar("softjax_policy_stack", default=())
)

_RELAXATION_FIELDS = (
    "mode",
    "method",
    "softness",
    "straight_through",
    "standardize",
    "gated_grad",
    "ot_kwargs",
    "implicit_diff",
    "sinkhorn_tol",
    "sinkhorn_max_iter",
    "lbfgs_tol",
    "lbfgs_max_iter",
    "log_prob_eps",
    "quantile_method",
)


def _freeze_mapping(value: dict[str, Any] | None) -> tuple[tuple[str, Any], ...] | None:
    if value is None:
        return None
    return tuple(sorted(value.items()))


def _thaw_mapping(
    value: dict[str, Any] | tuple[tuple[str, Any], ...] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    return dict(value)


@dataclass(frozen=True)
class RelaxationPolicy:
    """Declarative relaxation settings shared by SoftJAX operators.

    Fields left at their constructor defaults are treated as unset when the
    policy is merged into an outer scope, so nested policies override only the
    options they specify.

    **Arguments:**

    - `mode`: Regularizer family (`"hard"`, `"smooth"`, `"c0"`, `"c1"`, `"c2"`,
      plus elementwise extras such as `"_c1_pnorm"`).
    - `method`: Arraywise algorithm (`"softsort"`, `"neuralsort"`, `"ot"`, ...).
      `None` keeps the operator's own default.
    - `softness`: Positive softening strength.
    - `straight_through`: If True, operators use a hard forward pass and the
      requested soft mode for gradients.
    - `standardize`: Arraywise input standardization. `None` keeps the operator default.
    - `gated_grad`: Gated vs integrated value gradients. `None` keeps the operator default.
    - `ot_kwargs`: Extra keyword arguments forwarded to OT projection solvers.
    - `implicit_diff`, `sinkhorn_tol`, `sinkhorn_max_iter`, `lbfgs_tol`,
      `lbfgs_max_iter`: Numerical options merged into `ot_kwargs` when unset there.
    - `log_prob_eps`: Optional SoftIndex log-probability floor.
    - `quantile_method`: Quantile interpolation method.
    """

    mode: str | None = None
    method: str | None = None
    softness: float | None = None
    straight_through: bool | None = None
    standardize: bool | None = None
    gated_grad: bool | None = None
    ot_kwargs: dict[str, Any] | tuple[tuple[str, Any], ...] | None = None
    implicit_diff: bool | None = None
    sinkhorn_tol: float | None = None
    sinkhorn_max_iter: int | None = None
    lbfgs_tol: float | None = None
    lbfgs_max_iter: int | None = None
    log_prob_eps: float | None = None
    quantile_method: str | None = None
    _set: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    def __post_init__(self) -> None:
        specified = {name for name in _RELAXATION_FIELDS if getattr(self, name) is not None}
        object.__setattr__(self, "_set", frozenset(specified))
        object.__setattr__(self, "ot_kwargs", _freeze_mapping(_thaw_mapping(self.ot_kwargs)))

    def replace(self, **changes: Any) -> RelaxationPolicy:
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)

    def merge(self, inner: RelaxationPolicy) -> RelaxationPolicy:
        """Return `inner` overlaid on `self` (inner wins for fields it set)."""
        updates = {name: getattr(inner, name) for name in inner._set}
        return replace(self, **updates)

    def ot_options(self) -> dict[str, Any]:
        """Numerical options as a dict suitable for `ot_kwargs`."""
        out = dict(_thaw_mapping(self.ot_kwargs) or {})
        if "implicit_diff" not in out and self.implicit_diff is not None:
            out["implicit_diff"] = self.implicit_diff
        if "sinkhorn_tol" not in out and self.sinkhorn_tol is not None:
            out["sinkhorn_tol"] = self.sinkhorn_tol
        if "sinkhorn_max_iter" not in out and self.sinkhorn_max_iter is not None:
            out["sinkhorn_max_iter"] = self.sinkhorn_max_iter
        if "lbfgs_tol" not in out and self.lbfgs_tol is not None:
            out["lbfgs_tol"] = self.lbfgs_tol
        if "lbfgs_max_iter" not in out and self.lbfgs_max_iter is not None:
            out["lbfgs_max_iter"] = self.lbfgs_max_iter
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation of explicitly set fields."""
        data: dict[str, Any] = {}
        for name in _RELAXATION_FIELDS:
            if name not in self._set:
                continue
            value = getattr(self, name)
            if name == "ot_kwargs":
                value = _thaw_mapping(value)
            data[name] = value
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelaxationPolicy:
        unknown = set(data) - set(_RELAXATION_FIELDS)
        if unknown:
            raise ValueError(f"Unknown RelaxationPolicy fields: {sorted(unknown)}")
        return cls(**{k: data[k] for k in _RELAXATION_FIELDS if k in data})

    @classmethod
    def from_json(cls, payload: str) -> RelaxationPolicy:
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("RelaxationPolicy JSON must be an object")
        return cls.from_dict(parsed)

    def tree_flatten(self):
        # Treat the policy as a static pytree auxiliary so jit/vmap close over it.
        return ((), self.to_dict())

    @classmethod
    def tree_unflatten(cls, aux, _children):
        return cls.from_dict(aux)


def get_policy() -> RelaxationPolicy:
    """Return the innermost active policy, or an empty policy if none is set."""
    stack = _POLICY_STACK.get()
    if not stack:
        return RelaxationPolicy()
    return stack[-1]


def _push_policy(policy: RelaxationPolicy) -> RelaxationPolicy:
    stack = _POLICY_STACK.get()
    if stack:
        policy = stack[-1].merge(policy)
    _POLICY_STACK.set((*stack, policy))
    return policy


@contextmanager
def relaxation_scope(
    policy: RelaxationPolicy | None = None, **overrides: Any
) -> Iterator[RelaxationPolicy]:
    """Activate a relaxation policy for the duration of the `with` block.

    Nested scopes merge onto the outer policy: the inner policy wins for every
    field it sets, and inherits the rest.
    """
    if policy is None:
        policy = RelaxationPolicy(**overrides)
    elif overrides:
        policy = policy.merge(RelaxationPolicy(**overrides))
    stack = _POLICY_STACK.get()
    merged = stack[-1].merge(policy) if stack else policy
    token = _POLICY_STACK.set((*stack, merged))
    try:
        yield merged
    finally:
        _POLICY_STACK.reset(token)


# Register as a JAX pytree so policies can be closed over or passed through transforms.
try:
    import jax

    jax.tree_util.register_pytree_node(
        RelaxationPolicy,
        RelaxationPolicy.tree_flatten,
        RelaxationPolicy.tree_unflatten,
    )
except ImportError:  # pragma: no cover
    pass
