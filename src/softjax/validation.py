"""Centralized relaxation validation and policy application."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

import jax
from jax import tree_util as jtu

from softjax.policy import get_policy, RelaxationPolicy
from softjax.registry import get_operator, OperatorSpec
from softjax.utils import _validate_softness


def validate_combination(
    spec: OperatorSpec,
    *,
    mode: Any = None,
    method: Any = None,
    softness: Any = None,
) -> None:
    """Raise ValueError for unsupported operator / mode / method / softness."""
    # Hard paths ignore softness (notebooks pass softness=0 with mode="hard").
    if (
        spec.accepts_softness
        and softness is not None
        and mode not in ("hard", "_hard")
    ):
        _validate_softness(softness)

    if spec.supported_modes and mode is not None:
        if not isinstance(mode, jax.core.Tracer) and mode not in spec.supported_modes:
            raise ValueError(
                f"Invalid mode {mode!r} for {spec.name}. "
                f"Supported modes: {spec.supported_modes}."
            )

    if spec.accepts_method and method is not None:
        if method not in spec.supported_methods:
            raise ValueError(
                f"Invalid method {method!r} for {spec.name}. "
                f"Supported methods: {spec.supported_methods}."
            )
        if method == "smooth_sort" and mode not in (None, "smooth", "hard", "_hard"):
            raise ValueError(
                f"smooth_sort only supports mode='smooth', got mode={mode!r}"
            )


def resolve_arguments(
    spec: OperatorSpec,
    arguments: dict[str, Any],
    provided: set[str],
    policy: RelaxationPolicy | None,
) -> dict[str, Any]:
    """Fill unspecified relaxation fields from the active / explicit policy."""
    effective = get_policy()
    if policy is not None:
        effective = effective.merge(policy)

    if "mode" not in provided and effective.mode is not None:
        arguments["mode"] = effective.mode
    if spec.accepts_method and "method" not in provided and effective.method is not None:
        arguments["method"] = effective.method
    if spec.accepts_softness and "softness" not in provided and effective.softness is not None:
        arguments["softness"] = effective.softness
    if "standardize" in arguments and "standardize" not in provided:
        if effective.standardize is not None:
            arguments["standardize"] = effective.standardize
    if "gated_grad" in arguments and "gated_grad" not in provided:
        if effective.gated_grad is not None:
            arguments["gated_grad"] = effective.gated_grad
    if "gated" in arguments and "gated" not in provided and effective.gated_grad is not None:
        arguments["gated"] = effective.gated_grad
    if "quantile_method" in arguments and "quantile_method" not in provided:
        if effective.quantile_method is not None:
            arguments["quantile_method"] = effective.quantile_method
    if "log_prob_eps" in arguments and "log_prob_eps" not in provided:
        if effective.log_prob_eps is not None:
            arguments["log_prob_eps"] = effective.log_prob_eps

    if "ot_kwargs" in arguments:
        merged_ot = effective.ot_options()
        if "ot_kwargs" in provided and arguments["ot_kwargs"]:
            merged_ot = {**merged_ot, **arguments["ot_kwargs"]}
        elif merged_ot and "ot_kwargs" not in provided:
            arguments["ot_kwargs"] = merged_ot or None

    validate_combination(
        spec,
        mode=arguments.get("mode"),
        method=arguments.get("method"),
        softness=arguments.get("softness"),
    )
    return arguments


def _straight_through_call(fn: Callable, arguments: dict[str, Any]) -> Any:
    from softjax.straight_through import _replace_value_keep_grad

    soft_mode = arguments.get("mode", "smooth")
    hard_args = dict(arguments)
    hard_args["mode"] = "hard"
    fw_y = fn(**hard_args)
    bw_y = fn(**{**arguments, "mode": soft_mode})
    fw_leaves, fw_treedef = jtu.tree_flatten(fw_y, is_leaf=lambda x: x is None)
    bw_leaves, _ = jtu.tree_flatten(bw_y, is_leaf=lambda x: x is None)
    out_leaves = [
        _replace_value_keep_grad(f, b) for f, b in zip(fw_leaves, bw_leaves)
    ]
    return jtu.tree_unflatten(fw_treedef, out_leaves)


def relaxable(name: str) -> Callable[[Callable], Callable]:
    """Apply the active RelaxationPolicy and validate mode/method/softness.

    Existing call signatures are preserved. An optional keyword-only `policy`
    argument is accepted by the wrapper even if the wrapped function does not
    declare it. Explicit caller arguments always win over the policy.
    """

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            policy = kwargs.pop("policy", None)
            if policy is not None and not isinstance(policy, RelaxationPolicy):
                raise TypeError(
                    f"policy must be a RelaxationPolicy, got {type(policy).__name__}"
                )
            bound = sig.bind(*args, **kwargs)
            provided = set(bound.arguments)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            spec = get_operator(name)
            arguments = resolve_arguments(spec, arguments, provided, policy)
            effective = get_policy() if policy is None else get_policy().merge(policy)
            use_st = (
                effective.straight_through
                and "mode" not in provided
                and arguments.get("mode") not in ("hard", "_hard")
            )
            if use_st:
                return _straight_through_call(fn, arguments)
            return fn(**arguments)

        params = list(sig.parameters.values())
        if "policy" not in sig.parameters:
            params.append(
                inspect.Parameter(
                    "policy",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=RelaxationPolicy | None,
                )
            )
        wrapped.__signature__ = sig.replace(parameters=params)
        wrapped.__softjax_operator__ = name
        return wrapped

    return decorator
