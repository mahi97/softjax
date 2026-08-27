"""Tests for RelaxationPolicy, the operator registry, and policy propagation."""

import inspect
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import softjax as sj
from softjax.registry import OPERATORS

from . import common


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_registry_covers_all_public_operators():
    registered = set(OPERATORS)
    # Public callables that must have a registry entry.
    missing = []
    for name in sj.__all__:
        if name in {
            "RelaxationPolicy",
            "SoftBool",
            "SoftIndex",
            "get_policy",
            "relaxation_limitations",
            "relaxation_scope",
            "supported_relaxations",
            "random",
            "st",
            "grad_replace",
        }:
            continue
        if name.endswith("_st"):
            continue
        if name not in registered:
            missing.append(name)
    assert not missing, f"registry missing public operators: {missing}"


def test_registry_st_variants_exist():
    for spec in OPERATORS.values():
        if spec.st_variant is None:
            continue
        if spec.category == "random":
            assert hasattr(sj.random, spec.st_variant), spec.st_variant
        else:
            assert hasattr(sj, spec.st_variant), spec.st_variant


def test_supported_relaxations_and_limitations_api():
    all_info = sj.supported_relaxations()
    assert set(all_info) == set(OPERATORS)
    q = sj.supported_relaxations("quantile")
    assert q["hard_jax"] == "jax.numpy.quantile"
    assert "ot" in q["supported_methods"]
    assert q["output_semantics"] == "value"
    assert "derivative_guarantees" in q

    limits = sj.relaxation_limitations("quantile")
    assert "quantile" in limits
    assert any("OT c0" in item for item in limits["quantile"])
    assert set(sj.relaxation_limitations()) == set(OPERATORS)


def test_unknown_operator_errors():
    with pytest.raises(KeyError, match="Unknown SoftJAX operator"):
        sj.supported_relaxations("not_an_operator")


# ---------------------------------------------------------------------------
# Policy serialization / merge / pytree
# ---------------------------------------------------------------------------


def test_policy_serialization_roundtrip():
    policy = sj.RelaxationPolicy(
        mode="c2",
        method="ot",
        softness=0.25,
        straight_through=True,
        ot_kwargs={"lbfgs_tol": 1e-7},
        implicit_diff=True,
        sinkhorn_max_iter=123,
    )
    restored = sj.RelaxationPolicy.from_dict(policy.to_dict())
    assert restored.to_dict() == policy.to_dict()
    assert json.loads(policy.to_json()) == policy.to_dict()
    assert sj.RelaxationPolicy.from_json(policy.to_json()).to_dict() == policy.to_dict()


def test_policy_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown RelaxationPolicy fields"):
        sj.RelaxationPolicy.from_dict({"mode": "smooth", "not_a_field": 1})


def test_policy_replace_and_ot_options():
    policy = sj.RelaxationPolicy(mode="c1", lbfgs_tol=1e-8)
    replaced = policy.replace(softness=0.3)
    assert replaced.mode == "c1"
    assert replaced.softness == 0.3
    opts = replaced.ot_options()
    assert opts["lbfgs_tol"] == 1e-8


def test_empty_policy_is_identity_for_get_policy():
    assert sj.get_policy().to_dict() == {}


# ---------------------------------------------------------------------------
# Policy propagation and nesting
# ---------------------------------------------------------------------------


def test_policy_scope_changes_default_mode():
    x = jnp.array([0.2, -0.4, 0.7, 0.1])
    with_policy = None
    with sj.relaxation_scope(sj.RelaxationPolicy(mode="c0", softness=0.4)):
        with_policy = sj.sort(x, method="softsort")
        assert sj.get_policy().mode == "c0"
    without = sj.sort(x, method="softsort", mode="c0", softness=0.4)
    common.assert_allclose(with_policy, without, tol=1e-10)
    # Outside the scope the default is unchanged.
    default = sj.sort(x, method="softsort")
    assert not np.allclose(np.asarray(default), np.asarray(with_policy), atol=1e-4)


def test_explicit_kwargs_override_policy():
    x = jnp.array([0.2, -0.4, 0.7, 0.1])
    expected = sj.max(x, mode="smooth", softness=1.0, method="softsort")
    with sj.relaxation_scope(sj.RelaxationPolicy(mode="c2", softness=0.01)):
        actual = sj.max(x, mode="smooth", softness=1.0, method="softsort")
    common.assert_allclose(actual, expected, tol=1e-10)


def test_nested_policies_inner_overrides_outer():
    x = jnp.array([0.2, -0.4, 0.7, 0.1])
    outer = sj.RelaxationPolicy(mode="c1", softness=0.5, method="softsort")
    with sj.relaxation_scope(outer):
        with sj.relaxation_scope(sj.RelaxationPolicy(softness=0.2)):
            nested = sj.get_policy()
            assert nested.mode == "c1"
            assert nested.method == "softsort"
            assert nested.softness == 0.2
            actual = sj.min(x)
    expected = sj.min(x, mode="c1", softness=0.2, method="softsort")
    common.assert_allclose(actual, expected, tol=1e-10)


def test_policy_keyword_argument():
    x = jnp.array([0.2, -0.4, 0.7, 0.1])
    policy = sj.RelaxationPolicy(mode="c0", softness=0.3, method="neuralsort")
    actual = sj.sort(x, policy=policy)
    expected = sj.sort(x, mode="c0", softness=0.3, method="neuralsort")
    common.assert_allclose(actual, expected, tol=1e-10)


def test_policy_keyword_must_be_policy():
    x = jnp.array([0.2, -0.4, 0.7])
    with pytest.raises(TypeError, match="RelaxationPolicy"):
        sj.abs(x, policy="smooth")


def test_straight_through_policy_matches_st_wrapper():
    x = jnp.array([-0.5, 0.3, 1.0], dtype=jnp.float64)
    with sj.relaxation_scope(sj.RelaxationPolicy(mode="smooth", straight_through=True)):
        fw = sj.relu(x)
        grad = jax.grad(lambda z: jnp.sum(sj.relu(z)))(x)
    common.assert_allclose(fw, jax.nn.relu(x), tol=1e-12)
    common.assert_allclose(grad, jax.grad(lambda z: jnp.sum(sj.relu_st(z)))(x), tol=1e-8)


# ---------------------------------------------------------------------------
# Invalid combinations
# ---------------------------------------------------------------------------


def test_invalid_mode_is_rejected_by_registry():
    x = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Invalid mode"):
        sj.sort(x, mode="not-a-mode")


def test_invalid_method_is_rejected_by_registry():
    x = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Invalid method"):
        sj.argsort(x, method="not-a-method")


def test_smooth_sort_rejects_nonsmooth_mode():
    x = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="smooth_sort only supports"):
        sj.sort(x, method="smooth_sort", mode="c2")


def test_nonpositive_softness_rejected():
    x = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="softness must be positive"):
        sj.abs(x, softness=0.0)


def test_hard_mode_ignores_nonpositive_softness():
    x = jnp.array([1.0, 2.0, 3.0])
    common.assert_jax_parity(sj.argmax(x, mode="hard", softness=0.0), jax.nn.one_hot(2, 3))


# ---------------------------------------------------------------------------
# jit / vmap
# ---------------------------------------------------------------------------


def test_policy_scope_is_jit_compatible():
    x = jnp.array([0.3, -0.2, 0.8, 0.1], dtype=jnp.float64)

    def body(z):
        with sj.relaxation_scope(sj.RelaxationPolicy(mode="c0", softness=0.4)):
            return sj.abs(z)

    common.assert_allclose(body(x), jax.jit(body)(x), tol=1e-8)


def test_policy_scope_is_vmap_compatible():
    xs = jnp.array([[0.3, -0.2, 0.8], [0.1, 0.4, -0.5]], dtype=jnp.float64)

    def body(z):
        with sj.relaxation_scope(sj.RelaxationPolicy(mode="smooth", softness=0.5)):
            return sj.relu(z)

    common.assert_allclose(jax.vmap(body)(xs), jnp.stack([body(xs[0]), body(xs[1])]))


def test_policy_pytree_roundtrip_under_jit():
    policy = sj.RelaxationPolicy(mode="c1", softness=0.2)

    def fn(z, p):
        return sj.sign(z, policy=p)

    x = jnp.array([-0.4, 0.2, 0.9])
    common.assert_allclose(fn(x, policy), jax.jit(fn)(x, policy), tol=1e-8)


# ---------------------------------------------------------------------------
# Backward-compatible outputs
# ---------------------------------------------------------------------------


GOLDEN_X = jnp.array([-0.8, -0.1, 0.3, 0.5, 1.2], dtype=jnp.float64)


@pytest.mark.parametrize(
    "name, call_explicit, call_policy",
    [
        (
            "abs",
            lambda x: sj.abs(x, mode="c1", softness=0.2),
            lambda x: sj.abs(x, policy=sj.RelaxationPolicy(mode="c1", softness=0.2)),
        ),
        (
            "relu",
            lambda x: sj.relu(x, mode="smooth", softness=0.3),
            lambda x: sj.relu(x, policy=sj.RelaxationPolicy(mode="smooth", softness=0.3)),
        ),
        (
            "sign",
            lambda x: sj.sign(x, mode="c2", softness=0.15),
            lambda x: sj.sign(x, policy=sj.RelaxationPolicy(mode="c2", softness=0.15)),
        ),
        (
            "heaviside",
            lambda x: sj.heaviside(x, mode="c0", softness=0.4),
            lambda x: sj.heaviside(x, policy=sj.RelaxationPolicy(mode="c0", softness=0.4)),
        ),
        (
            "greater",
            lambda x: sj.greater(x, 0.0, mode="smooth", softness=0.2),
            lambda x: sj.greater(
                x, 0.0, policy=sj.RelaxationPolicy(mode="smooth", softness=0.2)
            ),
        ),
        (
            "sort",
            lambda x: sj.sort(x, mode="c0", method="softsort", softness=0.5),
            lambda x: sj.sort(
                x, policy=sj.RelaxationPolicy(mode="c0", method="softsort", softness=0.5)
            ),
        ),
        (
            "max",
            lambda x: sj.max(x, mode="smooth", method="neuralsort", softness=0.4),
            lambda x: sj.max(
                x,
                policy=sj.RelaxationPolicy(
                    mode="smooth", method="neuralsort", softness=0.4
                ),
            ),
        ),
        (
            "quantile",
            lambda x: sj.quantile(x, 0.5, mode="c1", method="softsort", softness=0.3),
            lambda x: sj.quantile(
                x,
                0.5,
                policy=sj.RelaxationPolicy(mode="c1", method="softsort", softness=0.3),
            ),
        ),
        (
            "argmax",
            lambda x: sj.argmax(x, mode="smooth", method="softsort", softness=0.4),
            lambda x: sj.argmax(
                x,
                policy=sj.RelaxationPolicy(mode="smooth", method="softsort", softness=0.4),
            ),
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_policy_matches_explicit_kwargs(name, call_explicit, call_policy):
    explicit = call_explicit(GOLDEN_X)
    via_policy = call_policy(GOLDEN_X)
    common.assert_allclose(via_policy, explicit, tol=0.0, err_msg=name)


def test_refactored_operators_keep_hard_outputs():
    x = GOLDEN_X
    common.assert_jax_parity(sj.abs(x, mode="hard"), jnp.abs(x))
    common.assert_jax_parity(sj.max(x, mode="hard"), jnp.max(x))
    common.assert_jax_parity(sj.sort(x, mode="hard"), jnp.sort(x))
    common.assert_jax_parity(
        sj.quantile(x, 0.5, mode="hard"), jnp.quantile(x, 0.5)
    )
    common.assert_jax_parity(sj.relu(x, mode="hard"), jax.nn.relu(x))


def test_relaxable_signature_still_has_original_parameters():
    sig = inspect.signature(sj.quantile)
    assert "mode" in sig.parameters
    assert "method" in sig.parameters
    assert "softness" in sig.parameters
    assert "policy" in sig.parameters
    assert sig.parameters["mode"].default == "smooth"
    assert sig.parameters["method"].default == "neuralsort"


def test_unrelaxable_operators_stay_callable():
    cond = jnp.array([0.2, 0.8])
    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    common.assert_allclose(sj.where(cond, x, y), x * cond + y * (1.0 - cond), tol=1e-12)
    common.assert_allclose(sj.logical_not(cond), 1.0 - cond, tol=1e-12)
    common.assert_finite(sj.sqrt(jnp.array([0.0, 1.0, 4.0])))
