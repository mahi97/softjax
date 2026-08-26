"""Audit every public operator and random sampler.

Covers hard-mode JAX parity, near-hard convergence, finite gradients,
jit / vmap agreement, and output shapes. Combinatorial method/mode
sweeps live in the dedicated test modules; this file is the checklist
that nothing public is left untested.
"""

import jax
import jax.numpy as jnp
import pytest

import softjax as sj

from . import common


# ---------------------------------------------------------------------------
# Catalog of public operators (must stay in sync with softjax.__all__)
# ---------------------------------------------------------------------------

VALUE_REDUCTIONS = {
    "max": (jnp.max, dict(axis=-1)),
    "min": (jnp.min, dict(axis=-1)),
    "sort": (jnp.sort, dict(axis=-1)),
    "median": (jnp.median, dict(axis=-1)),
    "quantile": (jnp.quantile, dict(q=0.5, axis=-1)),
    "percentile": (jnp.percentile, dict(q=50.0, axis=-1)),
}

ELEMENTWISE_HARD = {
    "abs": jnp.abs,
    "relu": jax.nn.relu,
    "sign": jnp.sign,
    "round": jnp.round,
    "heaviside": lambda x: jnp.heaviside(x, 0.5),
}

COMPARISON_HARD = {
    "greater": jnp.greater,
    "greater_equal": jnp.greater_equal,
    "less": jnp.less,
    "less_equal": jnp.less_equal,
    "equal": jnp.equal,
    "not_equal": jnp.not_equal,
}

RANDOM_FNS = (
    "categorical",
    "bernoulli",
    "choice",
    "rademacher",
    "permutation",
    "binomial",
    "multinomial",
)


def _public_names():
    return set(sj.__all__)


# ---------------------------------------------------------------------------
# Hard-mode JAX parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, spec", VALUE_REDUCTIONS.items(), ids=VALUE_REDUCTIONS)
def test_value_reduction_hard_matches_jax(name, spec):
    jnp_fn, kwargs = spec
    x = common.make_array((5,), "float64", "jax")
    soft_kwargs = dict(kwargs)
    if name == "percentile":
        soft_kwargs = dict(p=kwargs["q"], axis=kwargs["axis"])
    common.assert_jax_parity(
        getattr(sj, name)(x, mode="hard", **soft_kwargs),
        jnp_fn(x, **kwargs),
        msg=name,
    )


@pytest.mark.parametrize("name, jnp_fn", ELEMENTWISE_HARD.items(), ids=ELEMENTWISE_HARD)
def test_elementwise_hard_matches_jax(name, jnp_fn):
    x = common.make_array((4,), "float64", "jax")
    common.assert_jax_parity(getattr(sj, name)(x, mode="hard"), jnp_fn(x), msg=name)


def test_clip_hard_matches_jax():
    x = common.make_array((4,), "float64", "jax")
    common.assert_jax_parity(sj.clip(x, -0.3, 0.4, mode="hard"), jnp.clip(x, -0.3, 0.4))


@pytest.mark.parametrize("name, jnp_fn", COMPARISON_HARD.items(), ids=COMPARISON_HARD)
def test_comparison_hard_matches_jax(name, jnp_fn):
    x, y = common.pair_arrays((4,), "float64", "jax")
    common.assert_jax_parity(
        getattr(sj, name)(x, y, mode="hard"),
        jnp_fn(x, y).astype(x.dtype),
        msg=name,
    )


def test_isclose_hard_matches_jax():
    x, y = common.pair_arrays((4,), "float64", "jax")
    common.assert_jax_parity(
        sj.isclose(x, y, mode="hard"),
        jnp.isclose(x, y).astype(x.dtype),
    )


# ---------------------------------------------------------------------------
# Near-hard convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["max", "min", "sort", "median", "quantile", "percentile"])
@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_value_ops_near_hard_match_hard(name, mode):
    x = common.make_array((5,), "float64", "jax")
    kwargs = dict(axis=-1, mode=mode, softness=common.NEAR_HARD_SOFTNESS, method="softsort")
    if name == "quantile":
        kwargs["q"] = 0.5
    elif name == "percentile":
        kwargs["p"] = 50.0
    fn = getattr(sj, name)
    common.assert_allclose(fn(x, **kwargs), fn(x, mode="hard", **{
        k: v for k, v in kwargs.items() if k in ("axis", "q", "p")
    }))


# ---------------------------------------------------------------------------
# Finite gradients, jit, vmap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["abs", "relu", "sign", "round", "heaviside", "max", "min", "sort", "median"],
)
@pytest.mark.parametrize("mode", ["smooth", "c2"])
def test_public_op_finite_grad_jit_vmap(name, mode):
    x = common.gradient_input((4,), jnp.float64)
    fn = getattr(sj, name)
    extra = {} if name in ELEMENTWISE_HARD else dict(axis=-1)

    def loss(z):
        return jnp.sum(fn(z, mode=mode, softness=1.0, **extra))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"{name} {mode}")
    common.assert_allclose(grad, jax.jit(jax.grad(loss))(x), tol=1e-5)

    xs = jnp.stack([x, jnp.flip(x)])
    common.assert_allclose(jax.vmap(loss)(xs), jnp.stack([loss(xs[0]), loss(xs[1])]), tol=1e-5)


# ---------------------------------------------------------------------------
# Random samplers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", RANDOM_FNS)
def test_random_sampler_is_public(name):
    assert hasattr(sj.random, name)
    assert hasattr(sj.random, f"{name}_st")
    assert name in dir(sj.random)


def test_random_hard_parity_and_soft_contracts():
    key = jax.random.key(0)
    logits = jnp.array([0.2, 1.1, -0.4])
    p = jnp.array([0.2, 0.5, 0.3])

    cat = sj.random.categorical(key, logits, mode="hard")
    expected = jax.nn.one_hot(
        jax.random.categorical(key, logits), logits.shape[-1]
    )
    common.assert_jax_parity(cat, expected, msg="categorical")

    bern = sj.random.bernoulli(key, p, mode="hard")
    common.assert_jax_parity(
        bern, jax.random.bernoulli(key, p).astype(p.dtype), msg="bernoulli"
    )

    choice = sj.random.choice(key, 4, shape=(3,), mode="hard")
    common.assert_jax_parity(
        choice, jax.random.choice(key, 4, shape=(3,)), msg="choice"
    )

    perm = sj.random.permutation(key, 4, mode="hard")
    common.assert_jax_parity(perm, jax.random.permutation(key, 4), msg="permutation")

    rad = sj.random.rademacher(key, (3,), mode="hard")
    common.assert_jax_parity(rad, jax.random.rademacher(key, (3,)), msg="rademacher")

    soft_cat = sj.random.categorical(key, logits, mode="smooth", softness=1.0)
    common.assert_simplex(soft_cat)
    common.assert_finite(soft_cat)

    def cat_loss(z):
        return jnp.sum(sj.random.categorical(key, z, mode="smooth", softness=1.0))

    common.assert_finite(jax.grad(cat_loss)(logits), msg="categorical grad")
    common.assert_allclose(
        jax.jit(lambda z: sj.random.categorical(key, z, mode="smooth", softness=1.0))(
            logits
        ),
        soft_cat,
        tol=1e-5,
    )


def test_random_binomial_multinomial_soft_shapes():
    key = jax.random.key(1)
    p = jnp.array(0.4)
    out_b = sj.random.binomial(key, 3, p, mode="smooth", softness=1.0)
    assert out_b.shape == ()
    common.assert_finite(out_b)

    p_m = jnp.array([0.2, 0.5, 0.3])
    out_m = sj.random.multinomial(key, 3, p_m, mode="smooth", softness=1.0)
    assert out_m.shape == (3,)
    common.assert_finite(out_m)
    common.assert_allclose(jnp.sum(out_m), 3.0, tol=1e-5)


# ---------------------------------------------------------------------------
# Public API completeness
# ---------------------------------------------------------------------------


def test_public_api_has_no_missing_audit_targets():
    names = _public_names()
    # Core families that this audit (plus the dedicated modules) must cover.
    required = {
        "abs",
        "relu",
        "clip",
        "sign",
        "round",
        "heaviside",
        "max",
        "min",
        "argmax",
        "argmin",
        "sort",
        "argsort",
        "rank",
        "quantile",
        "argquantile",
        "percentile",
        "argpercentile",
        "median",
        "argmedian",
        "top_k",
        "where",
        "take",
        "take_along_axis",
        "choose",
        "random",
    }
    missing = required - names
    assert not missing, f"public API missing expected names: {missing}"
