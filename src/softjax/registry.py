"""Registry of SoftJAX operators, methods, output semantics, and limitations."""

from __future__ import annotations

from dataclasses import dataclass


ARRAYWISE_MODES = ("hard", "_hard", "smooth", "c0", "c1", "c2")
ELEMENTWISE_HELPER_MODES = ("smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm")
# Public elementwise/comparison ops forward `mode` into sigmoidal/softrelu, so
# the private p-norm families are valid even when omitted from the type hints.
ELEMENTWISE_MODES = ("hard", *ELEMENTWISE_HELPER_MODES)
RANDOM_MODES = ("hard", "_hard", "smooth", "c0", "c1", "c2")

ARG_METHODS = ("ot", "softsort", "neuralsort", "sorting_network")
VALUE_METHODS = (
    "ot",
    "softsort",
    "neuralsort",
    "fast_soft_sort",
    "smooth_sort",
    "sorting_network",
)

_OT_C0_HESSIAN = (
    "OT c0 (L2 LBFGS + ImplicitAdjoint) does not support second-order "
    "differentiation: the dual Hessian involves a Heaviside and the implicit "
    "linear solver receives non-finite inputs when differentiated twice."
)
_SPARSE_LOG = (
    "Sparse modes (c0/c1/c2) can produce exact zero probabilities whose log is "
    "-inf; use return_log_probs=True with log_prob_eps for bounded logs."
)
_SMOOTH_SORT_MODE = "smooth_sort supports only mode='smooth' (and hard/_hard)."
_VALUE_ONLY = (
    "fast_soft_sort, smooth_sort, and sorting_network value paths do not return "
    "a SoftIndex (top_k indices are None)."
)
_STATIC_COUNT = (
    "Soft binomial/multinomial currently require a concrete scalar nonnegative "
    "integer count."
)


@dataclass(frozen=True)
class OperatorSpec:
    """Description of one public SoftJAX operator."""

    name: str
    category: str
    hard_jax: str
    soft_impl: str
    supported_modes: tuple[str, ...]
    supported_methods: tuple[str, ...] = ()
    default_mode: str = "smooth"
    default_method: str | None = None
    default_softness: float = 0.1
    default_standardize: bool | None = None
    default_gated_grad: bool | None = None
    output_semantics: str = "value"
    derivative_guarantees: str = (
        "Finite first derivatives in every supported soft mode. "
        "Higher-order derivatives are finite for closed-form / C1+ regularizers."
    )
    limitations: tuple[str, ...] = ()
    accepts_softness: bool = True
    accepts_method: bool = False
    st_variant: str | None = None
    relaxable: bool = True


def _spec(**kwargs) -> OperatorSpec:
    return OperatorSpec(**kwargs)


def _elementwise(name: str, hard_jax: str, **kwargs) -> OperatorSpec:
    return _spec(
        name=name,
        category="elementwise",
        hard_jax=hard_jax,
        soft_impl=f"softjax.{name}",
        supported_modes=ELEMENTWISE_MODES,
        output_semantics="value",
        st_variant=f"{name}_st",
        **kwargs,
    )


def _comparison(name: str, hard_jax: str) -> OperatorSpec:
    return _spec(
        name=name,
        category="comparison",
        hard_jax=hard_jax,
        soft_impl=f"softjax.{name}",
        supported_modes=ELEMENTWISE_MODES,
        output_semantics="soft_bool",
        st_variant=f"{name}_st",
    )


def _arraywise(
    name: str,
    hard_jax: str,
    *,
    methods: tuple[str, ...],
    default_method: str,
    output_semantics: str,
    st_variant: str | None = None,
    extra_limitations: tuple[str, ...] = (),
) -> OperatorSpec:
    limitations = extra_limitations
    if output_semantics in {"soft_index", "value_and_soft_index"}:
        limitations = (_SPARSE_LOG, *limitations)
    if "smooth_sort" in methods:
        limitations = (_SMOOTH_SORT_MODE, *limitations)
    if output_semantics == "value_and_soft_index":
        limitations = (_VALUE_ONLY, *limitations)
    if "ot" in methods:
        limitations = (_OT_C0_HESSIAN, *limitations)
    return _spec(
        name=name,
        category="arraywise",
        hard_jax=hard_jax,
        soft_impl=f"softjax.{name}",
        supported_modes=ARRAYWISE_MODES,
        supported_methods=methods,
        default_method=default_method,
        output_semantics=output_semantics,
        accepts_method=True,
        default_standardize=True,
        default_gated_grad=True if output_semantics != "soft_index" else None,
        st_variant=st_variant,
        limitations=limitations,
    )


OPERATORS: dict[str, OperatorSpec] = {}


def _register(spec: OperatorSpec) -> OperatorSpec:
    OPERATORS[spec.name] = spec
    return spec


for _spec_obj in (
    _elementwise("abs", "jax.numpy.abs"),
    _elementwise("relu", "jax.nn.relu", default_gated_grad=False),
    _elementwise("clip", "jax.numpy.clip", default_gated_grad=False),
    _elementwise("sign", "jax.numpy.sign"),
    _elementwise("round", "jax.numpy.round"),
    _elementwise("heaviside", "jax.numpy.heaviside"),
    _spec(
        name="sigmoidal",
        category="elementwise",
        hard_jax="jax.nn.sigmoid",
        soft_impl="softjax.sigmoidal",
        supported_modes=ELEMENTWISE_HELPER_MODES,
        output_semantics="soft_bool",
    ),
    _spec(
        name="softrelu",
        category="elementwise",
        hard_jax="jax.nn.softplus",
        soft_impl="softjax.softrelu",
        supported_modes=ELEMENTWISE_HELPER_MODES,
        output_semantics="value",
        default_gated_grad=False,
    ),
    _comparison("greater", "jax.numpy.greater"),
    _comparison("greater_equal", "jax.numpy.greater_equal"),
    _comparison("less", "jax.numpy.less"),
    _comparison("less_equal", "jax.numpy.less_equal"),
    _comparison("equal", "jax.numpy.equal"),
    _comparison("not_equal", "jax.numpy.not_equal"),
    _comparison("isclose", "jax.numpy.isclose"),
    _spec(
        name="logical_not",
        category="logical",
        hard_jax="jax.numpy.logical_not",
        soft_impl="softjax.logical_not",
        supported_modes=(),
        output_semantics="soft_bool",
        accepts_softness=False,
        relaxable=False,
    ),
    _spec(
        name="logical_and",
        category="logical",
        hard_jax="jax.numpy.logical_and",
        soft_impl="softjax.logical_and",
        supported_modes=(),
        output_semantics="soft_bool",
        accepts_softness=False,
        relaxable=False,
    ),
    _spec(
        name="logical_or",
        category="logical",
        hard_jax="jax.numpy.logical_or",
        soft_impl="softjax.logical_or",
        supported_modes=(),
        output_semantics="soft_bool",
        accepts_softness=False,
        relaxable=False,
    ),
    _spec(
        name="logical_xor",
        category="logical",
        hard_jax="jax.numpy.logical_xor",
        soft_impl="softjax.logical_xor",
        supported_modes=(),
        output_semantics="soft_bool",
        accepts_softness=False,
        relaxable=False,
    ),
    _spec(
        name="all",
        category="logical",
        hard_jax="jax.numpy.all",
        soft_impl="softjax.all",
        supported_modes=(),
        output_semantics="soft_bool",
        accepts_softness=False,
        relaxable=False,
    ),
    _spec(
        name="any",
        category="logical",
        hard_jax="jax.numpy.any",
        soft_impl="softjax.any",
        supported_modes=(),
        output_semantics="soft_bool",
        accepts_softness=False,
        relaxable=False,
    ),
    _arraywise(
        "argmax",
        "jax.numpy.argmax",
        methods=ARG_METHODS,
        default_method="softsort",
        output_semantics="soft_index",
        st_variant="argmax_st",
    ),
    _arraywise(
        "argmin",
        "jax.numpy.argmin",
        methods=ARG_METHODS,
        default_method="softsort",
        output_semantics="soft_index",
        st_variant="argmin_st",
    ),
    _arraywise(
        "max",
        "jax.numpy.max",
        methods=VALUE_METHODS,
        default_method="softsort",
        output_semantics="value",
        st_variant="max_st",
    ),
    _arraywise(
        "min",
        "jax.numpy.min",
        methods=VALUE_METHODS,
        default_method="softsort",
        output_semantics="value",
        st_variant="min_st",
    ),
    _arraywise(
        "argsort",
        "jax.numpy.argsort",
        methods=ARG_METHODS,
        default_method="neuralsort",
        output_semantics="soft_index",
        st_variant="argsort_st",
    ),
    _arraywise(
        "sort",
        "jax.numpy.sort",
        methods=VALUE_METHODS,
        default_method="neuralsort",
        output_semantics="value",
        st_variant="sort_st",
    ),
    _arraywise(
        "rank",
        "softjax.rank (two jnp.argsort in hard mode)",
        methods=VALUE_METHODS,
        default_method="neuralsort",
        output_semantics="value",
        st_variant="rank_st",
    ),
    _arraywise(
        "argquantile",
        "jax.numpy.quantile (one/two-hot indices)",
        methods=ARG_METHODS,
        default_method="neuralsort",
        output_semantics="soft_index",
        st_variant="argquantile_st",
    ),
    _arraywise(
        "quantile",
        "jax.numpy.quantile",
        methods=VALUE_METHODS,
        default_method="neuralsort",
        output_semantics="value",
        st_variant="quantile_st",
    ),
    _arraywise(
        "argpercentile",
        "jax.numpy.percentile (one/two-hot indices)",
        methods=ARG_METHODS,
        default_method="neuralsort",
        output_semantics="soft_index",
        st_variant="argpercentile_st",
    ),
    _arraywise(
        "percentile",
        "jax.numpy.percentile",
        methods=VALUE_METHODS,
        default_method="neuralsort",
        output_semantics="value",
        st_variant="percentile_st",
    ),
    _arraywise(
        "argmedian",
        "jax.numpy.median (one/two-hot indices)",
        methods=ARG_METHODS,
        default_method="neuralsort",
        output_semantics="soft_index",
        st_variant="argmedian_st",
    ),
    _arraywise(
        "median",
        "jax.numpy.median",
        methods=VALUE_METHODS,
        default_method="neuralsort",
        output_semantics="value",
        st_variant="median_st",
    ),
    _arraywise(
        "top_k",
        "jax.lax.top_k",
        methods=VALUE_METHODS,
        default_method="neuralsort",
        output_semantics="value_and_soft_index",
        st_variant="top_k_st",
    ),
    _spec(
        name="where",
        category="selection",
        hard_jax="jax.numpy.where",
        soft_impl="softjax.where",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
    ),
    _spec(
        name="take",
        category="selection",
        hard_jax="jax.numpy.take",
        soft_impl="softjax.take",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
    ),
    _spec(
        name="take_along_axis",
        category="selection",
        hard_jax="jax.numpy.take_along_axis",
        soft_impl="softjax.take_along_axis",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
    ),
    _spec(
        name="choose",
        category="selection",
        hard_jax="jax.numpy.choose",
        soft_impl="softjax.choose",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
    ),
    _spec(
        name="dynamic_index_in_dim",
        category="selection",
        hard_jax="jax.lax.dynamic_index_in_dim",
        soft_impl="softjax.dynamic_index_in_dim",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
    ),
    _spec(
        name="dynamic_slice_in_dim",
        category="selection",
        hard_jax="jax.lax.dynamic_slice_in_dim",
        soft_impl="softjax.dynamic_slice_in_dim",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
    ),
    _spec(
        name="dynamic_slice",
        category="selection",
        hard_jax="jax.lax.dynamic_slice",
        soft_impl="softjax.dynamic_slice",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
    ),
    _spec(
        name="sqrt",
        category="autograd_safe",
        hard_jax="jax.numpy.sqrt",
        soft_impl="softjax.sqrt",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
        derivative_guarantees="Finite at x=0 (double-where).",
    ),
    _spec(
        name="arcsin",
        category="autograd_safe",
        hard_jax="jax.numpy.arcsin",
        soft_impl="softjax.arcsin",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
        derivative_guarantees="Finite at |x|=1 (double-where).",
    ),
    _spec(
        name="arccos",
        category="autograd_safe",
        hard_jax="jax.numpy.arccos",
        soft_impl="softjax.arccos",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
        derivative_guarantees="Finite at |x|=1 (double-where).",
    ),
    _spec(
        name="div",
        category="autograd_safe",
        hard_jax="jax.numpy.divide",
        soft_impl="softjax.div",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
        derivative_guarantees="Finite at y=0 (double-where).",
    ),
    _spec(
        name="log",
        category="autograd_safe",
        hard_jax="jax.numpy.log",
        soft_impl="softjax.log",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
        derivative_guarantees="Finite at x=0 (double-where).",
    ),
    _spec(
        name="norm",
        category="autograd_safe",
        hard_jax="jax.numpy.linalg.norm",
        soft_impl="softjax.norm",
        supported_modes=(),
        output_semantics="value",
        accepts_softness=False,
        relaxable=False,
        derivative_guarantees="Finite at the origin via autograd-safe sqrt.",
    ),
    _spec(
        name="categorical",
        category="random",
        hard_jax="jax.random.categorical",
        soft_impl="softjax.random.categorical",
        supported_modes=RANDOM_MODES,
        supported_methods=ARG_METHODS,
        default_method="softsort",
        default_standardize=False,
        output_semantics="soft_index",
        accepts_method=True,
        st_variant="categorical_st",
        limitations=(_SPARSE_LOG, _OT_C0_HESSIAN),
    ),
    _spec(
        name="bernoulli",
        category="random",
        hard_jax="jax.random.bernoulli",
        soft_impl="softjax.random.bernoulli",
        supported_modes=RANDOM_MODES,
        output_semantics="soft_bool",
        st_variant="bernoulli_st",
    ),
    _spec(
        name="choice",
        category="random",
        hard_jax="jax.random.choice",
        soft_impl="softjax.random.choice",
        supported_modes=RANDOM_MODES,
        supported_methods=ARG_METHODS,
        default_method="softsort",
        default_standardize=False,
        output_semantics="value",
        accepts_method=True,
        st_variant="choice_st",
        limitations=(_OT_C0_HESSIAN,),
    ),
    _spec(
        name="rademacher",
        category="random",
        hard_jax="jax.random.rademacher",
        soft_impl="softjax.random.rademacher",
        supported_modes=RANDOM_MODES,
        output_semantics="value",
        st_variant="rademacher_st",
    ),
    _spec(
        name="permutation",
        category="random",
        hard_jax="jax.random.permutation",
        soft_impl="softjax.random.permutation",
        supported_modes=RANDOM_MODES,
        supported_methods=ARG_METHODS,
        default_method="neuralsort",
        default_standardize=False,
        output_semantics="value",
        accepts_method=True,
        st_variant="permutation_st",
        limitations=(_OT_C0_HESSIAN,),
    ),
    _spec(
        name="binomial",
        category="random",
        hard_jax="jax.random.binomial",
        soft_impl="softjax.random.binomial",
        supported_modes=RANDOM_MODES,
        output_semantics="value",
        st_variant="binomial_st",
        limitations=(_STATIC_COUNT,),
    ),
    _spec(
        name="multinomial",
        category="random",
        hard_jax="jax.random.multinomial",
        soft_impl="softjax.random.multinomial",
        supported_modes=RANDOM_MODES,
        supported_methods=ARG_METHODS,
        default_method="softsort",
        default_standardize=False,
        output_semantics="value",
        accepts_method=True,
        st_variant="multinomial_st",
        limitations=(_STATIC_COUNT, _OT_C0_HESSIAN),
    ),
):
    _register(_spec_obj)


def get_operator(name: str) -> OperatorSpec:
    try:
        return OPERATORS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown SoftJAX operator {name!r}") from exc


def list_operators(*, category: str | None = None) -> tuple[str, ...]:
    names = []
    for name, spec in OPERATORS.items():
        if category is None or spec.category == category:
            names.append(name)
    return tuple(names)


def _spec_to_public_dict(spec: OperatorSpec) -> dict:
    return {
        "name": spec.name,
        "category": spec.category,
        "hard_jax": spec.hard_jax,
        "soft_impl": spec.soft_impl,
        "supported_modes": spec.supported_modes,
        "supported_methods": spec.supported_methods,
        "default_mode": spec.default_mode,
        "default_method": spec.default_method,
        "output_semantics": spec.output_semantics,
        "derivative_guarantees": spec.derivative_guarantees,
        "limitations": spec.limitations,
        "st_variant": spec.st_variant,
    }


def supported_relaxations(name: str | None = None) -> dict:
    """Return supported modes/methods (and defaults) for one or all operators."""
    if name is not None:
        return _spec_to_public_dict(get_operator(name))
    return {op_name: _spec_to_public_dict(spec) for op_name, spec in OPERATORS.items()}


def relaxation_limitations(name: str | None = None) -> dict[str, tuple[str, ...]]:
    """Return documented limitations for one or all operators."""
    if name is not None:
        return {name: get_operator(name).limitations}
    return {op_name: spec.limitations for op_name, spec in OPERATORS.items()}
