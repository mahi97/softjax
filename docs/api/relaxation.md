# Relaxation policy and registry

SoftJAX operators share a single internal description of how they are relaxed.
A [`RelaxationPolicy`][softjax.RelaxationPolicy] carries the mode, method,
softness, straight-through flag and numerical options. An operator registry
lists every public operator, its hard JAX counterpart, supported methods and
the derivative guarantees / limitations that apply.

Existing call signatures are unchanged. Explicit arguments always win over a
policy. A policy is used only for fields the caller did not pass.

```python
import jax.numpy as jnp
import softjax as sj

x = jnp.linspace(-1.0, 1.0, 6)

with sj.relaxation_scope(sj.RelaxationPolicy(mode="c2", softness=0.2)):
    # uses mode="c2", softness=0.2
    y = sj.sort(x)
    # explicit method wins; mode still comes from the scope
    z = sj.sort(x, method="softsort")

sj.supported_relaxations("quantile")
sj.relaxation_limitations("quantile")
```

Nested scopes merge: the inner policy overrides only the fields it sets.

```python
outer = sj.RelaxationPolicy(mode="c1", softness=0.5, method="softsort")
with sj.relaxation_scope(outer):
    with sj.relaxation_scope(sj.RelaxationPolicy(softness=0.1)):
        # mode="c1", method="softsort", softness=0.1
        sj.median(x)
```

`straight_through=True` uses a hard forward value and the requested soft mode
for gradients, matching [`softjax.st`][].

::: softjax.RelaxationPolicy

::: softjax.relaxation_scope

::: softjax.get_policy

::: softjax.supported_relaxations

::: softjax.relaxation_limitations
