from typing import Any, NamedTuple, Optional, Union, Callable

import jax
from jax import numpy as jnp
from optax._src import base
from optax._src import transform
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain
from optax import tree_utils as otu


class SophiaHState(NamedTuple):
    """State for Sophia-H and similar."""

    count: jax.Array  # shape=(), dtype=jnp.int32
    mu: base.Updates  # momentum
    nu: base.Updates  # EMA of hessian
    mask: Optional[base.Updates] = None


def scale_by_sophia_h(
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-8,
    gamma: float = 0.01,
    clip_threshold: Optional[float] = 1.0,
    project_to_flat: bool = False,
    sharp_fraction: float = 0.2,
    dampening_factor: int = 10,
    mu_dtype: Optional[Any] = None,
    print_win_rate_every_n_steps: int = 0,
) -> base.GradientTransformationExtraArgs:
    mu_dtype = canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree.map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu = jax.tree.map(jnp.zeros_like, params)
        if project_to_flat:
            mask = jax.tree.map(lambda x: jnp.ones_like(x, dtype=jnp.int8), params)
        else:
            mask = None
        return SophiaHState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, mask=mask)

    def update_fn(
        updates,
        state: SophiaHState,
        params=None,
        Hvp=None,
        vector=None,
        update_preconditioner=None,
    ):
        if params is None:
            raise ValueError("params must be provided to sophia's update function.")

        Hvp = jax.tree.map(lambda h, v: h * v, Hvp, vector)  # hutchinson

        nu = jax.lax.cond(
            update_preconditioner,
            lambda: otu.tree_update_moment(Hvp, state.nu, b2, 1),
            lambda: state.nu,
        )

        count_inc = safe_int32_increment(state.count)

        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        updates = jax.tree.map(lambda m, h: m / jnp.maximum(gamma * h, eps), mu_hat, nu)
        if clip_threshold is not None:
            sum_not_clipped = jax.tree.reduce(
                lambda x, y: x + y,
                jax.tree.map(lambda u: jnp.sum(jnp.abs(u) < clip_threshold), updates),
            )
            total_tree_size = sum(x.size for x in jax.tree.leaves(updates))
            win_rate = sum_not_clipped / total_tree_size
            jax.lax.cond(
                jnp.logical_and(
                    print_win_rate_every_n_steps > 0,
                    count_inc % print_win_rate_every_n_steps == 0,
                ),
                lambda: jax.debug.print("Sophia optimizer win rate: {}", win_rate),
                lambda: None,
            )

            updates = jax.tree.map(
                lambda u: jnp.clip(u, -clip_threshold, clip_threshold), updates
            )

        if project_to_flat:

            def make_mask():
                mask_in = Hvp
                flat_abs_hess = jnp.concatenate(
                    [jnp.ravel(jnp.abs(a)) for a in jax.tree.leaves(mask_in)]
                )
                pth_value = jax.lax.approx_max_k(
                    flat_abs_hess, int(sharp_fraction * flat_abs_hess.size)
                )[0][-1]
                new_mask = jax.tree.map(
                    lambda a: jnp.where(
                        jnp.abs(a) > pth_value,
                        jnp.array(dampening_factor, dtype=jnp.int8),
                        jnp.array(1, dtype=jnp.int8),
                    ),
                    mask_in,
                )
                return otu.tree_cast(new_mask, jnp.int8)

            new_mask = jax.lax.cond(
                update_preconditioner, make_mask, lambda: state.mask
            )

            updates = jax.tree.map(lambda u, m: u / m, updates, new_mask)
        else:
            new_mask = None

        mu = otu.tree_cast(mu, mu_dtype)
        state = SophiaHState(count=count_inc, mu=mu, nu=nu, mask=new_mask)
        return updates, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def sophia_h(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    gamma: float = 0.01,
    clip_threshold: Optional[float] = 1.0,
    project_to_flat: bool = False,
    sharp_fraction: float = 0.1,
    dampening_factor: int = 2,
    mu_dtype: Optional[Any] = None,
    print_win_rate_every_n_steps: int = 0,
) -> base.GradientTransformationExtraArgs:
    tx = [
        scale_by_sophia_h(
            b1=b1,
            b2=b2,
            eps=eps,
            gamma=gamma,
            clip_threshold=clip_threshold,
            project_to_flat=project_to_flat,
            sharp_fraction=sharp_fraction,
            dampening_factor=dampening_factor,
            mu_dtype=mu_dtype,
            print_win_rate_every_n_steps=print_win_rate_every_n_steps,
        ),
        transform.add_decayed_weights(weight_decay, mask=mask),
        transform.scale_by_learning_rate(learning_rate),
    ]
    return chain(*tx)
