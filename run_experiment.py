import jax
import jax.numpy as jnp
import flax
import optax

from image_classification_jax.run_experiment import run_experiment

from flat_sophia import sophia_h


def main(project_to_flat: bool = False):
    base_lr = 0.003
    warmup = 512
    lr = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, base_lr, warmup),
            optax.linear_schedule(base_lr, 0.0, 9700 - warmup),
        ],
        boundaries=[warmup],
    )

    # only lets through kernel weights for weight decay
    kernels = flax.traverse_util.ModelParamTraversal(lambda p, _: "kernel" in p)

    def kernel_mask(params):
        all_false = jax.tree.map(lambda _: False, params)
        out = kernels.update(lambda _: True, all_false)
        return out

    opt = sophia_h(
        lr,
        b1=0.965,
        b2=0.99,
        eps=1e-8,
        weight_decay=0.3,
        mask=kernel_mask,
        gamma=0.01,
        clip_threshold=1.0,
        project_to_flat=project_to_flat,
        sharp_fraction=0.2,
        dampening_factor=10,
        mu_dtype=jnp.bfloat16,
        print_win_rate_every_n_steps=0,
    )

    run_experiment(
        log_to_wandb=True,
        wandb_entity="evanatyourservice",
        wandb_project="image_classification_jax",
        global_seed=100,
        dataset="cifar10",
        batch_size=512,
        n_epochs=100,
        optimizer=opt,
        compute_in_bfloat16=True,
        l2_regularization=0.0,
        randomize_l2_reg=False,
        apply_z_loss=True,
        model_type="vit",
        n_layers=4,
        enc_dim=512,
        n_heads=8,
        n_empty_registers=0,
        dropout_rate=0.0,
        using_schedule_free=False,  # set to True if optimizer wrapped with schedule_free
        psgd_calc_hessian=True,  # set to True if using PSGD and want to calc and pass in hessian
        psgd_precond_update_prob=0.1,
    )


if __name__ == "__main__":
    # run without projecting to flat, should overfit quite a bit
    main()

    # now run with project to flat, should keep it from overfitting as much
    main(project_to_flat=True)
