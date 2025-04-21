import optax
from omegaconf import OmegaConf
import muon, multistep, utils


def get_optimizer(c: OmegaConf, params, num_microbatch_steps: int, tokens_per_microbatch: int):
    
    # get LR
    assert (c.peak_lr is not None) ^ ((c.peak_lr_scaled is not None) & (c.peak_lr_scaling is not None))
    if c.peak_lr is None:
        c.peak_lr = c.peak_lr_scaling * c.peak_lr_scaled

    # get schedule
    warmup_steps = int(c.warmup_frac * num_microbatch_steps)
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.peak_lr, warmup_steps, num_microbatch_steps)
    
    # gradient accumulation wrapper
    multistep_wrapper = multistep.singlesteps if c.grad_acc_steps==1 else multistep.multisteps

    # convert (t1 <-> b1), (t2 <-> b2)
    assert (c.b1 is None) | (c.t1 is None) # at most one can be specified in config
    assert (c.b2 is None) | (c.t2 is None) # at most one can be specified in config
    tokens_per_opt_step = c.grad_acc_steps * tokens_per_microbatch
    if c.b1 is None and c.t1 is not None: c.b1 = float(utils.halflife_to_decay(c.t1))
    if c.t1 is None and c.b1 is not None: c.t1 = float(utils.decay_to_halflife(c.b1))
    if c.b2 is None and c.t2 is not None: c.b2 = float(utils.halflife_to_decay(c.t2, tokens_per_opt_step))
    if c.t2 is None and c.b2 is not None: c.t2 = float(utils.decay_to_halflife(c.b2, tokens_per_opt_step))

    if c.optimizer == 'sgd':
        assert c.b2 is None
        assert c.t2 is None
        assert c.weight_decay == 0
        optimizer_factory = optax.inject_hyperparams(multistep_wrapper(optax.sgd, c.grad_acc_steps))
        optimizer = optimizer_factory(lr_schedule, c.b1)

    if c.optimizer == 'adamw':
        assert c.b1 is not None
        assert c.b2 is not None
        if c.b2_min is not None: c.b2 = max(c.b2, c.b2_min)
        optimizer_factory = optax.inject_hyperparams(multistep_wrapper(optax.adamw, c.grad_acc_steps))
        optimizer = optimizer_factory(lr_schedule, c.b1, c.b2, weight_decay=c.weight_decay)
    
    if c.optimizer == 'muon':
        assert c.b1 is not None
        assert c.b2 is None
        optimizer_factory = optax.inject_hyperparams(multistep_wrapper(muon.scale_by_muon, c.grad_acc_steps))
        optimizer = optimizer_factory(lr_schedule, beta=c.b1)

    return optimizer
