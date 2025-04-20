import jax
import jax.numpy as jnp
import optax
import wandb
import data, utils
import model as model_lib
import optimizer as optimizer_lib
from functools import partial
from flax import nnx
from tqdm.auto import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf.dictconfig import DictConfig


def loss_fn(model, batch):
    x, y, weights = data.get_in_out(batch)
    logits = model(x)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    mean_loss = jnp.sum(losses * weights) / weights.sum()
    return mean_loss


@partial(jax.jit, static_argnames='opt_graphdef')
def train_step(opt_graphdef, opt_state, batch):
    optimizer = nnx.merge(opt_graphdef, opt_state)
    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model, batch)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    lr = optimizer.opt_state.hyperparams['learning_rate'].value
    metrics = {'train_loss': loss, 'learning_rate': lr}
    return opt_state, metrics


@partial(jax.jit, static_argnames='model_graphdef')
def eval_step(model_graphdef, model_state, dataset):
    model = nnx.merge(model_graphdef, model_state)
    losses = jax.lax.map(partial(loss_fn, model), dataset)
    return {'eval_loss': losses.mean()}


def train_and_evaluate(c: DictConfig):

    # get model and dataset rng seed
    key = jax.random.key(c.seed)
    seed_model, seed_dataset = jax.random.randint(key, [2], 0, 1_000_000)

    # sharding
    # all devices are aligned across a single mesh axis called 'data'
    # we use FSDP to shard data, model, and optimzier parameters across this axis
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ('data',))
    data_sharding = NamedSharding(mesh, P('data')) # data parallelism

    # model
    model = model_lib.create_sharded_model(c.model, mesh, seed_model)
    model_graphdef, model_state = nnx.split(model)
    n_param = utils.get_num_model_params(model)
    print(f'{n_param=:_}')

    # dataset
    if c.num_tokens_train is None:
        c.num_tokens_train = ds_train_size if c.tokens_params_ratio is None else n_param * c.tokens_params_ratio
    get_batch, idx_train, idx_valid = data.load_ds(c.ds_path, c.model.L, c.opt.microbatch_size, c.batch_size_valid, c.num_tokens_valid, c.num_tokens_train, seed_dataset)
    with mesh: ds_valid = jnp.stack([jax.device_put(get_batch(idx), data_sharding) for idx in idx_valid])

    # optimizer
    num_microbatch_steps = len(idx_train)
    tokens_per_microbatch = c.opt.microbatch_size * c.model.L
    tx = optimizer_lib.get_optimizer(c.opt, model_state, num_microbatch_steps, tokens_per_microbatch)
    optimizer = nnx.Optimizer(model, tx)

    # start wandb
    if c.wandb_project is not None:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name)
        wandb.summary.update(dict(n_param=n_param))

    # training loop
    # note: metrics for each steps are processed only after asynchronously dispatching the next step
    pending_train_metrics = None
    pending_eval_metrics = None
    opt_graphdef, opt_state = nnx.split(optimizer)
    with mesh:
        pbar = tqdm(enumerate(idx_train))
        for step, seq_idx in pbar:

            # training step
            batch = jax.device_put(get_batch(seq_idx), data_sharding)
            opt_state, train_metrics = train_step(opt_graphdef, opt_state, batch)
            train_metrics |= {'train_tokens_seen': (step+1)*tokens_per_microbatch}

            # async logging
            if pending_train_metrics is not None:
                pbar.set_postfix_str(f'loss={pending_train_metrics["train_loss"]:.2f}')
                wandb.log(pending_train_metrics, step-1)
            pending_train_metrics = train_metrics
            if pending_eval_metrics is not None:
                wandb.log(pending_eval_metrics, step-1)
                pending_eval_metrics = None

            # eval step
            eval_every_steps = len(idx_train) // c.num_eval_steps
            if ((step+1) % eval_every_steps == 0) or ((step+1) == num_microbatch_steps):
                pending_eval_metrics = eval_step(model_graphdef, opt_state.model, ds_valid)

        wandb.log(pending_train_metrics, step)
        wandb.log(pending_eval_metrics, step)
