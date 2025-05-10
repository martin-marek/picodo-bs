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
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf.dictconfig import DictConfig


def loss_fn(model, x, pad=False): # [B, T]
    y = jnp.roll(x, -1, axis=1)
    loss_mask = data.pad_mask(x) if pad else jnp.ones(x.shape, dtype=bool)
    loss_mask = loss_mask.at[:, -1].set(False)
    logits = model(x)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return (losses * loss_mask).sum() / loss_mask.sum()


@partial(jax.jit, static_argnames='opt_graphdef')
def train_step(opt_graphdef, opt_state, batch):
    optimizer = nnx.merge(opt_graphdef, opt_state)
    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model, batch)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    return opt_state, loss


@partial(jax.jit, static_argnames=['opt_graphdef', 'pad'])
def eval_step(opt_graphdef, opt_state, dataset, pad=False):
    optimizer = nnx.merge(opt_graphdef, opt_state)
    losses = jax.lax.map(partial(loss_fn, optimizer.model, pad=pad), dataset)
    return losses.mean()


def train_and_evaluate(c: DictConfig):

    # get model and dataset rng seed
    key = jax.random.key(c.seed)
    key_model, key_dataset = jax.random.split(key)

    # sharding
    # all devices are aligned across a single mesh axis called 'data'
    # we use FSDP to shard data, model, and optimzier parameters across this axis
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = Mesh(create_device_mesh((num_fsdp_devices, c.num_tp_devices)), ('data', 'model'))
    print('sharding mesh:', ', '.join(f'{k}={v}' for k, v in mesh.shape.items()))
    assert c.model.N % c.num_tp_devices == 0

    # model
    model = model_lib.create_sharded_model(c.model, mesh, key_model)
    _, model_state = nnx.split(model)
    n_params = {
        'n_param_nonembed': 12 * c.model.L * c.model.D**2,
        'n_param_embed': c.model.D * c.model.V,
        'n_param_actual': utils.get_num_model_params(model),
    }
    for k, v in n_params.items():
        print(f'{k}={v:_}')

    # dataset
    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (n_params['n_param_nonembed'] + n_params['n_param_embed'])
    ds_train, ds_valid = data.load_ds(key_dataset, mesh, c.ds_path, c.model.T, c.opt.microbatch_size, c.num_tokens_valid, c.num_tokens_train)
    if (c.num_tokens_train is None): c.num_tokens_train = ds_train.size

    # optimizer
    num_microbatch_steps = len(ds_train)
    tokens_per_microbatch = c.opt.microbatch_size * c.model.T
    tx = optimizer_lib.get_optimizer(c.opt, model_state, num_microbatch_steps, tokens_per_microbatch)
    optimizer = nnx.Optimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name)
        wandb.summary.update(n_params)

    # training loop
    train_loss_sum, train_loss_num = jnp.zeros([]), 0
    with mesh:
        pbar = range(len(ds_train))
        if jax.process_index() == 0: pbar = tqdm(pbar)
        for step in pbar:
            
            # training step
            opt_state, batch_loss = train_step(opt_graphdef, opt_state, ds_train[step])
            train_loss_sum += batch_loss
            train_loss_num += 1

            # logging
            if (c.num_log_steps*(step+1)) % num_microbatch_steps < c.num_log_steps:
                metrics = {}
                metrics['train_loss'] = train_loss_sum / train_loss_num
                metrics['train_tokens_seen'] = (step+1) * tokens_per_microbatch
                metrics['learning_rate'] = opt_state.opt_state.hyperparams['learning_rate'].value
                if jax.process_index() == 0:
                    wandb.log(metrics, step)
                    pbar.set_postfix_str(f'loss={metrics["train_loss"]:.2f}')
                train_loss_sum, train_loss_num = jnp.zeros([]), 0

        # eval at end of training
        eval_loss = eval_step(opt_graphdef, opt_state, ds_valid, c.pad_eval)
        if jax.process_index() == 0:
            wandb.log({'eval_loss': eval_loss}, step)
            wandb.finish()
