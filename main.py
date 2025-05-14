import hydra
import train
from configs import resolver_setup
from omegaconf import OmegaConf, DictConfig
from utils import flatten_dict


# load default config
@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(c: DictConfig):

    # optionally load batch size config
    if 'bs_configs' in c:
        bs_config = c.bs_configs[f'bs{c.opt.batch_size}']
        for k, v in flatten_dict(bs_config).items():
            if OmegaConf.select(c, k) is None:
                OmegaConf.update(c, k, v)
        del c.bs_configs

    # optionally translate 1d scaling to generalized scaling config
    if 'scaling_1d' in c:
        OmegaConf.update(c, f'scaling.{c.scaling_1d.key}', c.scaling_1d.value, force_add=True)
        del c.scaling_1d
    
    # optionally apply generalized scaling config
    if 'scaling' in c:
        for k, scaling in flatten_dict(c.scaling).items():
            orig_val = OmegaConf.select(c, k)
            OmegaConf.update(c, k, scaling * orig_val)
        del c.scaling

    # run training job
    train.train_and_evaluate(c)


if __name__ == '__main__':
    main()
