import operator as op
import hydra
import train
from omegaconf import OmegaConf, DictConfig
OmegaConf.register_new_resolver('floordiv', op.floordiv)
OmegaConf.register_new_resolver('mul', op.mul)
OmegaConf.register_new_resolver('min', min)
OmegaConf.register_new_resolver('pow', pow)


import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


@hydra.main(version_base=None, config_path='configs', config_name='local')
def main(c: DictConfig):
    train.train_and_evaluate(c)


if __name__ == '__main__':
    main()
