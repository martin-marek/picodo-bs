import hydra
import train
from configs import resolver_setup
from omegaconf import OmegaConf, DictConfig


# load default config
@hydra.main(version_base=None, config_path='configs')
def main(c: DictConfig):

    # overwrite batch size params
    c = OmegaConf.merge(c, c.ablation.bs_config[c.opt.batch_size])
    del c.ablation.bs_config

    # scaled ablated hparam
    OmegaConf.update(c, c.ablation.hparam, c.ablation.scaling * OmegaConf.select(c, c.ablation.hparam))

    # run training job
    train.train_and_evaluate(c)


if __name__ == '__main__':
    main()
