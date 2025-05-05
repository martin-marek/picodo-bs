import hydra
import train
from configs import resolver_setup
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(c: DictConfig):
    train.train_and_evaluate(c)


if __name__ == '__main__':
    main()
