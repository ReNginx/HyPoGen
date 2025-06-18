"""
RL approximator training loop.
Works on DMC with states and pixel observations.
"""
import re
import warnings

import numpy as np
from omegaconf import OmegaConf

from utils import dmc

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import platform
import logging
import math

if platform.system() == 'Linux':
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from pathlib import Path
import pathlib

import hydra
import omegaconf
import torch
from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter

import utils.utils as utils
from utils.dataset import RLSolutionDataset, RLSolutionMetaDataset
from utils.dataloader import FastTensorDataLoader, FastTensorMetaDataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import io

torch.backends.cudnn.benchmark = True

# If using multirun, set the GPUs here:
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]


def make_approximator(input_dim, state_dim, action_dim, cfg, device=None):
    cfg.input_dim = input_dim
    cfg.state_dim = state_dim
    cfg.action_dim = action_dim
    if device is not None:
        cfg.device = device
    return hydra.utils.instantiate(cfg)

def parse_split_txt(path, input_to_model):
    splits = eval(open(path, 'r').read().strip())
    if input_to_model == 'rew':
        splits = [float(re.findall('linear-(-?\d+\.\d+)', s)[0]) for s in splits]
    elif input_to_model == 'dyn':
        splits = [float(re.findall('(-?\d+\.\d+)', s)[0]) for s in splits]
    elif input_to_model == 'rew_dyn':
        spd = [float(re.findall('linear-(-?\d+\.\d+)', s)[0]) for s in splits]
        dyn = [float(re.findall('dyn_(-?\d+\.\d+)', s)[0]) for s in splits]
        splits = [s for s in zip(spd, dyn)]
    splits.sort()
    return splits

def draw_reward_curve(reward_dict):
    # Initialize a figure and axis
    fig, ax = plt.subplots()

    # Separate the dictionary into two: one for mean and one for std
    mean_dict = {k: v for k, v in reward_dict.items() if 'mean' in k}
    std_dict = {k: v for k, v in reward_dict.items() if 'std' in k}

    # Sort the dictionaries by speed
    mean_dict = dict(sorted(mean_dict.items(), key=lambda item: float(item[0].split('@')[-1])))
    std_dict = dict(sorted(std_dict.items(), key=lambda item: float(item[0].split('@')[-1])))

    # Plot the mean and std against speed with error bars
    speeds = [float(k.split('@')[-1]) for k in mean_dict.keys()]
    means = list(mean_dict.values())
    stds = list(std_dict.values())
    ax.plot(speeds, means, '-o')
    ax.fill_between(speeds, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.2)


    # Set the title of the graph to the partition name and show the legend
    partition = list(mean_dict.keys())[0].split('/')[0]
    ax.set_title(partition)
    ax.legend(['mean', 'std'])

    ax.set_ylim([0, 1000])
    ax.set_xlim([-10, 10])

    # Convert the matplotlib figure to a PyTorch tensor
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    transform = transforms.ToTensor()
    img_tensor = transform(img)

    # Close the figure to free memory
    plt.close(fig)

    return img_tensor


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # hacked up way to see if we are using MAML or not
        self.is_meta_learning = True if 'meta' in self.cfg.approximator_name or 'pearl' in self.cfg.approximator_name else False

        self.setup()

        if cfg.input_to_model == 'rew':
            input_dim = self.dataset.reward_param_dim
        elif cfg.input_to_model == 'dyn':
            input_dim = self.dataset.dynamic_param_dim
        elif cfg.input_to_model == 'rew_dyn':
            input_dim = self.dataset.reward_dynamic_param_dim
        else:
            raise NotImplementedError

        self.input_to_model = cfg.input_to_model

        self.approximator = make_approximator(input_dim,
                                              self.dataset.state_dim,
                                              self.dataset.action_dim,
                                              self.cfg.approximator)

        rg_train = np.random.RandomState(self.cfg.seed)
        rg_test = np.random.RandomState(3)

        self.reward_parameters = OmegaConf.to_container(self.cfg.reward_parameters)

        # get the dynamics parameters
        try:
            self.dynamics_parameters = OmegaConf.to_container(self.cfg.dynamics_parameters)
        except omegaconf.errors.ConfigAttributeError:
            self.dynamics_parameters = {'use_default': True}
        self.rl_agent_fn = lambda reward_params, dynamics_params: dmc.make(self.cfg.domain_task, 1,
                                          1, reward_params,
                                 dynamics_params, rg_train, False)
        self.rl_agent_fn_test = lambda reward_params, dynamics_params: dmc.make(self.cfg.domain_task, 1,
                                          1, reward_params,
                                 dynamics_params, rg_test, False)
        
        self.timer = utils.Timer()
        self._global_epoch = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = SummaryWriter(str(self.work_dir))

        self.model_dir = self.work_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)

        self.rollout_dir = Path(self.cfg.rollout_dir).expanduser().joinpath(self.cfg.domain_task)

        # load dataset
        self.load_dataset()

        # save cfg and git sha
        utils.save_cfg(self.cfg, self.work_dir)
        utils.save_git_sha(self.work_dir)

        code_dir = pathlib.Path(__file__).parent.resolve()
        utils.save_code(code_dir, self.work_dir)

    def load_dataset(self):
        dataset_fn = RLSolutionMetaDataset if self.is_meta_learning else RLSolutionDataset
        dataloader_fn = FastTensorMetaDataLoader if self.is_meta_learning else FastTensorDataLoader
        train_fraction = self.cfg.train_fraction if hasattr(self.cfg, 'train_fraction') else None

        self.dataset = dataset_fn(
            self.rollout_dir,
            self.cfg.domain_task,
            self.cfg.input_to_model,
            self.cfg.seed,
            self.device,
            self.cfg.test_fraction,
            train_fraction,
        )

        if self.is_meta_learning:
            batch_size = int(self.dataset.n_tasks * self.cfg.k_shot * 2)
        else:
            batch_size = self.cfg.batch_size

        self.train_loader = dataloader_fn(*self.dataset.train_dataset[:], device=self.device,
                                          batch_size=batch_size, shuffle=True)
        self.test_loader = dataloader_fn(*self.dataset.test_dataset[:], device=self.device,
                                         batch_size=batch_size, shuffle=True)

        train_spd_txt = os.path.join(self.rollout_dir, f'train-{self.cfg.input_to_model}-params-seed-{self.cfg.seed}.txt')
        test_spd_txt = os.path.join(self.rollout_dir, f'test-{self.cfg.input_to_model}-params-seed-{self.cfg.seed}.txt')

        self.train_speeds = parse_split_txt(train_spd_txt, self.cfg.input_to_model)
        self.test_speeds = parse_split_txt(test_spd_txt, self.cfg.input_to_model)

    @property
    def global_epoch(self):
        return self._global_epoch

    def train(self):
        # predicates
        train_until_epoch = utils.Until(self.cfg.num_train_epochs)
        save_every_epoch = utils.Every(self.cfg.save_every_frames)

        metrics = dict()
        # best_valid_total_loss = math.inf
        # best_valid_value_loss = math.inf
        # best_valid_action_loss = math.inf
        # best_valid_td_loss = math.inf
        best_reward_loss = -math.inf

        while train_until_epoch(self.global_epoch):
            metrics.update()

            if self.is_meta_learning:
                self.train_loader.shuffle_indices()
                self.test_loader.shuffle_indices()

            save_this_epoch = save_every_epoch(self.global_epoch + 1)

            metrics.update(self.approximator.update(self.train_loader))

            if save_this_epoch:
                train_eval_dict, mean_reward_train = self.approximator.eval_env(self.rl_agent_fn_train, 3, self.train_speeds,
                                                                                self.reward_parameters, self.dynamics_parameters,
                                                                                self.input_to_model, 'train')
                test_eval_dict, mean_reward_test = self.approximator.eval_env(self.rl_agent_fn_test, 10, self.test_speeds,
                                                                              self.reward_parameters, self.dynamics_parameters,
                                                                              self.input_to_model, 'test')
                metrics.update({'train/mean_reward': mean_reward_train})
                metrics.update({'test/mean_reward': mean_reward_test})
                try:
                    self.logger.add_image('train/reward_curve', draw_reward_curve(train_eval_dict), self.global_epoch + 1)
                    self.logger.add_image('test/reward_curve', draw_reward_curve(test_eval_dict), self.global_epoch + 1)
                except Exception as err:
                    pass
                torch.save(train_eval_dict, os.path.join(self.work_dir, f'train_std_reward_{self.global_epoch:03d}.pth'))
                torch.save(test_eval_dict, os.path.join(self.work_dir, f'test_std_reward_{self.global_epoch:03d}.pth'))
                 
                if mean_reward_test > best_reward_loss:
                    best_reward_loss = mean_reward_test
                    self.approximator.save(self.model_dir, 'best_reward')
                    metrics.update({'test/best_reward': mean_reward_test})

            # metrics.update(self.approximator.eval(self.test_loader))

            # Log metrics
            print(f"Epoch {self.global_epoch + 1} "
                  f"\t Train loss {metrics['train/loss_total']:.3f} ")
                  # f"\t Valid loss {metrics['valid/loss_total']:.3f}")

            for k, v in metrics.items():
                self.logger.add_scalar(k, v, self.global_epoch + 1)
            utils.dump_dict(f"{self.work_dir}/train_valid.csv", metrics)

            # Save the model
            # if metrics['valid/loss_total'] <= best_valid_total_loss:
            #     best_valid_total_loss = metrics['valid/loss_total']
            #     self.approximator.save(self.model_dir, 'best_total')
            # if metrics['valid/loss_action_pred'] <= best_valid_action_loss:
            #     best_valid_action_loss = metrics['valid/loss_action_pred']
            #     self.approximator.save(self.model_dir, 'best_action')
            # if 'valid/loss_value_pred' in metrics:
            #     if metrics['valid/loss_value_pred'] <= best_valid_value_loss:
            #         best_valid_value_loss = metrics['valid/loss_value_pred']
            #         self.approximator.save(self.model_dir, 'best_value')
            # if 'valid/loss_td' in metrics:
            #     if metrics['valid/loss_td'] <= best_valid_td_loss:
            #         best_valid_td_loss = metrics['valid/loss_td']
            #         self.approximator.save(self.model_dir, 'best_td')

            self._global_epoch += 1
            if save_every_epoch(self.global_epoch):
                self.approximator.save(self.model_dir, self.global_epoch)
                self.save_snapshot()

            

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        print(f'Save Snap shot to {snapshot}')
        torch.save({
            'timer': self.timer,
            '_global_epoch': self._global_epoch,
            '_global_episode': self._global_episode,
            '_model_state_dict': self.approximator.get_model().state_dict(),
            '_model_optim_state_dict':  self.approximator.get_optim().state_dict(),
        }, self.work_dir / 'snapshot.pt')

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        print(f'Loading Snap shot from {snapshot}')
        payload = torch.load(self.work_dir / 'snapshot.pt')
        self.timer = payload['timer']
        self._global_epoch = payload['_global_epoch']
        self._global_episode = payload['_global_episode']
        print(f'Loading Epoch from {self._global_epoch}')
        self.approximator.get_model().load_state_dict(payload['_model_state_dict'])
        self.approximator.get_optim().load_state_dict(payload['_model_optim_state_dict'])


@hydra.main(version_base=None, config_path='cfgs', config_name='config_rl_approximator')
def main(cfg):
    log = logging.getLogger(__name__)
    AVAILABLE_GPUS = cfg.get('AVAILABLE_GPUS')
    print('AVAILABLE_GPUS', AVAILABLE_GPUS)
    try:
        device_id = AVAILABLE_GPUS[HydraConfig.get().job.num % len(AVAILABLE_GPUS)]
        cfg.device = f"{cfg.device}:{device_id}"
        log.info(f"Total number of GPUs is {AVAILABLE_GPUS}, running on {cfg.device}.")
    except omegaconf.errors.MissingMandatoryValue:
        pass

    workspace = Workspace(cfg)
    snapshot = workspace.work_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
