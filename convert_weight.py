from glob import glob
import io
import logging
from pathlib import Path
import os, sys
import zipfile

import hydra
from hydra.core.hydra_config import HydraConfig
import omegaconf
import torch
from utils import utils
from utils.dataloader import FastTensorDataLoader
from utils.dataset import RoboArmDataset


def make_approximator(input_dim, state_dim, action_dim, cfg, device=None):
    cfg.input_dim = input_dim
    cfg.state_dim = state_dim
    cfg.action_dim = action_dim
    if device is not None:
        cfg.device = device
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path(HydraConfig.get().runtime.output_dir)

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # hacked up way to see if we are using MAML or not
        self.is_meta_learning = (
            True
            if "meta" in self.cfg.approximator_name
            or "pearl" in self.cfg.approximator_name
            else False
        )

        self.setup()

        input_dim = self.dataset.dynamic_param_dim

        self.input_to_model = cfg.input_to_model

        self.approximator = make_approximator(
            input_dim,
            self.dataset.state_dim,
            self.dataset.action_dim,
            self.cfg.approximator,
        )

        self.timer = utils.Timer()

    def setup(self):
        # create logger
        self.model_dir = self.work_dir / "models"
        self.model_dir.mkdir(exist_ok=True)

        self.rollout_dir = Path(
            self.cfg.rollout_dir
        ).expanduser()  # .joinpath(self.cfg.domain_task)

        # load dataset
        self.load_dataset()

    def load_dataset(self):
        dataset_fn = RoboArmDataset
        dataloader_fn = FastTensorDataLoader

        self.dataset = dataset_fn(
            self.rollout_dir,
            self.cfg.domain_task,
            self.cfg.input_to_model,
            self.cfg.seed,
            self.device,
            self.cfg.test_fraction,
        )

        batch_size = self.cfg.batch_size

        self.train_loader = dataloader_fn(
            *self.dataset.train_dataset[:],
            device=self.device,
            batch_size=batch_size,
            shuffle=True,
        )
        self.test_loader = dataloader_fn(
            *self.dataset.test_dataset[:],
            device=self.device,
            batch_size=batch_size,
            shuffle=True,
        )

        train_spd_txt = os.path.join(
            self.rollout_dir,
            f"train-{self.cfg.domain_task}-{self.cfg.input_to_model}-params-seed-{self.cfg.seed}.txt",
        )
        test_spd_txt = os.path.join(
            self.rollout_dir,
            f"test-{self.cfg.domain_task}-{self.cfg.input_to_model}-params-seed-{self.cfg.seed}.txt",
        )

        open(train_spd_txt, "w").write(str(self.dataset.train_files))
        open(test_spd_txt, "w").write(str(self.dataset.test_files))


def convert_weights(input_params, weights, base_ckpt_path, work_dir):
    valid_idx = []
    for i in range(len(input_params)):
        path = os.path.join(work_dir, f"input_param{input_params[i].item():.3f}.pth")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"============= {path} already exists, skip")
        else:
            valid_idx.append(i)
    if len(valid_idx) == 0:
        return
    base_ckpt = zipfile.ZipFile(base_ckpt_path, "r")
    base_ckpt_params = io.BytesIO(base_ckpt.read("policy.pth"))
    base_ckpt_params = torch.load(base_ckpt_params)

    for i in valid_idx:

        def assign_weight(base_key, weight_key):
            for typ in [".weight", ".bias"]:
                assert (
                    base_ckpt_params[base_key + typ].shape
                    == weights[weight_key + typ][i].shape
                )
                base_ckpt_params[base_key + typ] = (
                    weights[weight_key + typ][i].detach().cpu()
                )

        for j in range(1, 3):
            base_key = f"mlp_extractor.policy_net.{(j-1)*2}"
            assign_weight(base_key, f"fc{j}")
        assign_weight("action_net", "fc3")

        with zipfile.ZipFile(
            os.path.join(work_dir, f"input_param{input_params[i].item():.3f}.pth"), "w"
        ) as fout:
            for f in base_ckpt.filelist:
                if f.filename == "policy.pth":
                    buffer = io.BytesIO()
                    torch.save(base_ckpt_params, buffer)
                    buffer.seek(0)
                    bytes_str = buffer.read()
                    fout.writestr(f.filename, bytes_str)
                else:
                    fout.writestr(f.filename, base_ckpt.read(f.filename))


@hydra.main(version_base=None, config_path="cfgs", config_name="weight_gen")
def main(cfg):
    log = logging.getLogger(__name__)
    AVAILABLE_GPUS = cfg.get("AVAILABLE_GPUS")
    print("AVAILABLE_GPUS", AVAILABLE_GPUS)
    try:
        device_id = AVAILABLE_GPUS[HydraConfig.get().job.num % len(AVAILABLE_GPUS)]
        cfg.device = f"{cfg.device}:{device_id}"
        log.info(f"Total number of GPUs is {AVAILABLE_GPUS}, running on {cfg.device}.")
    except omegaconf.errors.MissingMandatoryValue:
        pass

    workspace = Workspace(cfg)
    work_dir = str(workspace.work_dir)
    glob_path = os.path.join(
        work_dir.replace(cfg.experiment, cfg.ckpt_path),
        # '*',
        "models",
        f"step_{cfg.load_step:08d}",
        "*",
    )
    print(glob_path)
    cfg.ckpt_path = glob(glob_path)[0]
    print(f"============= ckpt path is ", cfg.ckpt_path)
    approximator = workspace.approximator

    approximator.load(cfg.ckpt_path, cfg.load_step)
    # input_params = torch.tensor(cfg.input_params).to(cfg.device)[..., None]
    if cfg.input_to_model == "cube":
        input_params = (
            torch.arange(0.01, 0.031, 0.002).float().to(cfg.device)[..., None]
        )
    elif cfg.input_to_model == "stiff":
        input_params = torch.arange(500, 1550, 100).float().to(cfg.device)[..., None]
    elif cfg.input_to_model == "damp":
        input_params = torch.arange(50, 155, 10).float().to(cfg.device)[..., None]
    elif cfg.input_to_model == "length":
        input_params = torch.arange(0.5, 2.1, 0.1).float().to(cfg.device)[..., None]

    if workspace.is_meta_learning:
        weights = approximator.forward_weights(input_params, 1024, workspace.dataset)
    else:
        with torch.no_grad():
            weights = approximator.forward_weights(input_params)

    cfg.base_ckpt_path = cfg.base_ckpt_path.replace("LiftCube", cfg.domain_task)
    print(f"============= {cfg.domain_task} base ckpt path is ", cfg.base_ckpt_path)
    convert_weights(input_params, weights, cfg.base_ckpt_path, workspace.model_dir)


if __name__ == "__main__":
    main()
