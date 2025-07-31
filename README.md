# HyPoGen: Optimization-Biased Hypernetworks for Generalizable Policy Generation

This repository contains the official implementation of **HyPoGen: Optimization-Biased Hypernetworks for Generalizable Policy Generation**, accepted at ICLR 2025.

**Authors:** Hanxiang Ren, Li Sun, Xulong Wang, Pei Zhou, Zewen Wu, Siyan Dong, Difan Zou, Youyi Zheng, Yanchao Yang

**Paper URL:** https://openreview.net/forum?id=CJWMXqAnAy



## 🚀 Quick Start

### Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Make sure you have `uv` installed on your system.

```bash
# Clone the repository
git clone https://github.com/ReNginx/HyPoGen.git
cd HyPoGen

# Install dependencies using uv
uv sync
```

### Data Download

The data can be downloaded via the [link](https://drive.google.com/file/d/1RrgO_Jz-ZjeDBXTPN1t8snYwjIljLa_4/view?usp=drive_link). Please download and extract to the `rollout_dir` under the project root.


## 🧪 Running Experiments

### Training Models

The project supports multiple experiment configurations for different environments and settings.

#### MuJoCo Experiments

**Single-seed Training:**
```bash
# Train with reward variation
python train.py --config-name=mujoco_rew_exp

# Train hypernetworks with dynamics variation
python train.py --config-name=mujoco_dyn_exp

# Train with both reward and dynamics variation
python train.py --config-name=mujoco_rew_dyn_exp
```

**Multi-seed Training:**
```bash
# The configurations automatically run multiple seeds (123, 233, 666, 999, 789)
# across different domain tasks (cheetah_run, walker_walk, finger_spin)

python train.py --multirun --config-name=mujoco_rew_exp 
python train.py --multirun --config-name=mujoco_dyn_exp 
python train.py --multirun --config-name=mujoco_rew_dyn_exp 
```
note you might need to modify the `AVAILABLE_GPUS` in the config to match your hardware. 

#### ManiSkill2 Experiments

**Training:**
```bash
# Train hypernetworks on the maniskill dataset
python train.py --config-name=maniskill_train

# Generate weights of the policy networks for evaluation.
python train.py --config-name=maniskill_weightgen
```
### Testing Models

#### Mujoco Task Evaluation

```bash
# Evaluate trained approximators across multiple tasks and seeds
python batch_eval_regressor.py \
    --approximator_rootdir results_approximator \
    --domain_task_list cheetah_run finger_spin walker_walk \
    --exp_name_list rew_exp dyn_exp rew_dyn_exp \
    --seeds 123 233 666 789 999 \
    --n_episodes 10 \
    --output_dir batch_eval_results
```

change `--seeds` or `--exp_name_list` `--domain_task_list` if you want to evaluate part of the models.

#### ManiSkill2 Evaluation
make sure you run the weightgen command before evaluation.
```bash
# Evaluate ManiSkill2 policies with parameter variations
cd maniskill
bash eval_maniskill.sh
```

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

This work builds upon several open-source projects:
- [HyperZero](https://github.com/SAIC-MONTREAL/hyperzero) where a heavy bulk of code is borrowed from.
- [ManiSkill2](https://github.com/haosulab/ManiSkill2) for robotic manipulation environments
- [DMControl](https://github.com/deepmind/dm_control) for continuous control tasks
- [Hydra](https://hydra.cc/) for configuration management


## 📄 Citation

If you find this work useful, please consider cite

```bibtex
@inproceedings{
    ren2025hypogen,
    title={HyPoGen: Optimization-Biased Hypernetworks for Generalizable Policy Generation},
    author={Hanxiang Ren and Li Sun and Xulong Wang and Pei Zhou and Zewen Wu and Siyan Dong and Difan Zou and Youyi Zheng and Yanchao Yang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=CJWMXqAnAy}
}
```