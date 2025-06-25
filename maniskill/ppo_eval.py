import os
import sys
import gym
import numpy as np
import mani_skill2.envs
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from mani_skill2.utils.wrappers import RecordEpisode
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from mani_skill2.utils.sapien_utils import check_actor_static


class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps: int) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self.pre_obs = None
        self._max_episode_steps = max_episode_steps

    def reset(self):
        self._elapsed_steps = 0
        self.pre_obs = super().reset()
        return self.pre_obs

    def compute_dense_reward(self, action):
        assert 0

    def step(self, action):
        ob, rew, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False
            info["TimeLimit.truncated"] = False
        return ob, rew, done, info


class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, done, info = super().step(action)
        info["is_success"] = info["success"]
        if info["success"]:
            done = True
        return ob, rew, done, info


def make_env(
    env_id,
    max_episode_steps: int = None,
    record_dir: str = None,
    cube_size: float = 0.02,
    agent_stiffness: int = 1e3,
    agent_damping: int = 1e2,
    urdf_path: str = None,
):
    def _init() -> gym.Env:
        import mani_skill2.envs

        env = gym.make(
            env_id,
            obs_mode="state",
            reward_mode="dense",
            control_mode=control_mode,
            cube_half_size=cube_size,
            agent_stiffness=agent_stiffness,
            agent_damping=agent_damping,
            urdf_path=urdf_path,
        )
        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env, max_episode_steps)
        if record_dir is not None:
            env = SuccessInfoWrapper(env)
            if "eval" in record_dir:
                env = RecordEpisode(
                    env, record_dir, info_on_video=False, render_mode="cameras"
                )
        return env

    return _init


# environment list
franka_list = ["LiftCube-v0", "PickCube-v0", "StackCube-v0"]
mobile_list = []


def append_to_file(filename, text_to_append):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as file:
        file.write(f"{text_to_append}\n")


if __name__ == "__main__":
    # add and parse argument
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_id", type=str, default="LiftCube-v0")
    parser.add_argument("--train_num", type=int, default=8)
    parser.add_argument("--eval_num", type=int, default=5)
    parser.add_argument("--eval_freq", type=int, default=12800)
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--rollout_steps", type=int, default=3200)
    parser.add_argument("--train_max_steps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=1)
    parser.add_argument("--project_name", type=str, default="maniskill")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--cube_size", type=float, default=0.02)
    parser.add_argument("--urdf_path", type=str, default=None)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--agent_stiffness", type=float, default=1e3)
    parser.add_argument("--agent_damping", type=float, default=1e2)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--rollout_path", type=str, default=None)

    args = parser.parse_args()
    # args.env_id = "StackCube-v0"
    method = "ppo"

    # if args.exp_name == "":
    args.exp_name = f"{method}_eval/{args.exp_name}"

    factor = args.length
    if factor != 1.0:
        args.urdf_path = (
            f"ManiSkill2/mani_skill2/assets/descriptions/panda_v2_{factor:.1f}.urdf"
        )
    else:
        args.urdf_path = f"ManiSkill2/mani_skill2/assets/descriptions/panda_v2.urdf"

    task_name = (
        args.env_id
        + f"_cube{args.cube_size:.3f}_stiff{args.agent_stiffness:.0f}_damp{args.agent_damping:.0f}_length{factor:.1f}"
    )
    log_path = f"logs/{args.exp_name}/{task_name}/log.txt"

    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        print(f"Task {task_name} has been evaluated!")
        sys.exit(0)

    if args.env_id in franka_list + mobile_list:
        if args.env_id in franka_list:
            control_mode = "pd_ee_delta_pose"
        elif args.env_id in mobile_list:
            control_mode = "base_pd_joint_vel_arm_pd_ee_delta_pose"
        else:
            assert 0
    else:
        print("Please specify a valid environment!")
        assert 0

    # initialize wandb
    # run = wandb.init(project=args.project_name)

    # set up eval environment

    # set up callback
    set_random_seed(args.seed)

    # set up sac algorithm
    policy_kwargs = dict(net_arch=[256, 256])
    # policy_kwargs = dict(net_arch=[3200, 3200, 3200, 3200])
    model = PPO.load(
        path=args.ckpt_path,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=args.rollout_steps // args.train_num,
        batch_size=400,
        n_epochs=15,
        tensorboard_log=f"./logs/{args.exp_name}/{task_name}",
        gamma=0.85,
        target_kl=0.05,
    )

    # set up model evaluation environment
    record_dir = (
        f"logs/{args.exp_name}/{task_name}/eval_videos_" + args.env_id[:-3] + "-our"
    )
    eval_env = SubprocVecEnv(
        [
            make_env(
                args.env_id,
                record_dir=record_dir,
                cube_size=args.cube_size,
                agent_stiffness=args.agent_stiffness,
                agent_damping=args.agent_damping,
                urdf_path=args.urdf_path,
            )
            for i in range(1)
        ]
    )
    eval_env = VecMonitor(eval_env)
    eval_env.seed(args.eval_seed)
    eval_env.reset()

    # model evaluation and save video
    returns, ep_lens = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=False,
        return_episode_rewards=True,
        n_eval_episodes=100,
    )

    # # create a dir on wandb to store the videos, copy these to wandb
    # os.makedirs(f"{wandb.run.dir}/video/{run.id}", exist_ok=True)
    # os.system(f"cp -r {record_dir} {wandb.run.dir}/video/{run.id}")

    env = gym.make(args.env_id, obs_mode="state", reward_mode="dense")
    success = np.array(ep_lens) < env._max_episode_steps
    success_rate = success.mean()
    print(f"Success Rate: {success_rate}")
    print(f"Episode Lengths: {ep_lens}")
    append_to_file(
        log_path, f"Success Rate: {success_rate};\nEpisode Lengths: {ep_lens}"
    )
