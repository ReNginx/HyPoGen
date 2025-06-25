"""
Efficient batch evaluation script for RL approximators.
Automatically finds best models and evaluates them using batch processing.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import platform

if platform.system() == "Linux":
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    os.environ["MUJOCO_GL"] = "egl"

import argparse
from pathlib import Path
import json
import ast
import numpy as np
import torch
from omegaconf import OmegaConf
from collections import defaultdict
from tqdm import tqdm
import csv
from joblib import Parallel, delayed

import utils.dmc as dmc
import utils.utils as utils
from train_rl_regressor import make_approximator

torch.backends.cudnn.benchmark = True


class BatchEvaluator:
    def __init__(self, args):
        self.args = args

        # Set up device
        if args.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)

        print(f"Using device: {self.device}")

        # Print GPU info if using CUDA
        if "cuda" in str(self.device) and torch.cuda.is_available():
            gpu_id = str(self.device).split(":")[-1] if ":" in str(self.device) else "0"
            gpu_name = torch.cuda.get_device_name(int(gpu_id))
            gpu_memory = (
                torch.cuda.get_device_properties(int(gpu_id)).total_memory / 1024**3
            )
            print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")

        utils.set_seed_everywhere(3)  # Fixed seed for evaluation

        # Results storage
        self.results = defaultdict(list)

    def find_best_models(self):
        """Find all model checkpoints for the given configuration and step."""
        approx_root_dir = Path(self.args.approximator_rootdir)
        model_paths = []

        # Determine the step pattern to search for
        if self.args.step_to_load == "best":
            step_pattern = "step_best_reward"
        else:
            step_pattern = f"step_{int(self.args.step_to_load):08d}"

        for domain_task in self.args.domain_task_list:
            for exp_name in self.args.exp_name_list:
                pattern = (
                    f"{exp_name}/**/**/*{domain_task}*/**/**/models/{step_pattern}"
                )
                approx_paths = sorted(approx_root_dir.glob(pattern))

                for approx_path in approx_paths:
                    workdir = approx_path.parents[1]
                    method = approx_path.parents[4].name
                    input_to_model = approx_path.parents[3].name
                    seed = approx_path.parents[1].name.split("_")[1]

                    # Filter by seeds if specified
                    if self.args.seeds and int(seed) not in self.args.seeds:
                        continue

                    model_info = {
                        "workdir": workdir,
                        "method": method,
                        "seed": int(seed),
                        "domain_task": domain_task,
                        "exp_name": exp_name,
                        "model_path": approx_path,
                        "input_to_model": input_to_model,
                        "step_to_load": self.args.step_to_load,
                    }
                    model_paths.append(model_info)

        print(
            f"Found {len(model_paths)} models to evaluate (step: {self.args.step_to_load})"
        )
        return model_paths

    def get_test_parameters(self, domain_task, exp_name, seed):
        """Get test parameters for the given configuration."""
        if exp_name == "rew_exp":
            rollout_data_folder = (
                f"./rollout_data/rollout_data_grid_v4_rew/{domain_task}"
            )
        elif exp_name == "dyn_exp":
            rollout_data_folder = (
                f"./rollout_data/rollout_data_grid_v4_dyn/{domain_task}"
            )
        elif exp_name == "rew_dyn_exp":
            rollout_data_folder = (
                f"./rollout_data/rollout_data_grid_v4_rew_dyn/{domain_task}"
            )
        else:
            assert False, f"Unknown exp_name: {exp_name}"

        rollout_data_folder = Path(rollout_data_folder)

        try:
            test_list_file = sorted(rollout_data_folder.glob(f"test*{seed}*.txt"))[0]
            with open(str(test_list_file), "r") as file:
                content = file.read()
            test_list = ast.literal_eval(content)

            test_values = []
            for test_name in test_list:
                if exp_name == "rew_dyn_exp":
                    # Handle combined reward-dynamics parameters
                    if "_dyn_" in str(test_name):
                        # Format: "rew_value_dyn_dyn_value"
                        parts = str(test_name).split("_dyn_")
                        rew_part = float(parts[0])
                        dyn_part = float(parts[1])
                        test_value = [rew_part, dyn_part]
                    else:
                        # Fallback parsing for different formats
                        try:
                            # Try to parse as tuple string like "(1.0, 2.0)"
                            test_value = ast.literal_eval(test_name)
                            if (
                                not isinstance(test_value, (list, tuple))
                                or len(test_value) != 2
                            ):
                                raise ValueError("Invalid rew_dyn format")
                        except:
                            # Skip malformed entries
                            continue
                else:
                    # Handle single parameter for rew_exp and dyn_exp
                    # First try direct conversion
                    try:
                        test_value = float(test_name)
                    except:
                        # Parse from file path - extract parameter value after "linear-"
                        import re

                        if "linear-" in str(test_name):
                            # Extract parameter value using regex
                            match = re.search(r"linear-(-?\d+\.?\d*)", str(test_name))
                            if match:
                                test_value = float(match.group(1))
                            else:
                                print(f"Could not extract parameter from: {test_name}")
                                continue
                        else:
                            # Fallback: old parsing method
                            split = str(test_name).split("-")
                            test_value = ""
                            for i in range(1, len(split)):
                                test_value = test_value + split[i] + "-"
                            test_value = test_value[:-1]
                            test_value = float(test_value)
                test_values.append(test_value)

            return sorted(test_values)
        except Exception as e:
            print(f"Error getting test parameters for seed {seed}: {e}")
            print(f"Error type: {type(e)}")
            import traceback

            traceback.print_exc()
            # Fallback to default range
            assert False, f"Unknown seed: {seed}"

    def load_approximator(self, model_info):
        """Load an approximator from model info."""
        cfg_path = model_info["workdir"] / "cfg.yaml"
        cfg = OmegaConf.load(cfg_path)

        # Determine input dimension
        if cfg.input_to_model == "rew":
            input_dim = 1
        elif cfg.input_to_model == "dyn":
            input_dim = 1
        elif cfg.input_to_model == "rew_dyn":
            input_dim = 2
        else:
            raise NotImplementedError(f"Unknown input_to_model: {cfg.input_to_model}")

        # Create environment to get state/action dimensions
        reward_parameters = OmegaConf.to_container(cfg.reward_parameters)
        try:
            dynamics_parameters = OmegaConf.to_container(cfg.dynamics_parameters)
        except:
            dynamics_parameters = {"use_default": True}

        rg = np.random.RandomState(3)
        env = dmc.make(
            model_info["domain_task"],
            1,
            1,
            reward_parameters,
            dynamics_parameters,
            rg,
            False,
        )

        state_dim = env.observation_spec().shape[0]
        action_dim = env.action_spec().shape[0]

        # Create and load approximator
        approximator = make_approximator(
            input_dim, state_dim, action_dim, cfg.approximator, device=str(self.device)
        )

        model_dir = model_info["workdir"] / "models"

        # Determine the step to load
        if model_info["step_to_load"] == "best":
            step_name = "best_reward"
        else:
            step_name = model_info["step_to_load"]

        approximator.load(model_dir, step_name)
        approximator.train(False)

        # Move model to device if using GPU
        if "cuda" in str(self.device):
            approximator.rl_net = approximator.rl_net.to(self.device)

        return approximator, cfg, env

    def create_env_fn(self, domain_task, cfg):
        """Create environment function for batch evaluation."""
        reward_template = OmegaConf.to_container(cfg.reward_parameters)
        try:
            dynamics_template = OmegaConf.to_container(cfg.dynamics_parameters)
        except:
            dynamics_template = {"use_default": True}

        def env_fn(reward_params, dynamics_params):
            rg = np.random.RandomState(3)
            return dmc.make(
                domain_task, 1, 1, reward_params, dynamics_params, rg, False
            )

        return env_fn, reward_template, dynamics_template

    def batch_evaluate_model(self, model_info):
        """Evaluate a single model across all test parameters."""
        print(
            f"Evaluating {model_info['method']} seed {model_info['seed']} on {model_info['domain_task']}"
        )

        try:
            # Load model
            approximator, cfg, _ = self.load_approximator(model_info)

            # Get test parameters
            test_params = self.get_test_parameters(
                model_info["domain_task"], model_info["exp_name"], model_info["seed"]
            )

            # Create environment function
            env_fn, reward_template, dynamics_template = self.create_env_fn(
                model_info["domain_task"], cfg
            )

            # Batch evaluate using the efficient eval_env method
            result_dict, mean_reward = approximator.eval_env(
                env_fn=env_fn,
                n_episodes=self.args.n_episodes,
                env_params=test_params,
                reward_template=reward_template,
                dynamics_template=dynamics_template,
                input_to_model=cfg.input_to_model,
                mode="test",
            )

            # Store results
            eval_result = {
                "model_info": model_info,
                "mean_reward": mean_reward,
                "detailed_results": result_dict,
                "test_params": test_params,
                "input_to_model": cfg.input_to_model,
            }

            return eval_result

        except Exception as e:
            print(f"Error evaluating model {model_info}: {e}")
            return None
        finally:
            # Clean up GPU memory
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

    def save_results(self, results):
        """Save evaluation results to files."""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save detailed results as JSON
        json_results = []

        for result in results:
            if result is None:
                continue

            model_info = result["model_info"]

            # Get list of all variance values from detailed results
            variance_values = []
            for key, value in result["detailed_results"].items():
                if key.startswith("test/std@"):
                    variance_values.append(value**2)  # Convert std to variance

            # Add variance information to the result
            result["mean_variance"] = np.mean(variance_values)

            # Prepare JSON entry
            json_entry = {
                "method": model_info["method"],
                "seed": model_info["seed"],
                "domain_task": model_info["domain_task"],
                "exp_name": model_info["exp_name"],
                "input_to_model": result["input_to_model"],
                "mean_reward": result["mean_reward"],
                "mean_std": np.sqrt(result["mean_variance"]),
                "detailed_results": result["detailed_results"],
                "test_params": result["test_params"],
            }
            json_results.append(json_entry)

        # Compute global method results
        method_summary = defaultdict(list)
        for result in results:
            if result is None:
                continue
            model_info = result["model_info"]
            key = f"{model_info['method']}_{model_info['domain_task']}_{model_info['input_to_model']}"
            method_summary[key].append(
                np.array([result["mean_reward"], result["mean_variance"]])
            )

        global_method_results = {}
        for key, summary in method_summary.items():
            summary = np.array(summary)
            mean_reward = np.mean(summary[:, 0])
            mean_variance = np.mean(summary[:, 1])
            mean_std = np.sqrt(mean_variance)
            std_of_means = np.std(summary[:, 0])

            global_method_results[key] = {
                "mean_reward": float(mean_reward),
                "mean_std": float(mean_std),
                "std_of_means": float(std_of_means),
                "n_seeds": int(len(summary)),
                "all_rewards": summary[:, 0].tolist(),
                "all_std": np.sqrt(summary[:, 1]).tolist(),
            }

        # Create final JSON structure with both individual results and global summaries
        final_json = {
            "global_method_results": global_method_results,
            "individual_results": json_results,
            "evaluation_settings": {
                "n_episodes": self.args.n_episodes,
                "domain_task_list": self.args.domain_task_list,
                "exp_name_list": self.args.exp_name_list,
                "seeds": self.args.seeds,
            },
        }

        # Save JSON
        with open(output_dir / "batch_evaluation_results.json", "w") as f:
            json.dump(final_json, f, indent=2)

        print(f"Results saved to {output_dir}")

    def run_evaluation(self):
        """Run the complete batch evaluation."""
        model_paths = self.find_best_models()

        if not model_paths:
            print("No models found for evaluation!")
            return

        # Run all evaluations in parallel
        print(f"Starting parallel evaluation of {len(model_paths)} models...")

        # Adjust n_jobs for GPU usage
        n_jobs = self.args.n_jobs
        if "cuda" in str(self.device) and n_jobs == -1:
            # For GPU, limit parallel jobs to avoid memory issues
            n_jobs = min(20, len(model_paths))
            print(
                f"GPU detected: limiting to {n_jobs} parallel jobs to avoid memory issues"
            )
        elif n_jobs == -1:
            print(f"Using all available CPU cores")
        else:
            print(f"Using {n_jobs} parallel jobs")

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self.batch_evaluate_model)(model_info) for model_info in model_paths
        )

        # Filter out None results
        results = [result for result in results if result is not None]

        self.save_results(results)

        # Print summary
        print(f"\nEvaluation Summary:")
        print(f"Total models evaluated: {len(results)}")

        # Group by method and domain_task
        method_summary = defaultdict(list)
        for result in results:
            model_info = result["model_info"]
            key = f"{model_info['method']}_{model_info['domain_task']}_{model_info['input_to_model']}"
            method_summary[key].append(
                np.array([result["mean_reward"], result["mean_variance"]])
            )

        for key, summary in method_summary.items():
            summary = np.array(summary)
            mean_reward = np.mean(summary[:, 0])
            mean_variance = np.mean(summary[:, 1])
            mean_std = np.sqrt(mean_variance)

            print(f"{key}: {mean_reward:.3f} ± {mean_std:.3f} (n={len(summary)})")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of RL approximators")
    parser.add_argument(
        "--approximator_rootdir",
        type=str,
        default="results_approximator",
        help="Root directory containing approximator results",
    )
    parser.add_argument(
        "--domain_task_list",
        type=str,
        nargs="+",
        default=["cheetah_run", "finger_spin", "walker_walk"],
        help="List of domain tasks to evaluate",
    )
    parser.add_argument(
        "--exp_name_list",
        type=str,
        nargs="+",
        default=["dyn_exp", "rew_exp", "rew_dyn_exp"],
        help="List of experiment names to evaluate",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[123, 233, 666, 789, 999],
        help="Specific seeds to evaluate (if None, evaluates all found seeds)",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=10, help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="batch_eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for evaluation (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 uses all available cores)",
    )
    parser.add_argument(
        "--step_to_load",
        type=str,
        default="best",
        help="Step to load for model evaluation ('best' for best_reward, or specific step number like '1000000')",
    )

    args = parser.parse_args()
    print(args)

    evaluator = BatchEvaluator(args)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
