"""
Script for evaluating and rolling out several RL agents.
The rollout dataset is saved for training RL approximators.
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import subprocess
import argparse
from pathlib import Path
import json
import ast

from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', type=str, default="results")
parser.add_argument('--domain_task_list', type=list, default=['cheetah_run', 'finger_spin', 'walker_walk'])
parser.add_argument('--approximator_rootdir', type=str, default="results_approximator")
parser.add_argument('--step_to_load', type=int, default=1000000)
parser.add_argument('--n_episodes', type=int, default=10)
parser.add_argument('--random_rollout', action='store_true', default=False)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--rollout_dir', type=str, default='rollout_data_erl')
parser.add_argument('--video_dir', type=str, default='video_logs_erl')
parser.add_argument('--exp_name_list', type=list, default=['dyn_exp', 'rew_exp'])
args = parser.parse_args()

def my_fun(command):
    # print('Now Command', command)
    process = subprocess.run(command, capture_output=True)
    # print(f"Returncode of the process: {process.returncode}")
    if process.returncode == 0:
        pass
    else:
        print(process.stderr)

print('Starting')

command_list = []

for domain_task in args.domain_task_list:
    for exp_name in args.exp_name_list:
        if exp_name == 'rew_exp':
            rollout_data_folder = f'./rollout_data/rollout_data_grid_v4_rew/{domain_task}'
        elif exp_name == 'dyn_exp':
            rollout_data_folder = f'./rollout_data/rollout_data_grid_v4_dyn/{domain_task}'
        rollout_data_folder = Path(rollout_data_folder)

        args.domain_task = domain_task
        args.exp_name = exp_name
        approx_root_dir = Path(args.approximator_rootdir)
        approx_paths = sorted(approx_root_dir.glob(f'{args.exp_name}/**/**/*{args.domain_task}*/**/**/models/step_best_reward'))
        for approx_p in approx_paths:
            print(approx_p)
            approx_workdir = approx_p.parents[1]
            method = approx_p.parents[5].name
            seed = approx_p.parents[2].name.split('_')[1]
            test_list_file = sorted(rollout_data_folder.glob(f'test*{seed}*.txt'))[0]
            with open(str(test_list_file), 'r') as file:
                content = file.read()
            test_list = ast.literal_eval(content)

            for test_name in test_list:
                rollout_data_name = sorted(rollout_data_folder.glob(f'*{test_name}*.npy'))[0]
                try:
                    test_value = float(test_name)
                    pass
                except:
                    split = str(test_name).split('-')
                    test_value = ''
                    for i in range(1, len(split)):
                        test_value = test_value + split[i] + '-'
                    test_value = test_value[:-1]
                    test_value = float(test_value)
                # print(f'Test Value = {test_value}')
                # print(rollout_data_name)
                command = [
                    'python',
                    'eval_regressor.py',
                    '--rl_regressor_workdir',
                    str(approx_workdir),
                    '--n_episodes',
                    str(args.n_episodes),
                    '--rollout_dir',
                    str(args.rollout_dir),
                    '--video_dir',
                    str(args.video_dir),
                    '--eval_mode',
                    'comparison_data',
                    '--task_name',
                    args.domain_task,
                    '--method',
                    str(method),
                    '--gt_npy',
                    str(rollout_data_name),
                    '--dynamic_value',
                    str(test_value),
                ]
                if args.vis:
                    command += ['--vis']

                # print(f"Running ")
                # for i in command:
                #     print(str(i) + ' ', end='')
                # print("")
                
                command_list.append(command)

print('Total Case : ', len(command_list))

Parallel(n_jobs=48)(delayed(my_fun)(command_list[i]) for i in range(len(command_list)))