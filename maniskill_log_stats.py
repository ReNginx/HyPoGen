#!/usr/bin/env python3
"""
Statistics calculation script for ManiSkill evaluation logs.
Finds all log.txt files and calculates mean/std of success rates and episode lengths.
"""

import os
import json
import ast
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
import argparse


class ManiSkillLogAnalyzer:
    """Analyzer for ManiSkill evaluation logs."""

    def __init__(self, logs_dir: str = "maniskill/logs"):
        self.logs_dir = Path(logs_dir)
        self.results = defaultdict(list)

    def parse_log_file(self, log_path: Path) -> Tuple[float, List[int], Dict[str, Any]]:
        """
        Parse a single log.txt file to extract success rate and episode lengths.

        Returns:
            success_rate: Float between 0 and 1
            episode_lengths: List of episode lengths
            metadata: Dictionary with file metadata
        """
        try:
            with open(log_path, "r") as f:
                content = f.read().strip()

            # Extract success rate
            success_match = re.search(r"Success Rate:\s*([\d.]+)", content)
            if not success_match:
                raise ValueError("Success rate not found in log file")

            success_rate = float(success_match.group(1))

            # Extract episode lengths
            episode_match = re.search(
                r"Episode Lengths:\s*(\[.*?\])", content, re.DOTALL
            )
            if not episode_match:
                raise ValueError("Episode lengths not found in log file")

            episode_lengths = ast.literal_eval(episode_match.group(1))

            # Extract metadata from path
            metadata = self._extract_metadata_from_path(log_path)

            return success_rate, episode_lengths, metadata

        except Exception as e:
            print(f"Error parsing {log_path}: {e}")
            return None, None, None

    def _extract_metadata_from_path(self, log_path: Path) -> Dict[str, Any]:
        """Extract metadata from the log file path structure."""
        parts = log_path.parts

        # Find relevant path components
        metadata = {}

        # Method (e.g., "hypogen-cube", "mlp_rl_td-length")
        for part in parts:
            if any(method in part for method in ["hypogen"]):
                method, experiment = part.split("-")
                metadata["method"] = method
                metadata["param_type"] = experiment
                break

        # Environment and parameters from directory name
        env_dir = log_path.parent.name
        metadata["env_config"] = env_dir

        # Parse environment configuration
        # Format: TaskName-v0_param1value1_param2value2_...
        if "_" in env_dir:
            parts_list = env_dir.split("_")
            metadata["task"] = parts_list[0]  # e.g., "PickCube-v0" or "LiftCube-v0"

            # Parse parameters
            params = {}
            for part in parts_list[1:]:
                # Extract parameter name and value
                match = re.match(r"([a-zA-Z]+)([\d.]+)", part)
                if match:
                    param_name, param_value = match.groups()
                    try:
                        params[param_name] = float(param_value)
                    except:
                        params[param_name] = param_value

            metadata["parameters"] = params

            # Determine parameter type (what's being varied)
            if "cube" in metadata["param_type"]:
                metadata["param_value"] = params["cube"]
            elif "length" in metadata["param_type"]:
                metadata["param_value"] = params["length"]
            elif "stiff" in metadata["param_type"]:
                metadata["param_value"] = params["stiff"]
            elif "damp" in metadata["param_type"]:
                metadata["param_value"] = params["damp"]
            else:
                metadata["param_type"] = "default"
                metadata["param_value"] = 1.0
        print(metadata)

        return metadata

    def find_all_log_files(self) -> List[Path]:
        """Find all log.txt files in the logs directory."""
        log_files = []

        if not self.logs_dir.exists():
            print(f"Logs directory {self.logs_dir} does not exist!")
            return log_files

        # Recursively find all log.txt files
        for log_file in self.logs_dir.rglob("log.txt"):
            log_files.append(log_file)

        print(f"Found {len(log_files)} log files")
        return log_files

    def calculate_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate mean and std statistics for a group of results."""
        if not data:
            return {}

        success_rates = [d["success_rate"] for d in data]
        episode_lengths_all = []
        mean_episode_lengths = []
        success_episodes_all = []  # Episodes with length < 200

        for d in data:
            episode_lengths_all.extend(d["episode_lengths"])
            mean_episode_lengths.append(np.mean(d["episode_lengths"]))

            # Extract success episodes (length < 200)
            success_episodes = [
                length for length in d["episode_lengths"] if length < 200
            ]
            success_episodes_all.extend(success_episodes)

        stats = {
            "count": len(data),
            "success_rate_mean": np.mean(success_rates),
            "success_rate_std": np.std(success_rates),
            "episode_length_mean": np.mean(episode_lengths_all),
            "episode_length_std": np.std(episode_lengths_all),
            "mean_episode_length_per_run_mean": np.mean(mean_episode_lengths),
            "mean_episode_length_per_run_std": np.std(mean_episode_lengths),
            "success_rates": success_rates,
            "mean_episode_lengths_per_run": mean_episode_lengths,
        }

        # Add success episode statistics if there are any success episodes
        if success_episodes_all:
            stats["success_episode_length_mean"] = np.mean(success_episodes_all)
            stats["success_episode_length_std"] = np.std(success_episodes_all)
            stats["success_episode_count"] = len(success_episodes_all)
        else:
            stats["success_episode_length_mean"] = None
            stats["success_episode_length_std"] = None
            stats["success_episode_count"] = 0

        return stats

    def group_and_analyze(self, parsed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Group results and calculate statistics by method + modality + task."""
        grouped_results = defaultdict(list)

        # Group by method, experiment (param_type), and task
        for data in parsed_data:
            metadata = data["metadata"]

            # Extract method name (remove experiment suffix if present)
            method_full = metadata.get("method", "unknown")
            if "-" in method_full:
                method = method_full.split("-")[0]  # e.g., "hypogen-cube" -> "hypogen"
            else:
                method = method_full

            # Extract task name (remove version if present)
            task_full = metadata.get("task", "unknown")
            if "-v" in task_full:
                task = task_full.split("-v")[0]  # e.g., "LiftCube-v0" -> "LiftCube"
            else:
                task = task_full

            # Get experiment (parameter type)
            experiment = metadata.get("param_type", "default")

            # Create grouping key: method_experiment_task
            group_key = f"{method}_{experiment}_{task}"
            grouped_results[group_key].append(data)

        # Calculate statistics for each group
        statistics = {}
        for group_key, group_data in grouped_results.items():
            statistics[group_key] = self.calculate_statistics(group_data)

            # Add metadata
            if group_data:
                sample_metadata = group_data[0]["metadata"]
                method_full = sample_metadata.get("method", "unknown")
                method = (
                    method_full.split("-")[0] if "-" in method_full else method_full
                )

                task_full = sample_metadata.get("task", "unknown")
                task = task_full.split("-v")[0] if "-v" in task_full else task_full

                statistics[group_key]["metadata"] = {
                    "method": method,
                    "experiment": sample_metadata.get("param_type"),
                    "task": task,
                    "sample_count": len(group_data),
                }

        # Add overall statistics across all data
        if parsed_data:
            statistics["overall"] = self.calculate_statistics(parsed_data)
            statistics["overall"]["metadata"] = {
                "method": "all",
                "experiment": "all",
                "task": "all",
                "sample_count": len(parsed_data),
            }

        return statistics

    def analyze_logs(self) -> Dict[str, Any]:
        """Main analysis function."""
        print("Starting ManiSkill log analysis...")

        # Find all log files
        log_files = self.find_all_log_files()

        if not log_files:
            print("No log files found!")
            return {}

        # Parse all log files
        parsed_data = []
        failed_files = []

        for log_file in log_files:
            success_rate, episode_lengths, metadata = self.parse_log_file(log_file)

            if success_rate is not None and episode_lengths is not None:
                parsed_data.append(
                    {
                        "log_file": str(log_file),
                        "success_rate": success_rate,
                        "episode_lengths": episode_lengths,
                        "metadata": metadata,
                    }
                )
            else:
                failed_files.append(str(log_file))

        print(f"Successfully parsed {len(parsed_data)} files")
        if failed_files:
            print(f"Failed to parse {len(failed_files)} files: {failed_files}")

        # Group and analyze
        statistics = self.group_and_analyze(parsed_data)

        # Prepare final results
        results = {
            "summary": {
                "total_log_files": len(log_files),
                "successfully_parsed": len(parsed_data),
                "failed_to_parse": len(failed_files),
                "failed_files": failed_files,
            },
            "statistics": statistics,
            "raw_data": parsed_data,
        }

        return results

    def save_results(
        self, results: Dict[str, Any], output_dir, output_file: str = "maniskill_log_stats.json"
    ):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(os.path.join(output_dir, output_file))

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        json_results = convert_numpy(results)

        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {output_path}")

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the analysis results."""
        print("\n" + "=" * 60)
        print("MANISKILL LOG ANALYSIS SUMMARY")
        print("=" * 60)

        summary = results.get("summary", {})
        statistics = results.get("statistics", {})

        print(f"Total log files found: {summary.get('total_log_files', 0)}")
        print(f"Successfully parsed: {summary.get('successfully_parsed', 0)}")
        print(f"Failed to parse: {summary.get('failed_to_parse', 0)}")

        print("\n" + "-" * 40)
        print("STATISTICS BY METHOD + EXPERIMENT + TASK")
        print("-" * 40)

        # Sort statistics by key for consistent output
        for key in sorted(statistics.keys()):
            stats = statistics[key]
            metadata = stats.get("metadata", {})

            print(f"\n{key}:")
            print(f"  Method: {metadata.get('method', 'unknown')}")
            print(f"  Experiment: {metadata.get('experiment', 'unknown')}")
            print(f"  Task: {metadata.get('task', 'unknown')}")
            print(f"  Sample count: {stats.get('count', 0)}")
            print(
                f"  Success rate: {stats.get('success_rate_mean', 0):.3f} ± {stats.get('success_rate_std', 0):.3f}"
            )
            print(
                f"  Episode length: {stats.get('episode_length_mean', 0):.1f} ± {stats.get('episode_length_std', 0):.1f}"
            )
            print(
                f"  Mean episode length per run: {stats.get('mean_episode_length_per_run_mean', 0):.1f} ± {stats.get('mean_episode_length_per_run_std', 0):.1f}"
            )

            # Print success episode statistics if available
            if stats.get("success_episode_length_mean") is not None:
                print(
                    f"  Success episode length: {stats.get('success_episode_length_mean', 0):.1f} ± {stats.get('success_episode_length_std', 0):.1f} (n={stats.get('success_episode_count', 0)})"
                )


def main():
    parser = argparse.ArgumentParser(description="Analyze ManiSkill evaluation logs")
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="maniskill/logs",
        help="Directory containing log files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="batch_eval_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="maniskill_log_stats.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--print_summary", action="store_true", help="Print detailed summary to console"
    )

    args = parser.parse_args()

    # Create analyzer and run analysis
    analyzer = ManiSkillLogAnalyzer(args.logs_dir)
    results = analyzer.analyze_logs()

    # Save results
    analyzer.save_results(results, args.output_dir, args.output_file)

    # Print summary if requested
    if args.print_summary:
        analyzer.print_summary(results)

    print(f"\nAnalysis complete! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
