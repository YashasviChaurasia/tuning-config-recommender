import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

import yaml
from loguru import logger

from recommender.actions import Action
from recommender.adapters import FMSAdapter


def load_actions_from_folder(folder_path):
    if not folder_path:
        return
    folder = Path(folder_path).resolve()

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder - {folder}")
    sys.path.insert(0, str(folder))
    classes = {}
    for _, module_name, _ in pkgutil.iter_modules([str(folder)]):
        module = importlib.import_module(module_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Action)
                and attr_name.startswith("Custom_")
            ):
                classes[attr_name] = attr
    logger.info(f"Following additional actions are detected - {classes}")
    return classes


def main():
    parser = argparse.ArgumentParser(description="Recommender CLI interface")
    parser.add_argument(
        "--rules-dir",
        required=False,
        type=str,
        default=None,
        help="Path to folder containing rules/actions in Python",
    )
    parser.add_argument(
        "--tuning-data-config",
        required=False,
        type=str,
        help="Path to tuning data config yaml",
    )
    parser.add_argument(
        "--accelerate-config",
        required=False,
        type=str,
        help="Path to accelerate config yaml",
    )
    parser.add_argument(
        "--tuning-config", required=False, type=str, help="Path to tuning config yaml"
    )
    parser.add_argument(
        "--compute-config", required=False, type=str, help="Path to compute config yaml"
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        type=str,
        default="./recommender_output",
        help="Path to compute config",
    )
    parser.add_argument(
        "--skip-estimator",
        required=False,
        type=bool,
        default=False,
        help="Path to compute config",
    )
    args = parser.parse_args()
    additional_actions = load_actions_from_folder(args.rules_dir)
    fms_adapter = FMSAdapter(
        base_dir=args.output_dir, additional_actions=additional_actions
    )

    result = fms_adapter.execute(
        train_config=yaml.safe_load(open(args.tuning_config)),
        compute_config=yaml.safe_load(open(args.compute_config)),
        dist_config=yaml.safe_load(open(args.accelerate_config)),
        data_config=yaml.safe_load(open(args.tuning_data_config)),
        unique_tag="",
        paths={},
        skip_estimator=args.skip_estimator,
    )
    print(result)


if __name__ == "__main__":
    main()
