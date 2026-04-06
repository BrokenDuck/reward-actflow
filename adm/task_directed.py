import json
import argparse

from diffusiongym import reward_registry

from .explore import setup_and_run, build_parser


def main(args):
    reward_entry = reward_registry.get(args.reward)
    reward = reward_entry.instantiate(**args.reward_kwargs)
    setup_and_run(args, reward=reward, mean_weight=args.mean_weight)


def add_extra_args(parser):
    # Task
    parser.add_argument("--reward", type=str, choices=reward_registry.list(), required=True)
    parser.add_argument(
        "--reward_kwargs",
        type=json.loads,
        default={},
        help="JSON string, e.g. '{\"alpha\": 0.1, \"beta\": 2}'"
    )
    parser.add_argument("--reward_opt", type=str, choices=["max", "min"], default="max")
    parser.add_argument("--mean_weight", type=float, default=1.0)


if __name__ == "__main__":
    parser = build_parser(add_extra_args)
    args = parser.parse_args()
    main(args)
