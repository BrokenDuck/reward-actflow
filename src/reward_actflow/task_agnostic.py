from diffusiongym import DummyReward

from reward_actflow.explore import setup_and_run, build_parser


def main(args):
    setup_and_run(args, reward=DummyReward(), mean_weight=0)


def add_extra_args(parser):
    # Baselines
    parser.add_argument("--no_uncertainty", action="store_true")
    parser.add_argument("--no_verifier", action="store_true")


if __name__ == "__main__":
    parser = build_parser(add_extra_args)
    args = parser.parse_args()
    main(args)
