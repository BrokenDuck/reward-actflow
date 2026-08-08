"""Task-agnostic ActFlow: expand the flow model over the verifier's valid set.

The reward carries no task objective beyond validity, so the surrogate is fitted
on verifier outcomes and the acquisition is pure (validity-gated) uncertainty.
"""

from reward_actflow.explore import build_parser, setup_and_run


def main(args):
    setup_and_run(args)


def add_extra_args(parser):
    # Baselines. Each resolves to a reward gate in `ExploreConfig`: dropping the
    # verifier leaves `r = sigma`, dropping the surrogate leaves `r = 1[v(x)]`.
    parser.add_argument("--no_uncertainty", action="store_true")
    parser.add_argument("--no_verifier", action="store_true")


if __name__ == "__main__":
    parser = build_parser(add_extra_args)
    args = parser.parse_args()
    main(args)
