"""Task-directed ActFlow: not yet ported to the rewritten diffusiongym.

Task-directed exploration fits the surrogate to a task reward rather than to
verifier outcomes and acquires on a UCB, `mean_weight * mu + sigma`, instead of
on uncertainty alone. Porting it needs two things that do not exist yet:

  * a task reward on the current interfaces. The rewards it selected from came
    from `diffusiongym.reward_registry`, the legacy `Reward` registry, whose
    remaining entries (images, molecules) still target the pre-rewrite API and
    do not import. They need to become `RewardProvider`s first.
  * a UCB gate in `reward_actflow.rewards.uncertainty`, and a surrogate fitted
    on `mean_weight`-scaled task rewards rather than on `batch.valids`.

Until then this fails here, with that list, rather than part-way through a run.
"""

import json


def main(args):
    raise NotImplementedError(__doc__)


def add_extra_args(parser):
    # Task
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument(
        "--reward_kwargs",
        type=json.loads,
        default={},
        help='JSON string, e.g. \'{"alpha": 0.1, "beta": 2}\'',
    )
    parser.add_argument("--reward_opt", type=str, choices=["max", "min"], default="max")
    parser.add_argument("--mean_weight", type=float, default=1.0)


if __name__ == "__main__":
    from reward_actflow.explore import build_parser

    parser = build_parser(add_extra_args)
    args = parser.parse_args()
    main(args)
