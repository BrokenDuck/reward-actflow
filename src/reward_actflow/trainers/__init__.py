"""ActFlow-R fine-tuning algorithms. Importing this registers them.

Named `trainers/`, not `ft/`, to avoid confusion with the deprecated
`reward_actflow.ft_methods`. Registration lives here rather than in
`mixture_replay.py` itself, mirroring diffusiongym's own convention
(`make.py`'s `_register_builtins()` registers `ORWCFM` etc., not their own
modules) — it keeps the trainer itself free of any registry dependency.
"""

from diffusiongym.registry import algorithm_registry

from reward_actflow.trainers.mixture_replay import (
    MixtureExperience,
    MixtureReplay,
    ReplaySource,
)

algorithm_registry.register("mixture_replay", MixtureReplay)

__all__ = [
    "MixtureExperience",
    "MixtureReplay",
    "ReplaySource",
]
