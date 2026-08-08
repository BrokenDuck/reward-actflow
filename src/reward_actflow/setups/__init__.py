from reward_actflow.setups.problem_setup import ProblemSetup
from reward_actflow.setups.toy import ToyProblemSetup

# `molecules.py` and `stable_diffusion.py` still target the pre-rewrite
# diffusiongym API (`Environment`, `BaseModel`, `diffusiongym.molecules`) and do
# not import. They are left in place, unregistered, until they are ported to
# `ModalityProvider` the way `reward_actflow/toy/providers.py` is.
setups: dict[str, type[ProblemSetup]] = {
    "toy": ToyProblemSetup,
}

__all__ = ["ProblemSetup", "ToyProblemSetup", "setups"]
