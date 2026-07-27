from .toy import ToyProblemSetup
from .molecules import QM9ProblemSetup, GEOMDrugsProblemSetup
from .stable_diffusion import StableDiffusionProblemSetup


setups = {
    "toy": ToyProblemSetup,
    "qm9": QM9ProblemSetup,
    "geom_drugs": GEOMDrugsProblemSetup,
    "stable_diffusion": StableDiffusionProblemSetup,
}
