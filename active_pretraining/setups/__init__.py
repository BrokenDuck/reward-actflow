from .mnist import MNISTProblemSetup
from .toy import ToyProblemSetup
from .molecules import QM9ProblemSetup


setups = {"mnist": MNISTProblemSetup, "toy": ToyProblemSetup, "qm9": QM9ProblemSetup}
