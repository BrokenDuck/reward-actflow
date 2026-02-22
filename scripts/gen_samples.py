from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import esm
from sgpo.models.continuous import ContinuousModel
import torch.nn.functional as F
from tqdm import tqdm

from adm.setups.proteins import ProteinModel, ProteinProblemSetup
from adm.trainer import ActivePretrainingConfig, ActivePretraining
from adm.run import build_parser, setup_logger
from adm.problem_setup import Batch
from diffusiongym import DDTensor
from diffusiongym.environments.endpoint import EndpointEnvironment
from diffusiongym.rewards.base import DummyReward
from vendi_score import vendi
import gpytorch


from matplotlib import pyplot as plt

def shim():
    # compatibility shim - put this before any imports that might call deepspeed.utils.is_initialized
    import importlib
    try:
        import deepspeed
    except Exception:
        # deepspeed not installed yet; nothing to do
        deepspeed = None

    if deepspeed is not None:
        # if utils.is_initialized already exists, do nothing
        try:
            if not (hasattr(deepspeed, "utils") and hasattr(deepspeed.utils, "is_initialized")):
                # prefer comm.is_initialized if available
                comm = None
                try:
                    # new DeepSpeed exposes the helper under deepspeed.comm (or deepspeed.comm.comm)
                    from deepspeed import comm as _ds_comm  # most common
                    comm = _ds_comm
                except Exception:
                    # fallback: try the nested comm module path sometimes used
                    try:
                        from deepspeed.comm import comm as _ds_comm2
                        comm = _ds_comm2
                    except Exception:
                        comm = None

                import types
                if not hasattr(deepspeed, "utils"):
                    deepspeed.utils = types.SimpleNamespace()

                if comm is not None and hasattr(comm, "is_initialized"):
                    deepspeed.utils.is_initialized = comm.is_initialized
                else:
                    # last resort: a safe function that returns False (won't crash callers that only expect a bool)
                    deepspeed.utils.is_initialized = lambda: False
        except Exception:
            # defensive: if anything goes wrong creating the shim, don't crash here
            pass

shim()

args = build_parser().parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = ActivePretrainingConfig.construct_from_args(args)
setup = ProteinProblemSetup(vars(args), device=device)

# Set up logging
logger = setup_logger(config, args.verbose)

apt = ActivePretraining(problem_setup=setup, config=config, logger=logger)

env = apt.env
esmfold = setup.esmfold
net = setup.base_model.sgpo_model
net : ContinuousModel

reward = DummyReward()
env.reward = reward

num_samples = 1000
N = 32
init = net.get_start(N)
x = DDTensor(init)
t = torch.ones((init.shape[0], )).to(net.device)* 0.5


samples = []
for i in tqdm(range(num_samples)):
    temp = env.sample(N, pbar=False)
    samples.append(Batch.from_sample(temp))



batch: Batch[DDTensor] = Batch.concat(samples)
setup.save_sample(batch.samples, None, 'samples')
metrics = setup.compute_metrics(batch)
print(metrics)