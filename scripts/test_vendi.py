import torch
from sgpo.models.continuous import ContinuousModel 
import time

from active_pretraining.setups.proteins import  ProteinProblemSetup
from active_pretraining.trainer import ActivePretrainingConfig, ActivePretraining
from active_pretraining.run import build_parser, setup_logger
from active_pretraining.problem_setup import Batch

from matplotlib import pyplot as plt


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

print('========= TRYING BATCH SAMPLE ================')

start = time.time()
N = 2
batch_size = 64
sample = env.batch_sample(N * batch_size, batch_size=batch_size)
batch = Batch.from_sample(sample)
print(time.time() - start)
print('======== DONE WITH BATCH SAMPLE =============')

start = time.time()
print(batch.samples.data.shape)
metrics = setup.compute_metrics(batch)
print(metrics, sample.valids.shape)

print(time.time() - start)