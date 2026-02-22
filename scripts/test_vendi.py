import torch
from sgpo.models.continuous import ContinuousModel

from adm.setups.proteins import ProteinProblemSetup
from adm.task_agnostic import TaskAgnosticConfig, TaskAgnostic, build_parser
from adm.utils import setup_logger
from adm.setups.problem_setup import SampleFile

import time
from pathlib import Path
import os

args = build_parser().parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

setup = ProteinProblemSetup(vars(args), device=device)

# Set up logging
config = TaskAgnosticConfig.construct_from_args(args)
logger = setup_logger(config.folder, args.verbose)

apt = TaskAgnostic(problem_setup=setup, config=config, logger=logger)
env = apt.env
esmfold = setup.esmfold
net = setup.base_model.sgpo_model
net : ContinuousModel

print('========= TRYING BATCH SAMPLE ================')

start = time.time()
N = 2
batch_size = 64
samples = env.batch_sample(N * batch_size, batch_size=batch_size)
print(time.time() - start)
print('======== DONE WITH BATCH SAMPLE =============')

start = time.time()
tmpdir = Path(os.environ['TMPDIR'])

sample_files = []
for i in range(N):
    sample_path = tmpdir / f'{i:04d}'
    setup.save_sample(samples.sample[i], {}, sample_path)
    sample_files.append(SampleFile(is_valid=bool(samples.valids[i].item()), file=tmpdir / f'{i:04d}.pt'))

metrics = setup.compute_metrics(sample_files)
print(metrics, samples.valids.shape)

print(time.time() - start)