import torch
from sgpo.models.continuous import ContinuousModel

from adm.setups.proteins import ProteinProblemSetup
from adm.task_agnostic import TaskAgnosticConfig, TaskAgnostic, build_parser
from adm.utils import setup_logger
from diffusiongym import DDTensor


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


print('============= SAMPLING FROM ENVIRONMENT =============')
N = 128

for i in range(1):
    n = N * i 
    init = net.get_start(N)
    x = DDTensor(init)
    t = torch.ones((init.shape[0], )).to(net.device)* 0.5

    samples = env.sample(N, x0=DDTensor(init), threshold=50, valid_pbar=True)
    if len(samples) > 0:
        print(f'No crashes yet at batch_size = {n}', flush=True)
        print(f'Avg validity: {samples.valids.mean()} @ {len(samples)}')
        