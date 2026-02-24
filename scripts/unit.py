from omegaconf import OmegaConf
import torch
from sgpo.models.continuous import ContinuousModel
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd

from adm.setups.proteins import ProteinProblemSetup
from adm.task_agnostic import TaskAgnosticConfig, TaskAgnostic, build_parser
from adm.setups.problem_setup import SampleFile
from adm.utils import setup_logger
from diffusiongym import DDTensor

from matplotlib import pyplot as plt


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



print('============= CHECKING SCHEDULER FUNCS =============')
N = 200
x0, _ = setup.base_model.sample_p0(N)
t = torch.arange(N) / N
sched = setup.base_model.scheduler

alpha = sched.alpha(x0, t).data[:, 0, 0]
beta = sched.beta(x0, t).data[:, 0, 0]
alpha_dot = sched.alpha_dot(x0, t).data[:, 0, 0]
beta_dot = sched.beta_dot(x0, t).data[:, 0, 0]

plt.plot(alpha, label='alpha')
plt.plot(beta, label='beta')
plt.plot(alpha_dot, label='alpha_dot')
plt.plot(beta_dot, label='beta_dot')
plt.legend()

plt.savefig('schedulers.png')


print('============= SAMPLING FROM ENVIRONMENT =============')
N = 32
init = net.get_start(N)
x = DDTensor(init)
t = torch.ones((init.shape[0], )).to(net.device)* 0.5


samples = env.sample(N, x0=DDTensor(init), threshold=50, debug=False, valid_pbar=True)

print('=========== CHECKING METRICS =================')

temp_dir = Path('/cluster/scratch/kprotopapas/temp')
os.makedirs(temp_dir, exist_ok=True)

sample_files = []
for i in range(N):
    sample_path = temp_dir / f'{i:04d}'
    setup.save_sample(samples.sample[i], {}, sample_path)
    sample_files.append(SampleFile(is_valid=bool(samples.valids[i].item()), file=temp_dir / f'{i:04d}.pt'))

sample_metrics = setup.compute_sample_metrics(sample_files)
metrics_df = pd.DataFrame.from_records(sample_metrics).T
print(metrics_df.head(3))
print(metrics_df['fitness'].mean())

print('=========== CHECKING VALIDITY =================')

samples_rand = env.sample(N, x0=DDTensor(init), threshold=50, debug=True, valid_pbar=True)

strings = setup.base_model.probs_to_sequence(samples.sample)
strings_rand = setup.base_model.probs_to_sequence(samples_rand.sample)

print(f'Validity: {samples.valids.mean()}')

conf = OmegaConf.load(args.cfg_path)
seq_len = conf.data.seq_len
infill_seed = torch.randint(0, net.model.network.vocab_size, (seq_len,)).to(
            torch.device('cuda'))  # random seed of token ids for now
# 1 if != pad, else 0
infill_mask = (torch.ones(seq_len) != net.tokenizer.pad_id-100).to(
    torch.device('cuda'))  # switch 30 for net.tokenizer.pad_id

# corrupt_mask: 1 for real tokens, 0 for pad (Equivalent to "fully corrupt all real tokens")
corrupt_mask = infill_mask.clone().to(torch.device('cuda')) 

tokens_net = net.sample(num_samples=N, infill_seed=infill_seed, infill_mask=infill_mask, corrupt_mask=corrupt_mask)
strings_net = [net.tokenizer.untokenize(t) for t in tokens_net]

random_tokens = torch.randint(0, net.model.network.vocab_size - 1, (N, seq_len))
random_strings = [net.tokenizer.untokenize(t) for t in random_tokens]

if esmfold is not None:
    results = []
    results_rand = []
    results_net = []
    results_fullrand = []
    with torch.no_grad():
        for i in tqdm(range(len(strings))):
            s = strings[i]
            sn = strings_net[i]
            snr = strings_rand[i]
            snfr = random_strings[i]
            results.append(esmfold.infer(s))
            results_net.append(esmfold.infer(sn))
            results_rand.append(esmfold.infer(snr))
            results_fullrand.append(esmfold.infer(snfr))
        

    plddts = torch.vstack([r['mean_plddt'] for r in results])
    plddts_net = torch.vstack([r['mean_plddt'] for r in results_net])
    plddts_rand = torch.vstack([r['mean_plddt'] for r in results_rand])
    plddts_fullrand = torch.vstack([r['mean_plddt'] for r in results_fullrand])


    print(f'APT mean pLDDT: {plddts.mean()}')
    print(f'SGPO means pLDDT: {plddts_net.mean()}')
    print(f'random score : {plddts_rand.mean()}')
    print(f'random tokens : {plddts_fullrand.mean()}')

else:
    print('No verifier')


print('=========== CHECKING SGPO BASE FUNCTIONS ============')
conf = OmegaConf.load(args.cfg_path)

residues = list(range(conf.data.seq_len))
mask = torch.zeros(conf.data.seq_len)
mask[residues] = 1
mask = mask.to(net.device).int()

init = net.get_start(32)
net.noise_schedule.sigmas = net.noise_schedule.sigmas.to(net.device).float()
attn_mask = torch.ones((init.shape[0], init.shape[1]), dtype=torch.bool, device=net.device)

infill_mask = (torch.ones(net.seq_len) != net.tokenizer.pad_id-100).to(net.device) 
attn_mask = torch.ones((init.shape[0], net.seq_len),dtype=torch.bool, device=net.device)
x = init
t = torch.ones((init.shape[0],)).to(net.device) * 0.
idx = (t * len(net.noise_schedule.sigmas)).round().clamp(0, len(net.noise_schedule.sigmas) - 1).long().to(net.device)
sigma = net.noise_schedule.sigmas.to(net.device)[idx].reshape((x.shape[0], 1, 1))
f_out = net.model.network.forward(x/(sigma**2 + 1).sqrt(), idx, attn_mask=attn_mask)

infill_mask = infill_mask[None, :, None]

out = net.model.network.pred_xstart(
    x,
    idx,
    attn_mask=attn_mask,
    sequence_output=f_out['sequence_output'],
    infill_mask=infill_mask,
    bad_word_ids=None,
    classifier=None,
    guidance_scale=0.
)
x0 = out['xstart']

samples = out['probs'].argmax(dim=-1)
strings = [net.tokenizer.untokenize(s) for s in samples]

results = []
with torch.no_grad():
    for s in tqdm(strings):
        results.append(esmfold.infer(s))

plddts = torch.vstack([r['mean_plddt'] for r in results])
# torch.save(plddts, 'plddts.pt')
