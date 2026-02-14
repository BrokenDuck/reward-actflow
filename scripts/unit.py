from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import esm
from sgpo.models.continuous import ContinuousModel
import torch.nn.functional as F
from tqdm import tqdm

from active_pretraining.setups.proteins import ProteinModel, ProteinProblemSetup
from active_pretraining.trainer import ActivePretrainingConfig, ActivePretraining
from active_pretraining.run import build_parser, setup_logger
from flowgym import FlowTensor

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
x = FlowTensor(init)
t = torch.ones((init.shape[0], )).to(net.device)* 0.5


samples = env.sample(N, x0=FlowTensor(init), threshold=50, debug=False, valid_pbar=True)
samples_rand = env.sample(N, x0=FlowTensor(init), threshold=50, debug=True, valid_pbar=True)

strings = setup.base_model.embed_to_sequence(samples.sample)
strings_rand = setup.base_model.embed_to_sequence(samples_rand.sample)

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