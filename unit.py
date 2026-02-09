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

print('============= SAMPLING FROM ENVIRONMENT =============')
init = net.get_start(32)
samples = env.sample(32, x0=FlowTensor(init))
strings = setup.base_model.embed_to_sequence(samples)

strings_net = net.sample(num_samples=32)



results = []
results_net = []
with torch.no_grad():
    for i in tqdm(range(len(strings))):
        s = strings[i]
        sn = strings_net[i]
        results.append(esmfold.infer(s))
        results_net.append(esmfold.infer(sn))

plddts = torch.vstack([r['mean_plddt'] for r in results])
plddts_net = torch.vstack([r['mean_plddt'] for r in results_net])


print(plddts)
print(plddts_net)


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
torch.save(plddts, 'plddts.pt')