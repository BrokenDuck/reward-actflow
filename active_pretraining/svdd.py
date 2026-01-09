from typing import Any, Iterable

import torch
from flowgym import Environment, D
from tqdm import tqdm
from itertools import pairwise


@torch.no_grad()
def sample_svdd_pm(
    env: Environment[D],
    n: int,
    m: int = 20,
    alpha: float = 0.0,
    pbar: bool = True,
    **kwargs: Any,
):
    x, kwargs = env.base_model.sample_p0(n, **kwargs)
    x, kwargs = env.base_model.preprocess(x, **kwargs)

    trajectories = [x.to("cpu")]

    t = torch.linspace(2e-2, 1, env.discretization_steps + 1)
    iterator: Iterable[tuple[Any, Any]] = pairwise(t)
    if pbar:
        iterator = tqdm(iterator, total=env.discretization_steps)

    for t0, t1 in iterator:
        dt = t1 - t0
        t_curr = t0 * torch.ones(n, device=env.device)

        drift, _ = env.drift(x, t_curr, **kwargs)
        diffusion = env.diffusion(x, t_curr)

        # Sample m seeds which represent the noise, then we pick the best noise later
        seeds = torch.randint(0, 2**32 - 1, (m,), device=env.device)
        values = torch.zeros(n, m)

        for j, seed in enumerate(seeds):
            torch.manual_seed(seed)
            epsilon = x.randn_like()

            x_next = x + dt * drift + torch.sqrt(dt) * diffusion * epsilon

            x_final = env.pred_final(x_next, t_curr + dt, **kwargs)
            if not env.reward.latent_space:
                x_final = env.base_model.postprocess(x_final)

            values[:, j] = env.reward(x_final, **kwargs)[0].cpu()
            trajectories.append(x.to("cpu"))

        if alpha > 0:
            weights = torch.softmax(values / alpha, dim=1)
            indices = torch.multinomial(weights, num_samples=1).squeeze(1)
        else:
            indices = torch.argmax(values, dim=1)

        seeds = seeds[indices]
        epsilons = []
        for i in range(n):
            torch.manual_seed(seeds[i])
            epsilon = x.randn_like()
            epsilons.append(epsilon[i])

        epsilon = type(x).collate(epsilons)
        x += dt * drift + torch.sqrt(dt) * diffusion * epsilon

        if isinstance(iterator, tqdm):
            iterator.set_postfix({ "reward": values[indices].mean().item() })

    rewards, valids = env.reward(x, **kwargs)

    sample = env.base_model.postprocess(x)

    if env.reward.latent_space:
        rewards, valids = env.reward(x, **kwargs)
    else:
        rewards, valids = env.reward(sample, **kwargs)

    return sample, trajectories, rewards, valids, kwargs
