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
    temperature: float = 0.0,
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
        epsilons = [x.randn_like() for _ in range(m)]
        values = torch.empty(n, m, device=env.device)

        for j, epsilon in enumerate(epsilons):
            x_next = x + dt * drift + torch.sqrt(dt) * diffusion * epsilon

            x_final = env.pred_final(x_next, t_curr + dt, **kwargs)
            if not env.reward.latent_space:
                x_final = env.base_model.postprocess(x_final)

            reward, valids = env.reward(x_final, **kwargs)
            reward[~valids] = -torch.inf
            values[:, j] = reward

        if temperature > 0:
            weights = torch.softmax(values / temperature, dim=1)
            indices = torch.multinomial(weights, num_samples=1).squeeze(1)
        else:
            indices = torch.argmax(values, dim=1)

        selected_epsilons = [epsilons[j][i] for i, j in enumerate(indices)]
        epsilon = type(x).collate(selected_epsilons)
        x += dt * drift + torch.sqrt(dt) * diffusion * epsilon

        trajectories.append(x.to("cpu"))

        if isinstance(iterator, tqdm):
            chosen = values[torch.arange(n), indices]
            finite_mask = torch.isfinite(chosen)

            mean_reward = torch.nan
            if finite_mask.any():
                mean_reward = chosen[finite_mask].mean().item()

            iterator.set_postfix({ "reward": mean_reward })

    sample = env.base_model.postprocess(x)

    if env.reward.latent_space:
        rewards, valids = env.reward(x, **kwargs)
    else:
        rewards, valids = env.reward(sample, **kwargs)

    return sample, trajectories, rewards, valids, kwargs
