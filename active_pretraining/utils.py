from typing import Any, TypeVar, Optional
import torch


T = TypeVar("T")


def index_dict(d: T, start: int, end: Optional[int] = None) -> T:
    """Recursively index into the leaves of a nested dictionary.

    Parameters
    ----------
    d : T
        Any value, if a dictionary, will be processed recursively.

    start : int
        The index to select from list/tensor leaves.
    
    end : Optional[int], optional
        The end index to select from list/tensor leaves, by default None.

    Returns
    -------
    T
        If d is a dictionary, returns a dictionary with the same keys and indexed leaves.
    """
    if end is None:
        idx = start
    else:
        idx = slice(start, end)

    if isinstance(d, dict):
        return {k: index_dict(v, start, end) for k, v in d.items()}

    elif isinstance(d, (list, tuple, torch.Tensor)):
        return d[idx]

    elif isinstance(d, (float, int, str)):
        return d

    else:
        raise TypeError(f"Unsupported leaf type: {type(d)}")


def add_valid_border(images: torch.Tensor, valids: torch.Tensor, thickness: int = 2) -> torch.Tensor:
    images = images.clone()
    if images.shape[1] == 1:
        images = torch.cat([images, images, images], dim=1)

    for i, valid in enumerate(valids):
        if valid:
            images[i, 1, :thickness, :] = 1.0
            images[i, 1, -thickness:, :] = 1.0
            images[i, 1, :, :thickness] = 1.0
            images[i, 1, :, -thickness:] = 1.0

    return images
