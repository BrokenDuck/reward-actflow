"""Sample molecules from the pre-trained GEOM model and render 3D ball-and-stick views."""

import torch
import dgl
import numpy as np
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

from diffusiongym import DummyReward, construct_env
from diffusiongym.molecules import GEOMBaseModel
from flowmol.analysis.molecule_builder import SampledMolecule
from diffusiongym.molecules.rewards.utils import is_valid, is_not_fragmented

RDLogger.DisableLog("rdApp.*")

OUT_DIR = Path("geom_3d_samples")
N_SAMPLES = 10
NUM_STEPS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATOM_COLORS = {
    "C": "#404040", "N": "#3050F8", "O": "#FF0D0D", "S": "#FFFF30",
    "F": "#90E050", "Cl": "#1FF01F", "Br": "#A62929", "I": "#940094",
    "P": "#FF8000", "H": "#FFFFFF",
}
ATOM_RADII = {
    "C": 0.30, "N": 0.28, "O": 0.27, "S": 0.36,
    "F": 0.25, "Cl": 0.34, "Br": 0.38, "I": 0.42,
    "P": 0.35, "H": 0.15,
}
BOND_COLORS = {"single": "#888888", "double": "#555555", "triple": "#333333"}


def render_mol_3d(mol: Chem.Mol, path: Path, elev: float = 20, azim: float = -60):
    """Render a ball-and-stick 3D image of the molecule and save as PNG."""
    conf = mol.GetConformer()
    pos = np.array(conf.GetPositions())

    pos -= pos.mean(axis=0)

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondTypeAsDouble()
        p1, p2 = pos[i], pos[j]

        if bt == 1.0:
            ax.plot(*zip(p1, p2), color=BOND_COLORS["single"], linewidth=2.5, zorder=1)
        elif bt == 2.0:
            perp = np.cross(p2 - p1, [0, 0, 1])
            norm = np.linalg.norm(perp)
            if norm < 1e-6:
                perp = np.cross(p2 - p1, [0, 1, 0])
                norm = np.linalg.norm(perp)
            offset = perp / norm * 0.07
            ax.plot(*(zip(p1 + offset, p2 + offset)), color=BOND_COLORS["double"], linewidth=2.0, zorder=1)
            ax.plot(*(zip(p1 - offset, p2 - offset)), color=BOND_COLORS["double"], linewidth=2.0, zorder=1)
        else:
            ax.plot(*zip(p1, p2), color=BOND_COLORS["triple"], linewidth=3.5, zorder=1)

    for idx in range(mol.GetNumAtoms()):
        sym = mol.GetAtomWithIdx(idx).GetSymbol()
        if sym == "H":
            continue
        c = ATOM_COLORS.get(sym, "#AAAAAA")
        r = ATOM_RADII.get(sym, 0.30)
        ax.scatter(*pos[idx], s=r * 3000, c=c, edgecolors="black",
                   linewidths=0.5, zorder=2, depthshade=True)
        ax.text(*pos[idx], sym, fontsize=7, ha="center", va="center",
                color="white", fontweight="bold", zorder=3)

    margin = 1.5
    lims = [pos[:, i].min() - margin for i in range(3)], [pos[:, i].max() + margin for i in range(3)]
    ax.set_xlim(lims[0][0], lims[1][0])
    ax.set_ylim(lims[0][1], lims[1][1])
    ax.set_zlim(lims[0][2], lims[1][2])

    ax.set_axis_off()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.view_init(elev=elev, azim=azim)

    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


@torch.no_grad()
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading GEOM pre-trained model on {DEVICE}...")
    base_model = GEOMBaseModel(device=DEVICE)
    atom_type_map = base_model.model.atom_type_map
    reward = DummyReward()
    env = construct_env(base_model, reward, NUM_STEPS, 0)

    print(f"Sampling {N_SAMPLES} molecules...")
    sample = env.sample(N_SAMPLES, pbar=True)
    graph = sample.sample.graph.clone()

    def _discretize(x):
        return torch.nn.functional.one_hot(
            x.argmax(dim=-1), num_classes=x.shape[-1]
        ).type_as(x)

    x_key = "x_1" if "x_1" in graph.ndata else "x_t"
    a_key = "a_1" if "a_1" in graph.ndata else "a_t"
    c_key = "c_1" if "c_1" in graph.ndata else "c_t"
    e_key = "e_1" if "e_1" in graph.edata else "e_t"

    graph.ndata[a_key] = _discretize(graph.ndata[a_key])
    graph.ndata[c_key] = _discretize(graph.ndata[c_key])
    graph.edata[e_key] = _discretize(graph.edata[e_key])
    graph.edata["ue_mask"] = sample.sample.ue_mask

    mols = []
    for g in dgl.unbatch(graph):
        mol = SampledMolecule(g.cpu(), atom_type_map).rdkit_mol
        if mol is not None:
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
        mols.append(mol)

    valid_mols = []
    for i, mol in enumerate(mols):
        if mol is not None and is_valid(mol) and is_not_fragmented(mol):
            valid_mols.append((i, mol))

            sdf_path = OUT_DIR / f"mol_{i:02d}.sdf"
            w = Chem.SDWriter(str(sdf_path))
            w.write(mol)
            w.close()

            render_mol_3d(mol, OUT_DIR / f"mol_{i:02d}_3d.png")
            render_mol_3d(mol, OUT_DIR / f"mol_{i:02d}_3d_alt.png", elev=10, azim=30)

            smiles = Chem.MolToSmiles(mol)
            print(f"  mol_{i:02d}: {smiles}  [VALID]")
        else:
            status = "None" if mol is None else "INVALID"
            print(f"  mol_{i:02d}: {status}")

    print(f"\n{len(valid_mols)}/{N_SAMPLES} valid molecules")
    print(f"\nSaved to {OUT_DIR}/:")
    print(f"  - {len(valid_mols)} .sdf files (3D coordinates)")
    print(f"  - {len(valid_mols)*2} _3d.png files (3D ball-and-stick renders, 2 views each)")


if __name__ == "__main__":
    main()
