"""Train/test splitting. The default is a scaffold split: no Bemis-Murcko scaffold appears in both sets,
which mimics predicting genuinely new chemotypes and avoids the optimistic bias of random splits."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import config
from .features import to_mols


def murcko_scaffolds(smiles) -> list[str]:
    scaffolds = []
    for smi, mol in zip(smiles, to_mols(smiles)):
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffolds.append(scaf if scaf else smi)   # acyclic molecules: each is its own group
    return scaffolds


def scaffold_split(smiles, test_size: float = config.TEST_SIZE, seed: int = config.SEED):
    """Balanced scaffold split (large scaffold groups -> train, remaining groups shuffled), as in Chemprop.

    Returns (train_idx, test_idx, scaffold_labels)."""
    scaffolds = murcko_scaffolds(smiles)
    groups = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[s].append(i)

    n_test_target = int(round(test_size * len(scaffolds)))
    big = [g for g in groups.values() if len(g) > n_test_target / 2]
    small = [g for g in groups.values() if len(g) <= n_test_target / 2]
    np.random.default_rng(seed).shuffle(small)

    train = [i for g in big for i in g]
    test: list[int] = []
    for g in small:
        if len(test) + len(g) <= n_test_target:
            test.extend(g)
        else:
            train.extend(g)
    return np.array(sorted(train)), np.array(sorted(test)), np.array(scaffolds)


def random_split(n: int, test_size: float = config.TEST_SIZE, seed: int = config.SEED):
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(round(test_size * n))
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])
