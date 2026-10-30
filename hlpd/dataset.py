"""Paired text datasets.

Every data file is a JSON object with two parallel lists:

    {"original": [...human-written passages...],
     "rewritten" | "sampled": [...machine-revised / generated passages...]}

``original[i]`` and ``rewritten[i]`` share the same content.  Training files
use the key ``sampled`` (ImBD convention), evaluation files use ``rewritten``;
both are accepted everywhere.
"""
import json
import warnings

from torch.utils.data import Dataset

MACHINE_KEYS = ("rewritten", "sampled", "machine", "generated")


def load_pairs(path):
    """Return ``(human_texts, machine_texts)`` from a data file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    human = data["original"]
    machine = None
    for k in MACHINE_KEYS:
        if k in data:
            machine = data[k]
            break
    if machine is None:
        raise KeyError(f"{path}: expected one of {MACHINE_KEYS} next to 'original'")
    n = min(len(human), len(machine))
    if len(human) != len(machine):
        warnings.warn(f"{path}: original={len(human)} vs machine={len(machine)} passages; "
                      f"using the first {n} pairs")
    return list(human[:n]), list(machine[:n])


class PairedTextDataset(Dataset):
    """Yields ``(human_text, machine_text)`` tuples."""

    def __init__(self, data_path):
        self.data_path = data_path
        self.human, self.machine = load_pairs(data_path)

    def __len__(self):
        return len(self.human)

    def __getitem__(self, index):
        return self.human[index], self.machine[index]


class PairedTextSplitDataset(PairedTextDataset):
    """Deterministic train / val split of a paired file (every ``1/val_ratio``-th
    pair goes to validation).  ``val_ratio=0`` keeps everything."""

    def __init__(self, data_path, split="train", val_ratio=0.0):
        super().__init__(data_path)
        total = len(self.human)
        if val_ratio <= 0:
            self.indices = list(range(total))
            return
        step = int(round(1.0 / val_ratio))
        val_indices = set(range(0, total, step))
        if split == "train":
            self.indices = [i for i in range(total) if i not in val_indices]
        elif split == "val":
            self.indices = sorted(val_indices)
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return self.human[i], self.machine[i]
