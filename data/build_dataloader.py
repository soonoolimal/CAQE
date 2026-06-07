"""
DataLoader builder for hidden vector chunk files (.pt).
Loads chunked hidden vectors sequentially for CAQE model training.
"""

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset


class ChunkDataset(IterableDataset):
    """Iterable dataset over a given list of hidden vector chunk files.

    Loads each chunk once, yields all samples within it, then moves to the next.
    With num_workers > 0, chunks are split evenly across workers for parallel loading.
    """

    def __init__(self, chunks: list[Path], shuffle: bool):
        if not chunks:
            raise FileNotFoundError("No chunk files provided.")
        self.chunks = chunks
        self.shuffle = shuffle

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        chunks = list(self.chunks)

        # split chunks evenly across workers
        if worker_info is not None:
            chunks = chunks[worker_info.id :: worker_info.num_workers]

        if self.shuffle:
            random.shuffle(chunks)

        for chunk_path in chunks:
            data = torch.load(chunk_path, map_location="cpu", weights_only=True)
            n = data["hidden"].shape[0]

            indices = list(range(n))
            if self.shuffle:
                random.shuffle(indices)

            for i in indices:
                hidden = data["hidden"][i]            # [hidden_dim]
                target = data["target_token_ids"][i]  # scalar tensor
                yield hidden, target


def split_chunks(chunk_dir: Path, val_ratio: float) -> tuple[list[Path], list[Path]]:
    """Splits sorted chunk files into train and validation lists.

    The last val_ratio fraction of chunks is held out for validation.
    Sorting ensures a deterministic and reproducible split.

    Sentences are shuffled before extraction so that chunks contain random sentences,
    making this tail-based split representative of the full corpus.
    """
    chunks = sorted(chunk_dir.glob("chunk_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"No chunk files found: {chunk_dir}")

    n_val = max(1, int(len(chunks) * val_ratio))
    train_chunks = chunks[:-n_val]
    val_chunks = chunks[-n_val:]

    return train_chunks, val_chunks


def make_dataloader(
    chunks: list[Path],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """Builds a DataLoader from a list of chunk files."""
    dataset = ChunkDataset(chunks, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
