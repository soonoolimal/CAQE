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

    def __len__(self) -> int:
        sample = torch.load(self.chunks[0], map_location="cpu", weights_only=True)
        return sample["hidden"].shape[0] * len(self.chunks)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        chunks = list(self.chunks)

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
                yield data["hidden"][i], data["target_token_ids"][i]


def split_chunks(chunk_dirs: list[Path] | Path, val_ratio: float) -> tuple[list[Path], list[Path]]:
    """Splits sorted chunk files into train and valid lists.

    Each directory is split independently so every corpus contributes proportionally to both splits.

    The last val_ratio fraction of chunks is held out so that sorting ensures a deterministic split.

    Sentences are shuffled before extraction so that chunks contain random sentences,
    making this tail-based split representative of the full corpus.
    """
    if isinstance(chunk_dirs, Path):
        chunk_dirs = [chunk_dirs]

    train_chunks, val_chunks = [], []
    for chunk_dir in chunk_dirs:
        chunks = sorted(chunk_dir.glob("chunk_*.pt"))
        if not chunks:
            raise FileNotFoundError(f"No chunk files found: {chunk_dir}")

        n_val = max(1, int(len(chunks) * val_ratio))

        train_chunks.extend(chunks[:-n_val])
        val_chunks.extend(chunks[-n_val:])

    return train_chunks, val_chunks


def make_loader(chunks: list[Path], batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    """Wraps a list of chunk files into a DataLoader."""
    return DataLoader(
        ChunkDataset(chunks, shuffle=shuffle),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
