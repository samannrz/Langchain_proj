# Data class
from contextlib import contextmanager
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset
from Task1 import *
from Task2 import *

class MRDataset(Dataset):
    """Base class for MR datasets (2D and 3D).

    Args:
        data_dir: Local path(s) to the directory containing hdf files with MR images.
        crop_factor: Factor by which to crop the k-space data (1 for no cropping).
        noise_level: Std of Gaussian noise to add to k-space (0 for no noise).
        random_seed: Random seed to use for noise simulation.

    Methods:
        __len__():
            Return the total number of frames in the dataset.

        __getitem__(idx):
            Get an item from the dataset.

            WARNING: This method does not accept negative indices.

            Args:
                idx: Global index of the frame to load.

            Returns:
                A tuple containing:
                - Low-resolution frame tensor (N x H x W).
                - High-resolution frame tensor (N x H x W).
    """  # noqa: D214

    def __init__(
        self,
        data_dir: list[str],
        crop_factor: int,
        noise_level: float,
        random_seed: int,
    ) -> None:
        """Initialize a dataset with local path(s) and a degradation."""
        # Indent the following lines
        self.data_dir = data_dir
        self.crop_factor = crop_factor
        self.noise_level = noise_level
        self.random_seed = random_seed
        self.files = []
        self._files_opened_count = 0 # Initialize the counter


        # Initialize files
        for p in self.data_dir:
            self.files.extend(self._list_files(p))

        # Generate a list of seeds to apply different degradation per volume
        self.random_seeds = np.random.RandomState(random_seed).randint(0, 2**32, len(self.files))

        # Load depths for all volumes
        self.depths = [self._get_depth(f) for f in self.files]
        self.cumulative_depths = np.cumsum(self.depths)

    def __len__(self) -> int:
        """Return the total number of frames in the dataset."""
        return sum(self.depths)

    def _list_files(self, path: str) -> list[Path]:
        files = [Path(f) for f in Path(path).iterdir() if Path(f).is_file() and Path(f).suffix == '.h5']

        return files

    def _map_index(self, global_frame_idx: int) -> tuple[int, int]:
        file_idx = np.searchsorted(self.cumulative_depths, global_frame_idx, side="right")
        local_frame_idx = global_frame_idx - (
            self.cumulative_depths[file_idx - 1] if file_idx > 0 else 0
        )
        return file_idx, local_frame_idx

    @contextmanager
    def _get_file(self, file_path: Path):
        handle = file_path
        self._files_opened_count += 1 # Increment the counter

        try:
          with handle.open("rb") as field_object, h5py.File(field_object, "r") as f:
            yield f
        except Exception as e:
          print(f"Error opening file {file_path}: {e}")
          raise # Re-raise the exception after printing


    def _get_depth(self, file: Path) -> int:
        with self._get_file(file) as f:
            return f["reconstruction_rss"].shape[0]


class MRDataset2D(MRDataset):
    """Dataset for high-resolution / simulated low-resolution 2D images."""

    def __getitem__(self, global_frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset."""
        # Exit with an error if the index is negative
        if global_frame_idx < 0:
            raise ValueError("Negative indices are not supported.")
        file_idx, local_frame_idx = self._map_index(global_frame_idx)

        with self._get_file(self.files[file_idx]) as f:
            # High-resolution frame
            hr_volume = np.abs(f["reconstruction_rss"][:])
            hr_normalized = normalize_volume(hr_volume)
            hr_frame = hr_normalized[local_frame_idx]

            # Low-resolution frame
            lr_volume = degrade_volume(
                hr_normalized,
                crop_factor=self.crop_factor,
                noise_level=self.noise_level,
                random_seed=self.random_seeds[file_idx],
            )

            lr_frame = lr_volume[local_frame_idx]

            # Convert to tensors
            hr_tensor = torch.tensor(hr_frame, dtype=torch.float32).unsqueeze(0)
            lr_tensor = torch.tensor(lr_frame, dtype=torch.float32).unsqueeze(0)


            return lr_tensor, hr_tensor