import random

from torch.utils.data import Dataset
import torch
from tqdm import tqdm


class FixedLenMultiTimeSeries(Dataset):
	def __init__(
			self,
			features: torch.Tensor,
			targets: torch.Tensor,
			feature_win: int,
			target_win: int,
			feature_target_intersection: int
	):
		"""
		denote FL as feature lag, TL as target lag, intersection as I and history size as H. Assume we want to predict
		at time t+FL+1, the time windows we provide are:

				ALL WINDOW: [0, 1, 2, 3,  ..., t-1, t, t+1,           ...               H-4, H-3, H-2, H-1]
				FEATURE WINDOW:                    [t, t+1, ..., t-FL-1, t+FL]
				TARGET WINDOW:                               [t+FL-I ,...t+FL, t+FL+1 , ..., t+FL+TL]

		as per the number of windows per ds - for every t we create a window, but also leaving room for the first
		window (FL timestamps) and the last window (TL timestamps), minus their intersection + 1 (cut t_0=0)

		:param features: a (normalized) tensor with shape (N,H,F), where:
						 - N: number of datasets
						 - H: number of samples per dataset
						 - F: number of features per sample
		:param targets: a (normalized) tensor with shape (N,H,T) where T is the number of targets to predict
		:param feature_win: feature history to consider
		:param target_win: target future to predict
		:param feature_target_intersection: intersection between target and features
		"""
		self.feature_win: int = feature_win
		self.target_win: int = target_win
		self.feature_target_intersect: int = feature_target_intersection
		self.normalized_features: torch.Tensor = features
		self.normalized_targets: torch.Tensor = targets
		self.n_datasets, self.n_samples_per_ds, self.n_features = self.normalized_features.shape
		self.n_windows_per_ds = self.n_samples_per_ds - self.feature_win + self.feature_target_intersect - self.target_win + 1

	def __len__(self) -> int:
		return self.n_datasets * self.n_windows_per_ds

	def __getitem__(
			self,
			idx: int
	) -> tuple[torch.Tensor, torch.Tensor]:
		ds_idx = idx // self.n_windows_per_ds
		win_idx = idx % self.n_windows_per_ds
		features_window = self.normalized_features[ds_idx, win_idx: win_idx + self.feature_win]
		target_window = self.normalized_targets[
						ds_idx,
						win_idx + self.feature_win - self.feature_target_intersect:
						win_idx + self.feature_win - self.feature_target_intersect + self.target_win
						]
		return features_window, target_window


class VariableLenMultiTimeSeries(Dataset):
	def __init__(
			self,
			features: list[torch.Tensor],
			targets: list[torch.Tensor],
			feature_win: int,
			target_win: int,
			feature_target_intersection: int
	):
		self.feature_win: int = feature_win
		self.target_win: int = target_win
		self.feature_target_intersect: int = feature_target_intersection
		self.normalized_features: list[torch.Tensor] = features
		self.normalized_targets: list[torch.Tensor] = targets

		self.windows = self.calculate_windows()

	def calculate_windows(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
		"""
		Pre-calculate all rolling windows and store them in a list.
		Each element in the list is a tuple containing a feature window and a target window.
		"""
		windows = []
		for ds_idx, (feature_dataset, target_dataset) in enumerate(
				zip(self.normalized_features, self.normalized_targets)):
			n_samples = feature_dataset.shape[0]
			n_windows = n_samples - self.feature_win + self.feature_target_intersect - self.target_win + 1
			for win_idx in range(n_windows):
				feature_window = feature_dataset[win_idx: win_idx + self.feature_win]
				target_window = target_dataset[
								win_idx + self.feature_win - self.feature_target_intersect:
								win_idx + self.feature_win - self.feature_target_intersect + self.target_win
								]
				windows.append((feature_window, target_window))
		return windows

	def __len__(self) -> int:
		return len(self.windows)

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		return self.windows[idx]


if __name__ == '__main__':
	n_datasets, n_features, n_targets = 100, 4, 3
	fixed_history_size = 3000
	feature_win, target_win, intersect = 120, 1, 0
	flmts = FixedLenMultiTimeSeries(
		features=torch.rand(n_datasets, fixed_history_size, n_features),
		targets=torch.rand(n_datasets, fixed_history_size, n_targets),
		feature_win=feature_win,
		target_win=target_win,
		feature_target_intersection=intersect
	)
	x_flmts, y_flmts = next(iter(flmts))

	histories = [random.randint(fixed_history_size - 100, fixed_history_size + 100) for _ in range(n_datasets)]
	vlmts = VariableLenMultiTimeSeries(
		features=[torch.rand(history_size, n_features) for history_size in histories],
		targets=[torch.rand(history_size, n_targets) for history_size in histories],
		feature_win=feature_win,
		target_win=target_win,
		feature_target_intersection=intersect
	)
	x_vlmts, y_vlmts = next(iter(vlmts))
