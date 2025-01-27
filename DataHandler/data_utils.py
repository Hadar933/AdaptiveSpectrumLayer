import pandas as pd
from scipy.stats.qmc import LatinHypercube
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def from_01_to_ab(x: np.ndarray, a: float, b: float) -> np.ndarray:
	"""
	assumes the data x  is in [0,1], and casts it to [a,b]
	:param x: input data (n,)
	:param a: min range
	:param b: max range
	:return: x with fixed range
	"""
	return a + x * (b - a)


def latin_hyper_cube_sampling(
		n_samples: int,
		dim: int = 3,
		show_plot: bool = True,
		save_to_csv: bool = False,
		orthogonal_sampling: bool = False,
		seed: int = 42
) -> np.ndarray:
	"""
	creates an engine of LHC sampler that samples new data points.
	:param n_samples: number of samples to sample
	:param dim: dimension of a sample
	:param show_plot: if true, shows the resulted samples in a scatter plot
	:param save_to_csv: if true, save the resultant np array to memory
	:param seed: for deterministic results
	:param orthogonal_sampling: if true uses orthogonal sampling. otherwise uses lhc sampling
	:return: a np array with shape (n,d) where n is the number of samples and d is the sample dimension
	"""
	engine = LatinHypercube(
		d=dim, strength=2 if orthogonal_sampling else 1,
		seed=seed,
		scramble=True,
		optimization='lloyd'
	)
	lhc_data = engine.random(n=n_samples)
	lhc_data[:, 0] = from_01_to_ab(lhc_data[:, 0], Aa, Ab)
	lhc_data[:, 1] = from_01_to_ab(lhc_data[:, 1], ka, kb)
	lhc_data[:, 2] = from_01_to_ab(lhc_data[:, 2], fa, fb)

	if show_plot and dim == 3:
		matplotlib.use('TkAgg')
		fig = plt.figure()
		lhc_ax = fig.add_subplot(111, projection='3d')
		lhc_ax.scatter(lhc_data[:, 0], lhc_data[:, 1], lhc_data[:, 2])
		lhc_ax.set_title(r'LHC with 160 samples, $f\in [5,20], K\in [0.01,1], A\in[1/6,1/3.5]$')
		lhc_ax.set_xlabel(r'$A$')
		lhc_ax.set_ylabel(r'$K$')
		lhc_ax.set_zlabel(r'$f$')
		plt.show()
	if save_to_csv:
		df = pd.DataFrame(lhc_data, columns=['A', 'K', 'f']).round(3)
		df.to_csv('exp_to_run.csv')
	return lhc_data


if __name__ == '__main__':
	fa, fb = 5.0, 20.0
	ka, kb = 0.01, 1.0
	Ab, Aa = 1/3.5, 1/6
	samples = latin_hyper_cube_sampling(160, 3)
