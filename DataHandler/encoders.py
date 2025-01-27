from typing import Optional

import numpy as np
import pandas as pd
from scipy import special
from loguru import logger
from scipy import signal


def derivative(
		x,
		dt: float,
		smooth: str = 'between',
		window_length: Optional[int] = None,
		polyorder: Optional[int] = None
) -> tuple:
	"""
	a function that performs 1st and 2nd derivatives with possible smoothing in between
	:param x: signal to derive
	:param dt: time delta between x[i+1] and x[i] for all i=0,1,...,len(x)-1
	:param smooth: one of the following
					- between: x <smooth> x_dot <smooth> x_ddot
					- signal: x <smooth> x_dot x_ddot
					- none: x <smooth> x_dot x_ddot
	:param window_length: possible window length for smoothing
	:param polyorder: possible polyorder for smoothing

	:return:
	"""
	if window_length is None: window_length = 51
	if polyorder is None: polyorder = 3
	smooth = smooth.lower()
	smooth_options = {'between', 'signal', 'none'}
	if smooth == 'none':
		xdot = np.gradient(x, dt)
		xddot = np.gradient(xdot, dt)
	elif smooth == 'signal':
		xdot = signal.savgol_filter(x, window_length, polyorder, deriv=1, delta=dt)
		xddot = np.gradient(xdot, dt)
	elif smooth == 'between':
		xdot = signal.savgol_filter(x, window_length, polyorder, deriv=1, delta=dt)
		xddot = signal.savgol_filter(x, window_length, polyorder, deriv=2, delta=dt)
	else:
		raise ValueError(f'Smoothing scheme `{smooth}` not in {smooth_options}.')

	return xdot, xddot


class Encoder:
	def __init__(self, encoding_tasks: list[str] | None = None):
		self.encoding_tasks = encoding_tasks or []
		self.task_map = {
			method: getattr(self, method) for method in
			[
				func for func in dir(self) if
				callable(getattr(self, func))
				and not func.startswith("_")
				and func != 'run'
			]
		}
		# fallback column names
		self._force_cols = ['F1', 'F2', 'F3', 'F4']
		self._angle_cols = ['phi', 'theta', 'psi']
		self._points_cols = ['p0', 'p1', 'p2']

	def run(
			self,
			data: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		for task in self.encoding_tasks:
			func = self.task_map.get(task)
			if func:
				data = func(data, **kwargs)
			else:
				logger.warning(f"Could not find encoder function {func}, moving on.")
		return data

	def center_of_mass(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols = kwargs.get('center_of_mass_cols', self._points_cols)
		logger.info(f'Encoding center of mass using cols = {cols}')
		p0, p1, p2 = (df[col] for col in cols)
		vec1 = p2 - p0  # AB
		vec2 = p2 - p1  # AC
		lambda1: float = -0.173
		lambda2: float = 0.857
		center_of_mass = lambda1 * vec1 + lambda2 * vec2  # TODO: not working..
		center_of_mass = (p0 + p1 + p2) / 3  # close enough to COM
		df['center_of_mass'] = center_of_mass
		return df

	def inertial_force(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		dt = kwargs.get('inertial_force_dt', (df.index[1] - df.index[0]).total_seconds())
		wing_mass = kwargs.get('inertial_force_wing_mass', 0.0007)  # kg (=3g)
		if 'center_of_mass' not in df.columns:
			df = self.center_of_mass(df, **kwargs)
		logger.info(f'Encoding inertial force using dt={dt}, wing_mass={wing_mass}')
		center_of_mass_y = np.vstack(df['center_of_mass'])[:, 1] / 1000  # the vertical disposition in meters
		dot, ddot = derivative(center_of_mass_y, dt)
		force = wing_mass * ddot
		df['F_inertia'] = force
		return df


	def hermite(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols = kwargs.get('hermite_cols', self._force_cols)
		orders = kwargs.get('orders', [2, 3, 4])
		logger.info(f'Encoding hermite with cols = {cols} and orders = {orders}')
		for n in orders:
			col_names = [f"hermite_{n}({col})" for col in cols]
			poly_h = special.hermite(n)
			df[col_names] = poly_h(df)
		return df

	def sin_cos(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols = kwargs.get('angle_cols', self._angle_cols)
		logger.info(f'Encoding angle with cols = {cols}')
		sin_col_names = [f"sin({col})" for col in cols]
		cos_col_names = [f"cos({col})" for col in cols]
		arg = 2 * np.pi * df[cols] / df[cols].max()
		df[sin_col_names] = np.sin(arg)
		df[cos_col_names] = np.cos(arg)
		return df

	def torque(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		radius = kwargs.get('encode_torque_radius', 0.06)  # in meters (=6cm)
		logger.info(f'Encoding torque with radius = {radius}')
		f1, f2, f3, f4 = df['F1'], df['F2'], df['F3'], df['F4']
		df['torque_x'] = (radius / 2) * ((f1 + f2) - (f3 + f4))
		df['torque_y'] = (radius / 2) * ((f2 + f3) - (f1 + f4))
		return df

	def sum_cols(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols = kwargs.get('sum_cols', self._force_cols)
		logger.info(f'Encoding Sum of {cols}')
		df['+'.join(cols)] = df[cols].sum(axis=1)
		return df


if __name__ == '__main__':
	from Utilities import utils

	exp_date = '19_10_2023'
	exp_name = '[F=7.886_A=M_PIdiv5.401_K=0.03]'
	merge_name = 'merged_data.pkl'
	merge_pp_name = 'merged_data_preprocessed_and_encoded.pkl'
	_df = pd.read_pickle(fr"E:\Hadar\experiments\{exp_date}\results\{exp_name}\{merge_pp_name}")
	enc = Encoder()
	_df = _df.rename(columns={'F_inertia': 'F_inertia_old'})

	df1 = enc.inertial_force(_df)

	_df['com_y'] = np.vstack(_df['center_of_mass'])[:, 1] / 1000
	_df['F1+F2+F3+F4-F_inertia'] = _df['F1+F2+F3+F4'] - _df['F_inertia']

	utils.plot_df_with_plotly(df1.drop(columns=['F_inertia_old']).iloc[:2000], ignore_cols='p0 p1 p2 center_of_mass'.split())
	z=2