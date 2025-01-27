from abc import ABC

import numpy as np
import pandas as pd
import torch
from loguru import logger

DataType = pd.DataFrame | torch.Tensor


class Preprocess(ABC):
	def __init__(self, preprocessing_tasks: list[str] | None):
		self.preprocessing_tasks = preprocessing_tasks or []
		self.task_map = {
			method: getattr(self, method) for method in
			[
				func for func in dir(self) if
				callable(getattr(self, func))
				and not func.startswith("_")
				and func != 'run'
			]
		}
		self._force_cols = ['F1', 'F2', 'F3', 'F4']
		self._angle_cols = ['phi', 'theta', 'psi']
		self._points_cols = ['p0', 'p1', 'p2']

	def run(
			self,
			data: DataType,
			**kwargs
	) -> DataType:
		for task in self.preprocessing_tasks:
			func = self.task_map.get(task)
			if func:
				data = func(data, **kwargs)
			else:
				logger.warning(f"Could not find preprocess function {func}, moving on.")
		return data


class DataFramePreprocess(Preprocess):
	def __init__(self, preprocessing_tasks: list[str] | None):
		"""
		a preprocessor class for dataframe data types
		:param preprocessing_tasks:
		"""
		super(DataFramePreprocess, self).__init__(preprocessing_tasks)

	def resample(
			self,
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		"""
		resamples the data with frequency 5000hz = 1/5000sec = 1/200microsec
		:param df:
		:return:
		"""

		def xyz_mean(batch):
			sum1, sum2, sum3 = 0, 0, 0
			for item in batch:
				sum1 += item[0]
				sum2 += item[1]
				sum3 += item[2]
			n = len(batch)
			return np.array([round(sum1 / n, 5), round(sum2 / n, 5), round(sum3 / n, 5)])

		resample_freq = kwargs.get('resample_freq', '200us')
		logger.info(f'Resampling with freq = {resample_freq}')
		agg_dict = {col: xyz_mean if col in self._points_cols else 'mean' for col in df.columns}

		df = df.resample(resample_freq).agg(agg_dict)
		return df

	@staticmethod
	def interpolate(
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		limit = kwargs.get('interpolate_limit', 20)
		method = kwargs.get('interpolate_method', 'linear')
		order = kwargs.get('interpolate_order', None)
		logger.info(f"Interpolating with limit = {limit}, method = {method} and order = {order}")
		df = df.interpolate(method=method, order=order, limit=limit)
		return df

	@staticmethod
	def drop(
			df: pd.DataFrame,
			**kwargs
	) -> pd.DataFrame:
		cols_to_drop = kwargs.get('drop_cols', ['theta'])
		logger.info(f'Dropping columns = {cols_to_drop}')
		df = df.drop(columns=[cols_to_drop])
		return df


if __name__ == '__main__':
	_df = pd.DataFrame([
		(0.0, np.nan, -1.0, 1.0),
		(np.nan, 2.0, np.nan, np.nan),
		(2.0, 3.0, np.nan, 9.0),
		(np.nan, 4.0, -4.0, 16.0)
	], columns=list('abcd'))
