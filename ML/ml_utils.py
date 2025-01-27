import os.path
from itertools import product
from typing import Any, Optional
from Utilities import utils
import pandas as pd
from loguru import logger

from ML.Core.trainer import Trainer


def generate_hyperparam_combinations(
		model_args: dict,
		global_args: dict,
		model_args_key: str,
		model_shared_pairs: Optional[dict[str, str]] = None,
		global_shared_pairs: Optional[dict[str, str]] = None
) -> list[dict[str, Any]]:
	"""
	generates a cross product of hyperparameters to consider
	:param model_args: a dictionary with keys that represent specific model argument and a value that is a list
						 of relevant hyperparams to consider
	:param global_args: like model_args but unrelated to the model itself (not part of model init arguments)
	:param model_args_key: the key directing to the model arguments sub dictionary
	:param model_shared_pairs: a dictionary that maps model arg name(s) to other model arg name(s), that indicates if
							   the former should be set the same as the latter (and not part of the product)
	:param global_shared_pairs: same as model_shared_pairs, just for the global arguments.
	:return: a list of dictionary that represents runtime arguments, which is the cross product of all given args.
	"""
	model_args_product = product(*(model_args[param] for param in model_args))
	model_param_names = model_args.keys()
	model_args_product_dicts = []
	for model_combination in model_args_product:
		combination_dict = dict(zip(model_param_names, model_combination))
		if model_shared_pairs is not None:
			for new_model_arg_key, existing_model_arg_key in model_shared_pairs.items():
				combination_dict[new_model_arg_key] = combination_dict[existing_model_arg_key]
		model_args_product_dicts.append(combination_dict)

	global_args_product = product(*(global_args[param] for param in global_args))
	global_param_names = global_args.keys()
	global_args_product_dicts = []
	for global_combination in global_args_product:
		combination_dict = dict(zip(global_param_names, global_combination))
		if global_shared_pairs is not None:
			for new_global_arg_key, existing_global_arg_key in model_shared_pairs.items():
				combination_dict[new_global_arg_key] = combination_dict[existing_global_arg_key]
		global_args_product_dicts.append(combination_dict)

	hyperparam_combinations = [
		{**{model_args_key: model_args_dict}, **global_args_dict} for global_args_dict in global_args_product_dicts for
		model_args_dict in model_args_product_dicts
	]
	return hyperparam_combinations


def predict_multiple_models(models_dirs: list[str], dataset_name: str, add_inputs: bool, plot: bool) -> pd.DataFrame:
	"""
	given a list of pretrained model directory paths runs a prediction and aggregates the losses and predictions
	:param models_dirs: list of model directory paths
	:param dataset_name: either train val or test
	:param add_inputs: if True, stacks the input to the prediction datafarme (TODO)
	:return: the loss dataframe for each model and each dataset.
	"""
	all_losses = pd.DataFrame()
	for model_dir in models_dirs:
		model_name = os.path.basename(model_dir)
		trainer = Trainer.from_model_dirname(model_dir)
		preds, losses = trainer.predict(dataset_name, add_inputs)
		if plot:
			utils.plot_model_predictions(preds)
		all_losses = pd.concat([all_losses, losses.rename(index={losses.index[-1]: model_name})], axis=0)
	all_losses['mean'] = all_losses.mean(axis=1)
	return all_losses


if __name__ == '__main__':
	results = predict_multiple_models(
		models_dirs=[
			# == OURS:
			# ==== LINEAR:
			# "/home/hadar/Thesis/ML/saved_models/LTSFLinear[128,1,4,True]_[ours,T=1_2024-01-04_15-25-42",
			# "/home/hadar/Thesis/ML/saved_models/LTSFLinear[128,16,4,False]_[ours,T=16_2024-01-04_15-27-00",
			# "/home/hadar/Thesis/ML/saved_models/LTSFLinear[512,32,4,False]_[ours,T=32_2024-01-04_15-37-24",
			# "/home/hadar/Thesis/ML/saved_models/LTSFLinear[512,64,4,False]_[ours,T=64_2024-01-04_15-38-31",
			# "/home/hadar/Thesis/ML/saved_models/LTSFLinear[128,128,4,False]_[ours,T=128_2024-01-04_15-29-26",

			# ==== TRANSFORMER:
			# "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[1, False, 4, 16, None, None, 0.05, 3, '3', 1, 16, 1, 'gelu', 2, 2, 3]_[ours,T=1]_2024-01-02_04-03-09",
			# "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[16, False, 4, 16, None, None, 0.05, 3, '3', 1, 16, 2, 'gelu', 2, 2, 3]_[ours,T=16]_2024-01-01_09-10-33",
			# "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[32, False, 4, 16, None, None, 0.05, 3, '3', 1, 16, 2, 'gelu', 2, 2, 3]_[ours,T=32]_2023-12-31_14-26-35",
			# "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[64, False, 4, 32, None, None, 0.05, 3, '3', 1, 32, 1, 'gelu', 2, 2, 3]_[ours,T=64]_2023-12-31_19-34-47",
			# "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[128, False, 4, 16, None, None, 0.05, 3, '3', 1, 16, 2, 'gelu', 2, 1, 3]_[ours,T=128]_2024-01-01_00-15-22",

			# ==== INFORMER:
			# "/home/hadar/Thesis/ML/saved_models/LTSFInformer[16, False, 4, 64, None, None, 0.05, 3, '3', 1, 64, 1, 'gelu', 2, 2, 3]_[ours,T=16_2024-01-04_22-43-27",
			# "/home/hadar/Thesis/ML/saved_models/LTSFInformer[32, False, 4, 16, None, None, 0.05, 3, '3', 1, 16, 2, 'gelu', 2, 1, 3]_[ours,T=32_2024-01-04_17-19-27",
			# "/home/hadar/Thesis/ML/saved_models/LTSFInformer[64, False, 4, 32, None, None, 0.05, 3, '3', 1, 32, 1, 'gelu', 2, 1, 3]_[ours,T=64_2024-01-04_18-58-12",
			# "/home/hadar/Thesis/ML/saved_models/LTSFInformer[128, False, 4, 64, None, None, 0.05, 3, '3', 1, 64, 1, 'gelu', 2, 2, 3]_[ours,T=128_2024-01-04_20-54-50"

			# ==== SEQ2SEQ:
			# '/home/hadar/Thesis/ML/saved_models/Input(512, 256, 4)seq2seq[1,8,16,1,True,8,16,3]_[ours,T=1_2024-01-22_22-51-07',
			# "/home/hadar/Thesis/ML/saved_models/Input(512, 256, 4)seq2seq[1,8,16,1,True,8,16,3]_ResidualReLU[ours,T=1]_2024-01-05_22-36-41",
			# "/home/hadar/Thesis/ML/saved_models/Input(512, 128, 4)seq2seq[16,8,16,1,True,8,16,3]_ResidualReLU[ours,T=16]_2024-01-05_02-42-17",
			# "/home/hadar/Thesis/ML/saved_models/Input(512, 128, 4)seq2seq[32,8,16,1,True,8,16,3]_ResidualReLU[ours,T=32]_2024-01-05_04-43-14",
			# "/home/hadar/Thesis/ML/saved_models/Input(512, 128, 4)seq2seq[64,4,16,1,True,4,16,3]_ResidualReLU[ours,T=64]_2024-01-05_07-26-00",
			# "/home/hadar/Thesis/ML/saved_models/Input(512, 128, 4)seq2seq[128,8,16,1,True,8,16,3]_ResidualReLU[ours,T=128]_2024-01-05_15-46-45"
			# ==== SEQ2SEQ w/ASL:

			# ==== AUTOFORMER:
			# "/home/hadar/Thesis/ML/saved_models/LTSFAutoFormer[1, False, 4, 8, None, None, 0.05, 4, '3', 1, 8, 2, 'gelu', 2, 2, 4, 25]_[ours,T=1_2024-01-10_16-35-19",
			# "/home/hadar/Thesis/ML/saved_models/LTSFAutoFormer[16, False, 4, 8, None, None, 0.05, 4, '3', 1, 8, 1, 'gelu', 2, 2, 4, 25]_[ours,T=16_2024-01-10_18-35-49",
			# "/home/hadar/Thesis/ML/saved_models/LTSFAutoFormer[32, False, 4, 8, None, None, 0.05, 4, '3', 1, 8, 2, 'gelu', 2, 2, 4, 25]_[ours,T=32_2024-01-10_20-42-52",
			# "/home/hadar/Thesis/ML/saved_models/LTSFAutoFormer[64, False, 4, 4, None, None, 0.05, 4, '3', 1, 4, 2, 'gelu', 2, 1, 4, 25]_[ours,T=64_2024-01-10_23-03-30",
			# "/home/hadar/Thesis/ML/saved_models/LTSFAutoFormer[128, False, 4, 4, None, None, 0.05, 4, '3', 1, 4, 1, 'gelu', 2, 1, 4, 25]_[ours,T=128_2024-01-11_01-22-24",

			# ==== FEDFORMER:

			# == PRSSM:
			# ==== LINEAR:
			# "/home/hadar/Thesis/ML/saved_models/LTSFLinear[256,1,5,True]_[prssm,T=1]_2024-01-05_23-57-40",
			# "/home/hadar/Thesis/ML/saved_models/LTSFLinear[256,16,5,True]_[prssm,T=16]_2024-01-05_23-58-38",
			# "/home/hadar/Thesis/ML/saved_models/LTSFLinear[256,32,5,True]_[prssm,T=32]_2024-01-05_23-59-36",
			# "/home/hadar/Thesis/ML/saved_models/LTSFLinear[256,64,5,True]_[prssm,T=64]_2024-01-06_00-00-28",
			# "/home/hadar/Thesis/ML/saved_models/LTSFLinear[256,128,5,True]_[prssm,T=128]_2024-01-06_00-01-10",

			# ==== TRANSFORMER
			# "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[1, False, 5, 64, None, None, 0.05, 3, '3', 1, 64, 2, 'gelu', 2, 2, 3]_[prssm,T=1]_2024-01-06_06-41-08",
			# "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[16, False, 5, 64, None, None, 0.05, 3, '3', 1, 64, 2, 'gelu', 2, 2, 3]_[prssm,T=16]_2024-01-06_07-53-12",
			# "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[32, False, 5, 64, None, None, 0.05, 3, '3', 1, 64, 2, 'gelu', 2, 2, 3]_[prssm,T=32]_2024-01-06_09-05-13",
			# "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[64, False, 5, 64, None, None, 0.05, 3, '3', 1, 64, 2, 'gelu', 2, 2, 3]_[prssm,T=64]_2024-01-06_10-10-48",
			# "/home/hadar/Thesis/ML/saved_models/LTSFVanillaTransformer[128, False, 5, 64, None, None, 0.05, 3, '3', 1, 64, 2, 'gelu', 2, 2, 3]_[prssm,T=128]_2024-01-06_11-03-36",

			# ==== INFORMER
			# "/home/hadar/Thesis/ML/saved_models/LTSFInformer[16, False, 5, 64, None, None, 0.05, 3, '3', 1, 64, 2, 'gelu', 2, 2, 3]_[prssm,T=16]_2024-01-06_10-58-21",
			# "/home/hadar/Thesis/ML/saved_models/LTSFInformer[32, False, 5, 64, None, None, 0.05, 3, '3', 1, 64, 2, 'gelu', 2, 2, 3]_[prssm,T=32]_2024-01-06_11-46-14",
			# "/home/hadar/Thesis/ML/saved_models/LTSFInformer[64, False, 5, 64, None, None, 0.05, 3, '3', 1, 64, 2, 'gelu', 2, 2, 3]_[prssm,T=64]_2024-01-06_12-22-23",
			# "/home/hadar/Thesis/ML/saved_models/LTSFInformer[128, False, 5, 64, None, None, 0.05, 3, '3', 1, 64, 2, 'gelu', 2, 2, 3]_[prssm,T=128]_2024-01-06_12-46-52",

			# ==== SEQ2SEQ:
			"/home/hadar/Thesis/ML/saved_models/Input(512, 256, 5)seq2seq[1,130,90,1,True,130,90,3]_[prssm,T=1]_2024-01-08_20-50-16",
			# "/home/hadar/Thesis/ML/saved_models/Input(512, 256, 5)seq2seq[16,70,130,1,True,70,130,3]_[prssm,T=16]_2024-01-08_22-12-53",
			# "/home/hadar/Thesis/ML/saved_models/Input(512, 256, 5)seq2seq[32,120,120,1,True,120,120,3]_[prssm,T=32]_2024-01-08_01-03-14",
			# "/home/hadar/Thesis/ML/saved_models/Input(512, 256, 5)seq2seq[64,120,120,1,True,120,120,3]_[prssm,T=64]_2024-01-08_06-17-06",
			# "/home/hadar/Thesis/ML/saved_models/Input(512, 256, 5)seq2seq[128,50,70,1,True,50,70,3]_[prssm,T=128]_2024-01-09_21-28-34"

			# ==== AUTOFORMER:
			# "/home/hadar/Thesis/ML/saved_models/LTSFAutoFormer[1, False, 5, 8, None, None, 0.05, 5, '3', 1, 8, 2, 'gelu', 2, 1, 5, 25]_[prssm,T=1_2024-01-12_13-56-17",
			# "/home/hadar/Thesis/ML/saved_models/LTSFAutoFormer[16, False, 5, 8, None, None, 0.05, 5, '3', 1, 8, 1, 'gelu', 2, 2, 5, 25]_[prssm,T=16_2024-01-12_15-18-29",
			# "/home/hadar/Thesis/ML/saved_models/LTSFAutoFormer[32, False, 5, 8, None, None, 0.05, 5, '3', 1, 8, 1, 'gelu', 2, 2, 5, 25]_[prssm,T=32_2024-01-12_16-38-56",
			# "/home/hadar/Thesis/ML/saved_models/LTSFAutoFormer[64, False, 5, 8, None, None, 0.05, 5, '3', 1, 8, 1, 'gelu', 2, 2, 5, 25]_[prssm,T=64_2024-01-12_18-01-46",
			# "/home/hadar/Thesis/ML/saved_models/LTSFAutoFormer[128, False, 5, 16, None, None, 0.05, 5, '3', 1, 16, 2, 'gelu', 2, 1, 5, 25]_[prssm,T=128_2024-01-12_19-27-14",

			# ==== FEDFORMER:

		],
		dataset_name='test',
		add_inputs=True,
		plot=True
	)
