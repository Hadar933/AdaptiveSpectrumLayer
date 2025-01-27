import os.path
import re
import subprocess

import dash
import matplotlib
import torch
import imageio
import scipy
import json

import plotly.offline as pyo
import numpy as np
import pandas as pd
from dash import html, dcc, Output, Input

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from plotly import graph_objects as go
from plotly_resampler import FigureResampler
from tqdm import tqdm
from typing import Literal, Optional

from ML.Core import trainer
from ML.Core.datasets import FixedLenMultiTimeSeries, VariableLenMultiTimeSeries
from ML.Zoo.Seq2seq import Seq2seq

from DataHandler import encoders

TIME_FORMAT = '%Y-%m-%d_%H-%M-%S'


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                               PLOTTING                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

def plot_fourier_reconstruction(x, fft, weights, reconstructed_x, fftfreq):
	"""
	run when inside ASL forawrd, using:
	plot_fourier_reconstruction(x[0], fft[0], weights.squeeze(1).permute(0, 2, 1)[0], reconstructed_x[0], self.fftfreq)
	Plot the input signal, its Fourier transform, learned weights, and reconstructed signal."""
	# Convert tensors to NumPy arrays
	x_np = x.detach().cpu().numpy()
	fft_np = fft.detach().cpu().numpy()
	weights_np = weights.detach().cpu().numpy()
	reconstructed_x_np = reconstructed_x.detach().cpu().numpy()

	# Create Pandas DataFrames
	x_df = pd.DataFrame(x_np, columns=[fr'$x_{i}$' for i in range(1, x_np.shape[1] + 1)],
						index=np.arange(len(x_np)) * (1 / 5000))
	fft_df = pd.DataFrame(np.abs(fft_np), columns=[fr'$|FFT[x]|_{i}$' for i in range(1, fft_np.shape[1] + 1)],
						  index=fftfreq.numpy())
	weights_df = pd.DataFrame(weights_np, columns=[fr'$W_{i}$' for i in range(1, weights_np.shape[1] + 1)],
							  index=fftfreq.numpy())
	reconstructed_x_df = pd.DataFrame(reconstructed_x_np,
									  columns=[fr'$x_{i}$' for i in range(1, reconstructed_x_np.shape[1] + 1)],
									  index=np.arange(len(reconstructed_x_np)) * (1 / 5000))

	# Plot using Pandas
	fig, axs = plt.subplots(4, 1, figsize=(10, 10))

	x_df.plot(ax=axs[0], title=r'Input $x$')
	axs[0].set_xlabel('Time [sec]')
	axs[0].set_ylabel('Normalized Force [N]')

	fft_df.plot(ax=axs[1], title=r'$|FFT[x]|$')
	axs[1].set_xlabel('Frequency (Hz)')
	axs[1].set_ylabel('Magnitude')
	# axs[1].set_yscale('log')
	weights_df.plot.bar(ax=axs[2], title=r'Learned weights $W$')  # Changed to bar plot
	axs[2].set_xlabel('Frequency (Hz)')
	axs[2].set_ylabel('Magnitude')
	axs[2].tick_params(axis='x', rotation=45)  # Rotate x-axis labels by 45 degrees
	labels = [item.get_text() for item in axs[2].get_xticklabels()]
	# Beat them into submission and set them back again
	axs[2].set_xticklabels([str(round(float(label), 2)) for label in labels])
	reconstructed_x_df.plot(ax=axs[3], title=r'Reconstructed $x$')
	axs[3].set_xlabel('Time [sec]')
	axs[3].set_ylabel('Normalized Force [N]')
	for ax in axs:
		ax.legend(loc='upper right')
		ax.grid(True)
	plt.tight_layout()
	plt.show()


def plot_model_weights(trainer, attention_plot=False, fourier_plot=False, in_time=True, cmap='Blues'):
	"""
	plot the weights and the test batch in the same figure
	"""
	test_batch = next(iter(trainer.data_dict['test']['loader']))[0]  # (B,T,F)
	seq2seq = trainer.model.to('cpu')
	if in_time:
		_ = seq2seq(test_batch)
		fft_weights = seq2seq.adl1.fft_weights.detach().numpy().squeeze()  # (B,N_freqs,F)
		fft_freqs = seq2seq.adl1.fftfreq.detach().numpy().squeeze()  # (N_freqs,)
		attn_weights = seq2seq.attention.attn_weights.detach().numpy().squeeze()  # (B,T)
		# note that for test, batch samples are consecutive in time.
		fig, axs = plt.subplots(1, 2, figsize=(15, 5))
		# Plot the Fourier weights
		im1 = axs[0].imshow(fft_weights[:, :, 3], cmap=cmap, interpolation='nearest', aspect='auto')
		axs[0].set_xlabel('Frequency (Hz)')
		axs[0].set_ylabel('Window')
		axs[0].set_title('Fourier Weights for Shifted Windows')
		fig.colorbar(im1, ax=axs[0])
		axs[0].set_xticks(range(len(fft_freqs)))
		axs[0].set_xticklabels(fft_freqs.astype(np.int16))
		axs[0].tick_params(axis='x', rotation=45)
		# for label in axs[0].xaxis.get_ticklabels()[::2]:
		#     label.set_visible(False)
		# Plot the attention weights
		im2 = axs[1].imshow(attn_weights, cmap=cmap, interpolation='nearest', aspect='auto',
							norm=matplotlib.colors.LogNorm(vmin=attn_weights.min(), vmax=attn_weights.max()))
		axs[1].set_xlabel('Time Step')
		axs[1].set_ylabel('Window')
		axs[1].set_title('Attention Weights for Shifted Windows')
		fig.colorbar(im2, ax=axs[1])
		plt.tight_layout()
		plt.show()

	else:
		i = 0  # sample index
		x = test_batch[[i]]  # (1,T,F)
		_ = seq2seq(x)
		fft_weights = seq2seq.adl1.fft_weights.detach().numpy().squeeze()  # (N_freqs,F)
		fft_magnitude = seq2seq.adl1.fft_magnitude.detach().numpy().squeeze()  # (N_freqs,F)
		fft_freqs = seq2seq.adl1.fftfreq.detach().numpy().squeeze()  # (N_freqs,)
		attn_weights = seq2seq.attention.attn_weights.detach().numpy().squeeze()  # (T,)
		if fourier_plot:
			fig, axs = plt.subplots(1, fft_weights.shape[1], figsize=(16, 5))
			for i in range(fft_weights.shape[1]):  # Loop over each feature F
				im = axs[i].imshow(np.vstack(fft_weights[:, i]).T,
								   extent=(0, len(fft_weights), fft_magnitude[:, i].min(), fft_magnitude[:, i].max()),
								   aspect='auto', cmap=cmap)
				axs[i].plot(fft_magnitude[:, i])
				axs[i].set_title(fr'$|FFT[Input=F_{i + 1}]|$', fontsize=14)
				axs[i].set_xticks(range(len(fft_freqs)))
				axs[i].set_xticklabels(fft_freqs.astype(np.int16))
				axs[i].tick_params(axis='x', rotation=45)
				for label in axs[i].xaxis.get_ticklabels()[::2]:
					label.set_visible(False)
			fig.text(0.5, 0.02, 'Frequency (Hz)', ha='center', va='center', fontsize=14)
			fig.text(0.01, 0.5, 'Amplitude', ha='center', va='center', rotation='vertical', fontsize=14)
			cbar = fig.colorbar(im, ax=axs[i])
			cbar.set_label('Weight Magnitude', fontsize=14)
			plt.tight_layout()
			plt.show()

		if attention_plot:
			plt.figure(figsize=(15, 5))
			plt.imshow(np.vstack(attn_weights).T, extent=(0, len(attn_weights), x.min(), x.max()), aspect='auto',
					   cmap=cmap)
			plt.plot(x.squeeze().numpy())
			plt.colorbar()
			plt.title(f'Attention-Weighted Input Window')
			plt.xlabel('Frame #')
			plt.ylabel('Amplitude')
			plt.legend(['F1', 'F2', 'F3', 'F4'])
			plt.show()


def plot_model_predictions(
		pred_dict: dict[int, pd.DataFrame],
		datasets: Optional[list[int]] = None,
		trim: Optional[int] = None,
		fontsize: int = 14
) -> None:
	"""
	takes in a dictionary of datasets predictions and plots them side by side
	:param pred_dict: dictionary where each value is a df with inputs, true labels and predictions.
					  These are generated using the `.predict` method of a `Trainer` instance
	:param datasets: a list of dataset keys to plot (the dictionary will be trimmed accordingly)
	:param trim: an integer that slices each dataframe (all rows > trim are dropped)
	:param fontsize: fontsize for labels and legends
	"""
	if datasets is not None:
		pred_dict = {key: value for key, value in pred_dict.items() if key in datasets}
	num_dataframes = len(pred_dict)
	fig, axs = plt.subplots(2, num_dataframes, sharex=True, sharey='row', figsize=(len(pred_dict) * num_dataframes, 10))
	force_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown']
	pred_true_colors = ['tab:purple', 'tab:cyan', 'tab:pink']
	for idx, (df_name, df) in enumerate(pred_dict.items()):
		df = df.iloc[:trim, :] if trim is not None else df
		inputs_cols = [col for col in df.columns if 'input' in col.lower()]
		true_labels_cols = [col for col in df.columns if 'true' in col.lower()]
		predictions_cols = [col for col in df.columns if 'pred' in col.lower()]
		inputs = df[inputs_cols].values
		true_labels = df[true_labels_cols].values
		predictions = df[predictions_cols].values
		# Plotting inputs in the top row:
		input_names = ['Fx', 'Fy', 'Fz', 'My', 'Mz'] if len(inputs_cols) == 5 else ['F1', 'F2', 'F3', 'F4']
		for i, label in zip(range(inputs.shape[1]), input_names):
			axs[0, idx].plot(inputs[:, i], label=label, color=force_colors[i])
		# Plotting targets and predictions in the bottom row:
		labels = [r"$\theta$", r"$\phi$", r"$\psi$"]
		nan_index = np.where(np.isnan(true_labels[:, 0]))[0][0]
		for i in range(true_labels.shape[1]):
			axs[1, idx].plot(true_labels[:nan_index, i], label=f'True {labels[i]}', linestyle='dashed',
							 color=pred_true_colors[i])
			axs[1, idx].plot(predictions[:nan_index, i], label=f'Pred {labels[i]}', linestyle='solid',
							 color=pred_true_colors[i])
		axs[1, idx].set_xlim(0, nan_index)
	axs[0, 0].set_ylabel('Force and Torque Values [N] or [Nm]', fontsize=fontsize)
	axs[0, idx].legend(loc='lower right', fontsize=fontsize)
	axs[1, idx].legend(loc='lower right', fontsize=fontsize)
	axs[1, 0].set_ylabel('Angle Targets & Predictions [rad]', fontsize=fontsize)
	trim_suffix = f" (Trimmed to {trim} samples)" if trim is not None else ""
	fig.suptitle(f"Forces, Kinematics, and Predictions across Multiple Datasets {trim_suffix}", fontsize=fontsize + 2)
	fig.text(0.5, 0.04, 'Frames', ha='center', fontsize=fontsize)
	plt.tight_layout(rect=[0, 0.06, 1, 0.99])
	for ax in axs.flatten():        ax.grid(True)
	plt.show()


def results_plotter(
		kinematics_path: str,
		forces_path: str
) -> None:
	""" an interactive dash plot for the kinematics and forces torch tensors """
	# matplotlib.use('TkAgg')

	kinematics = torch.load(kinematics_path)
	forces = torch.load(forces_path)
	app = dash.Dash(__name__)
	app.layout = html.Div([
		html.H1("Kinematics and Forces Data Visualization"),
		dcc.Dropdown(
			id='dataset-dropdown',
			options=[{'label': f'Dataset {i}', 'value': i} for i in
					 range(len(kinematics) if isinstance(kinematics, list) else kinematics.size(0))],
			value=0  # Default value
		),
		dcc.Graph(id='combined-graph')
	])

	@app.callback(
		Output('combined-graph', 'figure'),
		[Input('dataset-dropdown', 'value')]
	)
	def update_graph(selected_dataset):
		""" updates the graph based on the dataset index chosen in the dropdown menu """
		phi_trace = go.Scatter(y=kinematics[selected_dataset][:, 0].numpy(), mode='lines', name=r'phi')
		psi_trace = go.Scatter(y=kinematics[selected_dataset][:, 1].numpy(), mode='lines', name=r'psi')
		theta_trace = go.Scatter(y=kinematics[selected_dataset][:, 2].numpy(), mode='lines', name=r'theta')
		force_traces = [go.Scatter(y=forces[selected_dataset][:, i].numpy(), mode='lines', name=fr'F_{i + 1}')
						for i in range(4)]
		traces = [phi_trace, psi_trace, theta_trace] + force_traces
		layout = go.Layout(
			title=f'Kinematics and Forces for Experiment {selected_dataset}',
			xaxis=dict(title='Sample'),
			yaxis=dict(title='Value')
		)
		figure = {'data': traces, 'layout': layout}
		return figure

	app.run_server(debug=True)


def plot_df_with_plotly(
		df_dict: dict[int, pd.DataFrame],
		ignore_cols: list[str] | None = None,
		title: str = "Data vs Time",
		x_title: str = "time / steps",
		y_title: str = "Data",
		save_path: str | None = None
) -> None:
	"""plots a df with plotly resampler"""
	df = pd.concat(df_dict.values(), axis=1)
	if ignore_cols is None:
		ignore_cols = []
	fig = FigureResampler(go.Figure())
	cols_to_plot = [col for col in df.columns if col not in ignore_cols]
	if isinstance(df.index, pd.TimedeltaIndex):
		x_axis = df.index.total_seconds()
		x_title = 'time [sec]'
	else:
		x_axis = df.index
	for col in cols_to_plot:
		fig.add_trace(go.Scattergl(name=col, showlegend=True), hf_x=x_axis, hf_y=df[col])
	fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, margin=dict(l=20, r=20, t=30, b=0),
					  height=700)
	if save_path is not None:
		fig.write_html(save_path)
	pyo.plot(fig, filename='temp-plot.html', auto_open=True)


def plot_dict_with_plotly(
		data: dict[str, list],
		title: str = "Data vs Time",
		x_title: str = "time / steps",
		y_title: str = "Data",
		save_path: str | None = None
) -> None:
	"""plots a df with plotly resampler"""
	fig = FigureResampler(go.Figure())
	for key in data.keys():
		fig.add_trace(go.Scattergl(name=key, showlegend=True), hf_x=data.index, hf_y=data[key])
	fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, margin=dict(l=20, r=20, t=30, b=0),
					  height=700)
	if save_path is not None:
		fig.write_html(save_path)
	pyo.plot(fig, filename='temp-plot.html', auto_open=True)


def visualize_attention(
		model: Seq2seq,
		test_loader
) -> list[np.ndarray]:
	"""
	@deprecated
	:param model: a sequence to sequence torch model
	:param test_loader: a loader that contains test data
	:return:
	"""
	model.train(False)
	device = torch.device('cpu')
	model = model.to(device)
	images = []
	x_start, x_end = -1, test_loader.dataset.feature_win - 1  # moving x axis
	with torch.no_grad():
		for i, (inputs_i, true_i) in tqdm(enumerate(test_loader), total=len(test_loader)):
			x_start += 1
			x_end += 1
			inputs_i = inputs_i.to(device)
			_ = model(inputs_i)
			x_vals = torch.arange(x_start, x_end)
			y_vals = inputs_i.squeeze(0)[0]
			weights = model.attention.attn_weights.squeeze(0)[0]
			image = plot_with_background(x_vals, y_vals, weights, 'time', 'value',
										 f'Attention weights [{i}/{len(test_loader)}]')
			images.append(image)
			plt.imshow(image)
			plt.show()
	imageio.mimsave('attention_weights_over_time.mp4', images, fps=10)
	return images


def plot_with_background(
		x_values: torch.Tensor,
		y_values: torch.Tensor,
		w_values: torch.Tensor,
		x_label: str,
		y_label: str,
		title: str
) -> np.ndarray:
	"""
	@deprecated
	plots a figure given x_values and y_values and colors the background using w_values
	:param x_values: x-axis values (time mostly) with shape (win_size,)
	:param y_values: y-axis values (forces mostly) with shape (win_size, input_size)
	:param w_values: color mapping (attention weights mostly) with shape (win_size,)
	:return: an RGB image that represents the colored window plot
	"""
	margin = 0.1
	n_plots = y_values.shape[1]
	fig, ax = plt.subplots(figsize=(8, 6))
	for i in range(n_plots):  # plotting the data itself
		ax.plot(x_values, y_values[:, i])
	cmap = plt.get_cmap('coolwarm')
	for j in range(len(x_values) - 1):  # plotting the background
		color = cmap(w_values[j].item())
		ax.fill_betweenx([y_values.min() - margin, y_values.max() + margin], x_values[j], x_values[j + 1], color=color)
	# Add a colorbar as a color index
	cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position of the colorbar
	norm = plt.Normalize(w_values.min(), w_values.max())  # Normalize colorbar to original 'w' values
	cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
	cb.set_label('w values')
	ax.set_xlim(x_values.min(), x_values.max())
	ax.set_ylim(y_values.min() - margin, y_values.max() + margin)
	ax.set_xlabel(x_label), ax.set_ylabel(y_label), ax.set_title(title)
	fig.canvas.draw()
	image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
	image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	plt.close(fig)
	return image


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                      DATA SPLITTING AND LOADING                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

def load_data_from_prssm_paper(
		path: str = "G:\\My Drive\\Master\\Lab\\Thesis\\DataHandler\\flapping_wing_aerodynamics.mat",
		kinematics_key: Literal['ds_pos', 'ds_u_raw', 'ds_u'] = "ds_pos",
		forces_key: Literal['ds_y_raw', 'ds_y'] = "ds_y_raw",
		return_all: bool = False,
		forces_to_take: list[int] | None = None
) -> dict | tuple[torch.Tensor, torch.Tensor]:
	"""
	loads the data from the PRSSM paper, based on a string that represents the request
	:param path: path to the .mat file
	:param kinematics_key: one of the following options:
						- 'ds_pos' (default): raw stroke, deviation and rotation
						- 'ds_u_raw': The 7 kinematic variables derived from ds_pos
						- 'ds_u': The standardized* versions of ds_u_raw
	:param forces_key: one of the following options:
						- 'ds_y_raw' (default): raw targets Fx (normal), Fy (chord-wise), Mx, My, Mz
						- 'ds_y': the standardized* version of ds_y_raw
	:param return_all: if true, returns the entire data dictionary
	:param forces_to_take: the forces columns to consider. If not specified, takes all of them (5)
	* beware - this is normalized w.r.t an unknown training data
	:return:
	"""
	try:
		mat: dict = scipy.io.loadmat(path)
	except FileNotFoundError:
		mat: dict = scipy.io.loadmat("DataHandler/flapping_wing_aerodynamics.mat")
	if return_all: return mat
	kinematics = torch.Tensor(mat[kinematics_key])
	forces = torch.Tensor(mat[forces_key])
	forces = forces if forces_to_take is None else forces[:, :, forces_to_take]
	return kinematics, forces


def train_val_test_split(
		use_variable_length_dataset: bool,
		features: torch.Tensor,
		targets: torch.Tensor,
		train_percent: float,
		val_percent: float,
		feature_window_size: int,
		target_window_size: int,
		intersect: int,
		batch_size: int,
		features_normalizer,
		targets_normalizer
) -> dict:
	"""
	creates a time series train-val-test dataset-split (not history) for multiple multivariate time series.
	since a prediction should be on a single dataset, we create a dataset (and a dataloader) for each of
	the datasets. In inference time, we can simply choose the dataloader using its index.
	:param use_variable_length_dataset: if true, uses a dataset that is suited for list of tensors [t1,t2,...,tn],
										with each ti shaped as (hi,f). Otherwise, uses a dataset that handles a
										multi-dim tensor with shape (n,h,f), i.e fixed history for all n tensors.
	:param features: the data itself
	:param targets: the target(s)
	:param train_percent: percentage of data to be considered as training data
	:param val_percent: same, just for validation
	:param feature_window_size: the number of samples in every window (history size)
	:param target_window_size: the number of samples in the predicted value (usually one)
	:param intersect: the intersection size between the feature and target windows
	:param batch_size: the batch size for the training/validation sets.
	:param features_normalizer: a normalizer object for the features of the training data
	:param targets_normalizer: a normalizer object for the targets of the training data
	:return: for the training and validation - regular pytorch loader. for the test - a loader for every dataset.
	"""
	DatasetClass = VariableLenMultiTimeSeries if use_variable_length_dataset else FixedLenMultiTimeSeries
	n_datasets = len(features) if use_variable_length_dataset else features.shape[0]
	train_size = int(train_percent * n_datasets)
	val_size = int(val_percent * n_datasets)

	# train loaders and per-dataset class:
	features_train, targets_train = features[:train_size], targets[:train_size]
	features_train = features_normalizer.fit_transform(features_train)  # statistics are set from training data
	targets_train = targets_normalizer.fit_transform(targets_train)  # same
	train_dataset = DatasetClass(features_train, targets_train, feature_window_size, target_window_size, intersect)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	all_train_datasets = [
		DatasetClass(features_train[i].unsqueeze(0), targets_train[i].unsqueeze(0), feature_window_size,
					 target_window_size, intersect) for i in range(train_size)
	]
	all_train_dataloaders = [
		torch.utils.data.DataLoader(all_train_datasets[i], batch_size=1, shuffle=False) for i in
		range(len(all_train_datasets))
	]

	# validation loaders and per-dataset class:
	features_val, targets_val = features[train_size:train_size + val_size], targets[train_size:train_size + val_size]
	features_val = features_normalizer.transform(features_val)
	targets_val = targets_normalizer.transform(targets_val)
	val_dataset = DatasetClass(features_val, targets_val, feature_window_size, target_window_size, intersect)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	all_val_datasets = [
		DatasetClass(features_val[i].unsqueeze(0), targets_val[i].unsqueeze(0), feature_window_size,
					 target_window_size, intersect) for i in range(val_size)
	]
	all_val_dataloaders = [
		torch.utils.data.DataLoader(all_val_datasets[i], batch_size=1, shuffle=False) for i in
		range(len(all_val_datasets))
	]

	# test loaders and per-dataset class:
	features_test, targets_test = features[train_size + val_size:], targets[train_size + val_size:]
	test_size = len(features_test) if isinstance(features_test, list) else features_test.shape[0]
	features_test = features_normalizer.transform(features_test)
	targets_test = targets_normalizer.transform(targets_test)
	test_dataset = DatasetClass(features_test, targets_test, feature_window_size, target_window_size, intersect)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	all_test_datasets = [
		DatasetClass(features_test[i].unsqueeze(0), targets_test[i].unsqueeze(0),
					 feature_window_size, target_window_size, intersect) for i in range(test_size)
	]
	all_test_dataloaders = [
		torch.utils.data.DataLoader(all_test_datasets[i], batch_size=1, shuffle=False)
		for i in range(len(all_test_datasets))
	]
	return {
		'train': {
			'data': train_dataset,
			'loader': train_loader,
			'all_datasets': all_train_datasets,
			'all_dataloaders': all_train_dataloaders
		},
		'val': {
			'data': val_dataset,
			'loader': val_loader,
			'all_datasets': all_val_datasets,
			'all_dataloaders': all_val_dataloaders
		},
		'test': {
			'data': test_dataset,
			'loader': test_loader,
			'all_datasets': all_test_datasets,
			'all_dataloaders': all_test_dataloaders
		}

	}


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                  OTHER                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def get_gpu_usage_percentage() -> float:
	"""
	returns the GPU percentage usage, used inside a forward, as part ot tqdm logs
	:return: gpu percentage
	"""
	output = subprocess.check_output(
		['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits']
	).decode('utf-8').strip()
	gpu_memory_usage, gpu_memory_total = map(int, output.split(', '))
	gpu_memory_usage_percent = (gpu_memory_usage / gpu_memory_total) * 100
	return round(gpu_memory_usage_percent, 3)


def flatten_dict(d, parent_key='', sep='_'):
	"""
	Flatten a nested dictionary.

	Args:
		d (dict): Nested dictionary.
		parent_key (str): Prefix to use for keys.
		sep (str): Separator between parent and child keys.

	Returns:
		dict: Flattened dictionary.
	"""
	items = []
	for k, v in d.items():
		new_key = f"{parent_key}{sep}{k}" if parent_key else k
		if isinstance(v, dict):
			items.extend(flatten_dict(v, new_key, sep=sep).items())
		else:
			items.append((new_key, v))
	return dict(items)


def save_np_to_matlab_mat(
		data: np.ndarray,
		save_path: str,
		key_name: Optional[str] = None
) -> None:
	scipy.io.savemat(save_path, {key_name if key_name else 'data': data})


def mpl():
	""" boilerplate code for matplotlib importing and tkagg usage """
	import matplotlib
	import matplotlib.pyplot as plt
	matplotlib.use('TkAgg')


def update_json(yaml_path: str, new_data):
	""" updates a given yaml file with new data dictionary """
	if os.path.exists(yaml_path):
		with open(yaml_path, "r", encoding='utf-8') as f:
			old_data = json.load(f)
	else:
		old_data = {}
	with open(yaml_path, "w", encoding='utf-8') as f:
		json.dump({**old_data, **new_data}, f, ensure_ascii=False, indent=4)


def _format_df_torch_entries(
		df: pd.DataFrame
) -> pd.DataFrame:
	""" takes in a df where every entry is a torch tensor and returns a new df with unpacked tensor values as cols """
	old_cols = df.columns.tolist()
	new_df = pd.DataFrame()

	def create_new_columns(row, col_name):
		tensor = row[col_name]
		if tensor.dim() == 0:  # tensor is scalar
			new_col_names = [col_name]
			new_cols = [tensor.item()]
		else:
			new_col_names = [f"{col_name}_{i}" for i in range(len(tensor))]
			new_cols = tensor.numpy().tolist()
		return pd.Series(new_cols, index=new_col_names)

	for col_name in old_cols:
		if col_name.startswith('pred_') or col_name.startswith('true_'):
			new_cols = df.apply(lambda row: create_new_columns(row, col_name), axis=1)
			new_df = pd.concat([new_df, new_cols], axis=1)

	return new_df


def remove_and_trim_datasets(
		kinematics_path: str,
		forces_path: str,
		save: bool = True,
		basic_plot: bool = True,
		keep_as_list: bool = True
) -> None:
	"""
	a basic function that plots the data and allows to slice it and/or remove it altogether.
	:param kinematics_path: path to tensor or list of tensors kinematics
	:param forces_path: path to tensor or list of tensors forces
	:param save: if true, saves the results
	:param basic_plot: if true, plot a jpg, otherwise plots an interactive plot on html
	:param keep_as_list: if false, stacks as torch tensor, otherwise keeps values in list (for var-len dataset)
	"""
	forces_to_keep = []
	kinematics_to_keep = []
	forces = torch.load(forces_path)
	kinematics = torch.load(kinematics_path)
	for dataset_idx in tqdm(range(len(forces))):
		curr = torch.cat((forces[dataset_idx], kinematics[dataset_idx]), dim=1)
		df = pd.DataFrame(curr, columns='f1 f2 f3 f4 theta phi psi'.split())
		if basic_plot:
			df.plot()
			xticks_location = np.linspace(df.index.min(), df.index.max(), num=20)  # 20 ticks
			plt.xticks(xticks_location, rotation=45)
			plt.grid()
			plt.show()
		else:
			plot_df_with_plotly(df)
		if input('Is this one bad? [y/Any]') == 'y':
			continue
		curr_force = forces[dataset_idx]
		curr_kinematics = kinematics[dataset_idx]
		trim_suffix_index = input('add suffix index: ')
		if trim_suffix_index:
			curr_force = curr_force[:int(trim_suffix_index)]
			curr_kinematics = curr_kinematics[:int(trim_suffix_index)]
		forces_to_keep.append(curr_force)
		kinematics_to_keep.append(curr_kinematics)

	if not keep_as_list:
		forces_to_keep = torch.stack(forces_to_keep)
		kinematics_to_keep = torch.stack(kinematics_to_keep)
	if save:
		torch.save(forces_to_keep, forces_path.replace('.pt', '_cleaned.pt'))
		torch.save(kinematics_to_keep, kinematics_path.replace('.pt', '_cleaned.pt'))


def from_loss_logs_to_df(all_logs_combined: str, plot, split_by):
	"""
	WHICH = 'ours'
	df = from_loss_logs_to_df(ours_s if WHICH == 'ours' else prssm_s, True,
							  split_by='\n    \n    \n' if WHICH == 'ours' else '\n\n\n')
	:param all_logs_combined:
	:param plot:
	:param split_by:
	:return:
	"""
	loss_logs_list = all_logs_combined.split(split_by)
	dfs = []
	for loss_logs in loss_logs_list:
		lines = loss_logs.split('\n')
		names = lines[0].replace('data #', 'data_#')
		space_remover = lambda s: re.sub(r'\s+', ' ', s)
		names = space_remover(names).split(' ')[1:]
		vals = space_remover(lines[1][1:])[1:].split(' ')
		vals = [float(v) for v in vals]
		if vals[0] == 0.0: vals = vals[1:]
		df = pd.DataFrame([vals], columns=names)
		dfs.append(df)
	# concat all dfs:
	df = pd.concat(dfs, axis=0)
	df.index = ['Seq2Seq', 'Seq2Seq+ASL', 'Transformer', 'Linear', 'AutoFormer', 'FedFormer', 'NLinear']
	if plot:
		plt.figure(figsize=(10, 6))
		for i, model in enumerate(df.index):
			# Get the color of the scatter plot for the current model
			color = plt.gca()._get_lines.get_next_color()

			plt.scatter(range(len(df.columns)), df.loc[model],
						label=f'{model} (Mean MAE: {df.loc[model, "mean"]:.4f})', color=color)

			# Plot a horizontal line at the mean loss for each model with the same color as the scatter plot
			plt.axhline(y=df.loc[model, "mean"], color=color, linestyle='--')

		# Add the mean loss as text next to the horizontal line

		plt.title('HUJI dataset full results')
		plt.xlabel('Dataset #')
		plt.ylabel('Test MAE Loss')
		plt.legend()
		# plt.grid(linestyle='-.', linewidth=0.5)
		ax = plt.gca()
		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.yaxis.set_major_locator(MultipleLocator(0.1))
		ax.xaxis.set_minor_locator(AutoMinorLocator(1))
		ax.yaxis.set_minor_locator(AutoMinorLocator(1))
		ax.grid(which='major', color='#CCCCCC', linestyle='--')
		ax.grid(which='minor', color='#CCCCCC', linestyle=':')
		plt.show()
	return df


def attention_corr_with_2nd_deriv(trainer):
	"""
	plot the weights and the test batch in the same figure
	"""
	test_batch = next(iter(trainer.data_dict['test']['loader']))[0]  # (B,T,F)
	seq2seq = trainer.model.to('cpu')

	_ = seq2seq(test_batch)
	attn_weights = seq2seq.attention.attn_weights.detach().numpy().squeeze()  # (B,T)
	force_derivatives = [encoders.derivative(test_batch[:, :, i], dt=1 / 5000) for i in range(test_batch.shape[-1])]
	z=1


if __name__ == '__main__':
	ours_best = '/home/hadar/Thesis/ML/saved_models/Input(512, 512, 4)seq2seq[1,eemb10,ehid110,nl1,biFalse,demb10,dhid110,out3adlTrue,cat_adlFalse,complexFalse,gateTrue,multidimfftFalse,freq_thres210,drop0.1,perfreqTrue]_ADL[ours,T=1_2024-02-16_07-15-45'
	prssm_best = ''
	t = trainer.Trainer.from_model_dirname(ours_best)
	attention_corr_with_2nd_deriv(t)