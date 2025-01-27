import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from numpy.linalg import norm
import pandas as pd
import Camera.tracker


def smooth(y, kernel_size: int, how):
	y_smooth = getattr(pd.Series(y).rolling(window=kernel_size, center=True), how)()
	y_smooth = y_smooth.fillna(method='bfill').fillna(method='ffill').values
	return y_smooth


def find_start_based_on_pairwise_df_rows_dist(df, threshold, kernel_size=0, smooth_type=''):
	if kernel_size > 0 and smooth_type:
		df = df.apply(lambda col: smooth(col, kernel_size, smooth_type))
	for t in range(len(df) - 1):
		diff = (df.iloc[t + 1] - df.iloc[t]).abs()
		if (diff > threshold).any():
			return t
	raise ValueError('Did not find camera start...')


def find_signal_start_based_on_value_threshold(threshold, df, kernel_size=0, smooth_type=''):
	if kernel_size > 0 and smooth_type:
		df = df.apply(lambda col: smooth(col, kernel_size, smooth_type))
	starts = []
	for col_values in df.values.T:
		start_candidate = np.argmax(col_values > threshold)  # for bool array, argmax returns first True(=1) value
		if start_candidate != 0:
			starts.append(start_candidate)
	start = np.min(starts) if len(starts) > 0 else None
	return start


def find_camera_start_based_on_trajectory(threshold, trajectory_3d):
	for t in range(trajectory_3d.shape[1] - 1):
		curr = trajectory_3d[:, t, :]
		next = trajectory_3d[:, t + 1, :]
		dist = norm(next - curr, axis=1)
		if (dist > threshold).any():
			return t
	raise ValueError('Did not find camera start...')


def find_camera_start_based_on_images(images_path, tracking_params, threshold=150.0, verbose=True):
	images = os.listdir(images_path)
	images = sorted([os.path.join(images_path, f) for f in images if f.endswith(".jpg")])

	prev_image = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
	blob_detector = Camera.tracker.NaiveBlobFinder(tracking_params['NumBlobs'], **tracking_params)
	blobs = blob_detector.run(prev_image, verbose=verbose)
	p0 = np.array([[*b.pt, b.size / 2] for b in blobs])
	plot_every_n = 500

	def get_region_avg_diff(img1, img2, x, y, radius):
		region1 = img1[int(y - radius):int(y + radius) + 1, int(x - radius):int(x + radius) + 1]
		region2 = img2[int(y - radius):int(y + radius) + 1, int(x - radius):int(x + radius) + 1]
		return np.abs(region1 - region2).mean()

	def plot_points(p0):
		plt.figure(figsize=(12, 12))  # Adjusted to make the plots a bit smaller

		for idx, (x, y, radius) in enumerate(p0, start=1):
			# Extract region for the current point from both images
			region1 = prev_image[int(y - radius):int(y + radius) + 1, int(x - radius):int(x + radius) + 1]
			region2 = curr_image[int(y - radius):int(y + radius) + 1, int(x - radius):int(x + radius) + 1]
			diff = np.abs(region1 - region2)

			# Plot region from the first image
			plt.subplot(3, 3, idx)
			plt.imshow(region1, cmap='gray')
			plt.title(f"Point {idx} - Region1", fontsize=14)  # Increased fontsize

			# Plot region from the second image
			plt.subplot(3, 3, idx + 3)
			plt.imshow(region2, cmap='gray')
			plt.title(f"Point {idx} - Region2", fontsize=14)  # Increased fontsize

			# Plot absolute difference
			plt.subplot(3, 3, idx + 6)
			plt.imshow(diff, cmap='gray')
			plt.title(f"Point {idx} - Abs Diff", fontsize=14)  # Increased fontsize

		plt.tight_layout()
		plt.show()

	for idx, image_path in enumerate(images[1:], start=1):
		curr_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

		# Calculate region average differences for each point
		motion_magnitudes = [get_region_avg_diff(prev_image, curr_image, x, y, r) for x, y, r in p0]
		if idx % plot_every_n == 0:
			plot_points(p0)
		# Check if any region's average difference exceeds the threshold
		if any(magnitude > threshold for magnitude in motion_magnitudes):
			return idx  # This image index is where motion starts
		prev_image = curr_image
	return None  # No motion detected


def plot_start(camera_start, forces_start, camera_df, forces_df) -> None:
	ax = camera_df.plot()
	plt.axvline(camera_start, color='red', linestyle='--')
	plt.text(x=camera_start, y=0.9 * camera_df.max().max(), color='red', s='Camera\n start')

	forces_df.plot(ax=ax)
	plt.axvline(forces_start, color='blue', linestyle='--')
	plt.text(x=forces_start, y=0.9 * forces_df.max().max(), color='blue', s='Forces\n start')

	plt.show()


def trim_and_join(force_start, camera_start, force_sample_freq, camera_sample_freq, force_df, angles_df,
				  trajectory_3d_df):
	angles_df = angles_df[camera_start:]
	angles_df.index = pd.timedelta_range(start='0', periods=len(angles_df), freq=camera_sample_freq)

	trajectory_3d_df = trajectory_3d_df[camera_start:]
	trajectory_3d_df.index = pd.timedelta_range(start='0', periods=len(trajectory_3d_df), freq=camera_sample_freq)

	force_df = force_df[force_start:]
	force_df.index = pd.timedelta_range(start='0', periods=len(force_df), freq=force_sample_freq)
	return angles_df, force_df, trajectory_3d_df


from scipy.signal import hilbert
from scipy.signal import find_peaks


def find_start_indices(y):
	# Compute the envelope of the signal
	analytic_signal = hilbert(y)
	envelope = np.abs(analytic_signal)

	# Threshold the envelope
	envelope[envelope >= -0.4] = -0.4
	envelope[envelope <= -0.5] = -0.5

	# Find the start of the signal
	peaks, _ = find_peaks(envelope, height=-0.5, distance=350)

	# Create a DataFrame to store the start and stop indices
	start_indices = peaks
	stop_indices = [peaks[i + 1] for i in range(len(peaks) - 1)] + [len(y) - 1]

	StartStop = pd.DataFrame({'Start_Indices': start_indices, 'Stop_Indices': stop_indices})

	return StartStop


def merge_data(
		exp_date: str,
		parent_dirname: str,
		photos_sub_dirname: str,
		trajectory_3d: np.ndarray,
		angles_df: pd.DataFrame,
		forces_df: pd.DataFrame,
		camera_freq: str,
		force_freq: str,
		camera_threshold: float,
		force_threshold: float,
		smooth_kernel_size: int,
		smooth_method: str,
		show_start_indicators: bool
) -> pd.DataFrame:
	main_path = f"{parent_dirname}\\experiments\\{exp_date}\\results\\{photos_sub_dirname.split('Photos')[1]}"
	if os.path.exists(f"{main_path}\\merged_data.pkl"):
		logger.info(f"Loading {main_path}\\merged_data.pkl from memory...")
		return pd.read_pickle(f"{main_path}\\merged_data.pkl")

	trajectory_3d_df = pd.DataFrame({
		'p0': [xyz for xyz in trajectory_3d[0]],
		'p1': [xyz for xyz in trajectory_3d[1]],
		'p2': [xyz for xyz in trajectory_3d[2]],
	})
	forces_start = find_start_based_on_pairwise_df_rows_dist(
		df=forces_df,
		threshold=force_threshold,
		kernel_size=smooth_kernel_size,
		smooth_type=smooth_method
	)
	camera_start = find_camera_start_based_on_trajectory(
		threshold=camera_threshold,
		trajectory_3d=trajectory_3d
	)

	if show_start_indicators:
		plot_start(camera_start, forces_start, angles_df, forces_df)

	angles_df, forces_df, trajectory_3d_df = trim_and_join(
		force_start=forces_start,
		camera_start=camera_start,
		force_sample_freq=force_freq,
		camera_sample_freq=camera_freq,
		force_df=forces_df,
		angles_df=angles_df,
		trajectory_3d_df=trajectory_3d_df
	)
	all_merged = angles_df.join(forces_df).join(trajectory_3d_df)
	all_merged.to_pickle(f"{main_path}\\merged_data.pkl")
	pd.DataFrame({
		'camera_start': [camera_start],
		'force_start': [forces_start]}
	).to_pickle(f"{main_path}\\starts.pkl")

	return all_merged
