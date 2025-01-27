from typing import Union, Optional
import matplotlib.pyplot as plt
from Camera import camera_gui
from matplotlib import animation
import os
import cv2
from tqdm import tqdm
import pickle
import matplotlib as mpl
import numpy as np
from pathlib import Path
import re
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                            vid <--> image conversion                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


def images2vid(image_folder, output_file, fps, start=None, end=None, crop_with_GUI=False):
	""" takes in a directory of images and converts them to a video with desired fps """
	if crop_with_GUI:
		y, x, h, w = camera_gui.get_rectangle_coordinates(os.path.join(image_folder, '000001.jpg'))
	images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
	if start and end:
		images = images[start:end]
	frame = cv2.imread(os.path.join(image_folder, images[0]))
	if crop_with_GUI:
		frame = crop(frame, y, x, h, w)
	height, width, _ = frame.shape
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

	for image in tqdm(images, desc='Converting to mp4', unit='frames'):
		image_path = os.path.join(image_folder, image)
		frame = cv2.imread(image_path)
		if crop_with_GUI:
			frame = crop(frame, y, x, h, w)
		video.write(frame)
	cv2.destroyAllWindows()
	video.release()


def video2images(video_path, output_dir, num_frames=-1):
	os.makedirs(output_dir, exist_ok=True)
	cap = cv2.VideoCapture(video_path)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	for frame_number in tqdm(range(frame_count), total=min(frame_count, num_frames)):
		ret, frame = cap.read()
		if not ret or frame_number == num_frames: break
		frame_path = os.path.join(output_dir, f'frame_{frame_number:04d}.png')
		cv2.imwrite(frame_path, frame)
	cap.release()


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                            Cropping                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

def crop(im: np.ndarray, y: int, x: int, h: int, w: int):
	""" crops the image as a rectangle """
	return im[y:y + h, x:x + w]


def crop_directory(camera_dirname, dirname_to_crop, cropped_dirname, y, x, h, w):
	"""
	@deprecated
	:param camera_dirname: the dirname of the current camera
	:param dirname_to_crop: the name of the directory that contains images to crop
	:param cropped_dirname:
	:return:
	"""
	new_cropped_path = os.path.join(camera_dirname, cropped_dirname)
	os.makedirs(new_cropped_path)
	all_images = os.listdir(dirname_to_crop)
	print(f"Cropping {dirname_to_crop} with y={y}, x={x}, h={h} and w={w} "
		  f"and saving to {new_cropped_path}...")
	for im_name in tqdm(all_images):
		if not im_name.endswith('jpg') and not im_name.endswith('png'):
			continue
		new_cropped_image_path = os.path.join(new_cropped_path, im_name)
		cropped_im = crop(cv2.imread(os.path.join(dirname_to_crop, im_name)), y, x, h, w)
		cv2.imwrite(new_cropped_image_path, cropped_im)


def get_crop_params(
		images_path: str,
		crop_params_filename: str,
		first_image_name: str,
):
	"""
	Loads the global cropping obtained from the PCC camera app and adds another user
	cropping if desired.
	when using the .crop function, and only app cropping, the dimensions of the image will stay
	the same. If user cropping is used on top, the image will be cropped to these dimensions
	@:param app_cropping_path: camera image configurations .adj file
	:param images_path:
	:param crop_params_filename: a string for
	:param first_image_name:
	:return:
	"""
	parent_dir = Path(images_path).parent
	if crop_params_filename in os.listdir(parent_dir):
		with open(os.path.join(parent_dir, crop_params_filename), 'rb') as f:
			y, x, h, w = pickle.load(f)
	else:
		y, x, h, w = camera_gui.get_rectangle_coordinates(os.path.join(images_path, first_image_name))
		with open(os.path.join(parent_dir, crop_params_filename), 'wb') as f:
			pickle.dump([y, x, h, w], f)
	return y, x, h, w


def _extract_camera_crop_params(adj_file):
	with open(adj_file, 'rb') as f:
		content = f.read()
	y = int(re.search(rb'<CropRectangleY>(\d+)</CropRectangleY>', content).group(1))
	x = int(re.search(rb'<CropRectangleX>(\d+)</CropRectangleX>', content).group(1))
	h = int(re.search(rb'<CropRectangleHeight>(\d+)</CropRectangleHeight>', content).group(1))
	w = int(re.search(rb'<CropRectangleWidth>(\d+)</CropRectangleWidth>', content).group(1))
	return y, x, h, w


def shift_image_based_on_crop(
		points_2d: np.ndarray,
		crop_params_path: str,
		cam_cropping_file: str,
		add_manual_crop: bool
):
	"""
	when cropping images, the origin is shifted. we undo this process using the crop params
	:param points_2d: an array of points, that can be either (n_traj, m_points_per_traj, 2) or (m_points,2)
	:param crop_params_path: path to the crop parameters pickle
	:param cam_cropping_file: the adj file from the PCC camera app
	:param add_manual_crop: if true, applies the user crop fix as well
	:return: points_2d, shifted
	"""
	y_cam, x_cam, w_cam, h_cam = _extract_camera_crop_params(cam_cropping_file)
	points_2d += np.array([x_cam, y_cam])
	if add_manual_crop:
		with open(crop_params_path, 'rb') as f:
			y_user, x_user, h_user, w_user = pickle.load(f)
		points_2d += np.array([x_user, y_user])
	return points_2d


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                            Plotting                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


def image_plotter(
		path_or_img: Union[str, np.ndarray],
		thres: float = 0.0,
		adaptive_thres_block_size: int = 0,
		adaptive_thres_param_c: int = 0
):
	""" plots a given image or image path with possible thresholding """
	if isinstance(path_or_img, str):
		image = cv2.imread(path_or_img, cv2.IMREAD_GRAYSCALE)
	elif isinstance(path_or_img, np.ndarray):
		image = path_or_img
	else:
		raise ValueError('Unsupported path or image')
	if thres > 0:
		image = cv2.threshold(image, thres, np.max(image), cv2.THRESH_BINARY)[1]
	if adaptive_thres_block_size > 0 and adaptive_thres_param_c > 0:
		image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
									  adaptive_thres_block_size, adaptive_thres_param_c)
	plt.imshow(image, cmap='gray')
	plt.show()
	return image


def plot_trajectories(
		trajectory_3d: np.ndarray,
		wing_plane_jmp: int = -1,
		leading_edge_vec_jump: int = -1,
		add_to_title: str = '',
		center_of_mass_point: Optional[np.ndarray] = None
):
	"""
	plots one or more trajectories in 3d, where z: depth, y: height, x: width
	:param trajectory_3d: a np array (n_trajectories,m_points_per_trajectory, 3)
	:param wing_plane_jmp: if > 0, plots a triangle that represents the wing plane every wing_plane_jmp frames
	:param leading_edge_vec_jump if > 0 plots the leading edge vector every leading_edge_vec_jump frames.
	:param add_to_title: a string we can concat to the tile of the plot
	:param center_of_mass_point: we can also plot another 4th point on our 3D plotter, which is usually the COM point
	"""
	mpl.use('TkAgg')
	fig = plt.figure(figsize=(12, 12))
	ax = fig.add_subplot(projection='3d')
	colormap = mpl.colormaps['binary']
	n_pts, m_pts_per_traj, _ = trajectory_3d.shape
	ax.plot3D(trajectory_3d[0, :, 0], trajectory_3d[0, :, 1], trajectory_3d[0, :, 2], linewidth=2, label='tip')
	ax.plot3D(trajectory_3d[1, :, 0], trajectory_3d[1, :, 1], trajectory_3d[1, :, 2], linewidth=2, label='bottom')
	ax.plot3D(trajectory_3d[2, :, 0], trajectory_3d[2, :, 1], trajectory_3d[2, :, 2], linewidth=2, label='base')

	if center_of_mass_point is not None:
		ax.plot3D(center_of_mass_point[:, 0], center_of_mass_point[:, 1], center_of_mass_point[:, 2], linewidth=2,
				  label='C.O.M')

	if wing_plane_jmp > 0:
		triangle_points_order = [0, 1, 2, 0]

		for t in range(0, m_pts_per_traj, wing_plane_jmp):
			x = trajectory_3d[:, t, 0][triangle_points_order]
			y = trajectory_3d[:, t, 1][triangle_points_order]
			z = trajectory_3d[:, t, 2][triangle_points_order]

			# Calculate color based on the time increment
			color = colormap(t / m_pts_per_traj)  # You can choose a different colormap

			# Create a Poly3DCollection with gradient color
			ax.add_collection3d(Poly3DCollection([list(zip(x, y, z))], color=color, edgecolor='k', linewidths=1))

	if leading_edge_vec_jump > 0:
		p0, p1, p2 = trajectory_3d
		leading_edge = p2 - p0

		for t in range(0, m_pts_per_traj, leading_edge_vec_jump):
			ax.quiver(p0[t, 0], p0[t, 1], p0[t, 2], leading_edge[t, 0], leading_edge[t, 1], leading_edge[t, 2])

	ax.view_init(elev=-80, azim=-90)
	ax.set_xlabel('x [mm]'), ax.set_ylabel('y [mm]'), ax.set_zlabel('z [mm]')
	plt.legend()
	plt.title("Points on the wing " + add_to_title)
	plt.axis('equal')
	plt.show()


def trajectory_video(trajectory_3d: np.ndarray,interval):
	matplotlib.use('WebAgg')
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	def update(num, data, lines):
		for i, line in enumerate(lines):
			line.set_data(data[i][:2, :num])
			line.set_3d_properties(data[i][2, :num])

	# N = trajectory_3d.shape[1]
	N = 100
	num_trajectories = trajectory_3d.shape[0]
	data = [trajectory_3d[i, :, :].T for i in range(num_trajectories)]
	lines = [ax.plot(data[i][0, 0:1], data[i][1, 0:1], data[i][2, 0:1])[0] for i in range(num_trajectories)]

	x_max = max(data[i].max(axis=1)[0] for i in range(num_trajectories))
	y_max = max(data[i].max(axis=1)[1] for i in range(num_trajectories))
	z_max = max(data[i].max(axis=1)[2] for i in range(num_trajectories))
	x_min = min(data[i].min(axis=1)[0] for i in range(num_trajectories))
	y_min = min(data[i].min(axis=1)[1] for i in range(num_trajectories))
	z_min = min(data[i].min(axis=1)[2] for i in range(num_trajectories))

	# Setting the axes properties
	ax.set_xlim3d([x_min, x_max])
	ax.set_xlabel('x [mm]')
	ax.set_ylim3d([y_min, y_max])
	ax.set_ylabel('y [mm]')
	ax.set_zlim3d([z_min, z_max])
	ax.set_zlabel('z [mm]')

	ax.view_init(elev=-80, azim=-90)  # Set the view angle
	plt.axis('equal')  # Ensure equal aspect ratio

	ani = animation.FuncAnimation(fig, update, N, fargs=(data, lines), interval=interval, blit=False)
	ani.save('traj.gif')
	# plt.show()


def plot_angles(
		angles: np.ndarray,
		n_samples: Optional[int] = None,
		convert2deg: bool = True,
		add_to_title: str = ''
):
	"""
	plots the wing angles in one plot
	:param n_samples: the number of samples. if provided, shows plot with time [sec] x-axis.
	:param convert2deg: if true, converts from radians to degrees
	:param add_to_title: a string we can concat to the tile of the plot
	:param angles: (m_points,3) for [theta,phi,psi]
	"""
	if convert2deg:
		angles = np.degrees(angles)
	num_samples = angles.shape[-1]
	for i, label in enumerate([r'$\theta$ elevation', r'$\phi$ stroke', r'$\psi$ pitch']):
		if n_samples:
			plt.plot(np.arange(0, num_samples) / n_samples, angles[i], label=label)
		else:
			plt.plot(angles[i], label=label)
	plt.title('Wing Angles ' + add_to_title)
	plt.xlabel('time [sec]' if n_samples else 'frame [#]'), plt.ylabel(f"angle [{'deg' if convert2deg else 'rad'}]")
	plt.legend(), plt.grid(), plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                            Other                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

def copy_keypoint(original_keypoint: cv2.KeyPoint):
	""" copy the content of one cv2 keypoint object to another"""
	copied_keypoint = cv2.KeyPoint()
	copied_keypoint.pt = original_keypoint.pt
	copied_keypoint.size = original_keypoint.size
	copied_keypoint.angle = original_keypoint.angle
	copied_keypoint.response = original_keypoint.response
	copied_keypoint.octave = original_keypoint.octave
	copied_keypoint.class_id = original_keypoint.class_id
	return copied_keypoint


if __name__ == '__main__':
	exp_date = '23_10_2023'
	exp_name = '[F=8.078_A=M_PIdiv5.062_K=0.8]'
	results_dir = fr"E:\Hadar\experiments\{exp_date}\results\{exp_name}"
	raw_cam2dir = fr"E:\Hadar\experiments\{exp_date}\cam2\Photos{exp_name}"
	raw_cam3dir = fr"E:\Hadar\experiments\{exp_date}\cam3\Photos{exp_name}"
	start, end, fps = 3000, 5200, 120
	trajectory_3d = np.load(fr"{results_dir}\trajectories.npy")[:, start:end, :]
	trajectory_video(trajectory_3d,fps)
	plot_angles(
		angles=np.load(f"{results_dir}\\angles.npy")[:, start:end],
	)
	plot_trajectories(
		trajectory_3d=trajectory_3d, wing_plane_jmp=100
		# center_of_mass_point=np.vstack(
		# 	pd.read_pickle(fr"{base_path}\merged_data_preprocessed_and_encoded.pkl")['center_of_mass'])
	)
	if input('continue? [y/n]') == 'y':
		images2vid(raw_cam2dir, r'G:\My Drive\Master\Lab\figures_for_thesis\cam2vid.mp4', fps, start, end)
		images2vid(raw_cam3dir, r'G:\My Drive\Master\Lab\figures_for_thesis\cam3vid.mp4', fps, start, end)
