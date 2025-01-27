from typing import Optional
import numpy as np
import scipy
import cv2
from loguru import logger

from Camera import camera_utils, camera_gui


def triangulate(
		proj_mat1_path: str,
		proj_mat2_path: str,
		points1: np.ndarray,
		points2: np.ndarray
):
	"""
	uses the projection matrix provided in matlab's stereoParams and triangulates the points in both arrays
	:param proj_mat1_path: path to first projection matrix
	:param proj_mat2_path: path to second projection matrix
	:param points1: shaped (n_trajs, m_pts_per_traj,2)
	:param points2: same
	:return: np array of cartesian coordinates, shaped as (n_trajs, m_pts_per_traj,3)
	"""
	logger.info("Performing Triangulation...")
	assert points1.shape[-1] == 2 and points2.shape[-1] == 2
	proj1 = scipy.io.loadmat(proj_mat1_path)['cameraMatrix1']
	proj2 = scipy.io.loadmat(proj_mat2_path)['cameraMatrix2']
	points_3d = []
	for traj_i in range(points1.shape[0]):
		points1_i = points1[traj_i].T  # (2,m_pts_per_traj)
		points2_i = points2[traj_i].T  # same
		# cv2.undistortPoints(points1_i,proj1,1,) TODO
		homog_points_3d = cv2.triangulatePoints(proj1, proj2, points1_i, points2_i)  # (4, m_pts_per_traj)
		cartesian_points_3d = homog_points_3d[:3, :] / homog_points_3d[-1, :]
		points_3d.append(cartesian_points_3d.T)
	points_3d = np.stack(points_3d)
	return points_3d


def evaluate_3d_distances(
		proj2_path: str,
		proj3_path: str,
		im2_path: str,
		im3_path: str,
		real_dist: float | None = None,
		crop_params2_path: str | None = None,
		crop_params3_path: str | None = None
) -> None:
	"""
	measures the distance between two points on an image

	:param proj2_path: projection matrix for the first image (from matlab)
	:param proj3_path: same for the second image (from matlab)
	:param im2_path: first image for triangulation
	:param im3_path: 2nd image for triangulation
	:param real_dist: the real distance between the points
	:param crop_params3_path: if provided, assumes that the image is cropped, so we shift it
	:param crop_params2_path: same, just for camera #2
	"""
	points2 = camera_gui.select_points(im2_path)
	points3 = camera_gui.select_points(im3_path)
	if crop_params2_path and crop_params3_path:
		points2 = camera_utils.shift_image_based_on_crop(points2, crop_params2_path)
		points3 = camera_utils.shift_image_based_on_crop(points3, crop_params3_path)
	points_3d = triangulate(proj2_path, proj3_path, points2.reshape((1, 2, 2)), points3.reshape((1, 2, 2)))[0]
	dist = np.sqrt(np.sum((points_3d[0] - points_3d[1]) ** 2))
	print(f"Calculated distance is {dist:.1f} [mm] ({dist / 10:.2f} [cm])")
	if real_dist is not None:
		error = np.abs(dist - real_dist)
		print(f"Error: {error}")


def xyz2euler(trajectories_3d: np.ndarray) -> np.ndarray:
	# """
	#   assumes that trajectories_3d contains 3 points that are ordered as such
	#     (tip) ------------------------------- (base)
	#         \ (0)<---------span----------(2)/
	# trailing \  \   |                      /
	#   edge    \  \  |chord                /
	#            \  \ |                    /
	#             \  (1)                  /
	#               ---------------------
	#     where (i) is the index in trajectories_3d. From what I've seen, this is always the case as the points are
	#     sorted using L2 distance from (0,0).
	# takes in a np array of trajectories in cartesian coordinates and converts it to array of wing angles phi,theta,psi
	# :param trajectories_3d: a np array (n_trajectories,m_points_per_trajectory,3) where 3 indicates [x,y,z]
	# :return: a np array (m_points_per_trajectory, 3) where 3 indicates [theta,phi,psi], and
	#          - theta : elevation angle
	#          - phi: stroke angle
	#          - psi: pitch angle
	# """
	xy_plane = (0, 0, 1, 0)  # Z = 0
	xz_plane = (0, 1, 0, 0)  # Y = 0 (stroke plane)

	p0, p1, p2 = trajectories_3d
	span = normalize_vector(p2 - p0)
	trailing_edge = normalize_vector(p1 - p0)
	chord = trailing_edge * np.sin(angle_between_vectors(span, trailing_edge))[:, np.newaxis]

	theta = angle_between_vector_and_plane(span, xz_plane)

	span_projected_to_stroke_plane = span * np.cos(theta)[:, np.newaxis]
	phi = angle_between_vector_and_plane(span_projected_to_stroke_plane, xy_plane)

	stroke_plane_normal = np.array(xz_plane[:3])
	surf_sp = normalize_vector(np.cross(span, stroke_plane_normal))  # parallel to stroke plane & span, follows φ
	yax = -1 * normalize_vector(np.cross(span, surf_sp))

	ypsi = np.sum(chord * yax, axis=1)
	xpsi = np.sum(chord * surf_sp, axis=1)
	psi = np.arctan2(ypsi, xpsi)

	return np.array([theta, phi, psi])


def normalize_vector(v: np.ndarray) -> np.ndarray:
	"""
	:param v: a matrix with shape (n,3) that represents the trajectory of a vector
	:return: a row-wise l2 normalized vector
	"""
	return v / np.linalg.norm(v, axis=1)[:, np.newaxis]


def angle_between_vectors(
		u: np.ndarray,
		v: np.ndarray
) -> np.ndarray:
	""" returns the angle between u and v for two vectors.
	 supports two cases for vector v - either a single vector (usually unit vector e_i) or a matrix (n,3)
	"""
	if len(v.shape) == 1:
		norm = np.arccos(np.dot(u, v) / (np.linalg.norm(v) * np.linalg.norm(u, axis=1)))
	else:
		norm = np.arccos(np.sum(u * v, axis=1) / (np.linalg.norm(v, axis=1) * np.linalg.norm(u, axis=1)))
	return norm


def angle_between_vector_and_plane(
		vec: np.ndarray,
		plane: tuple
) -> np.ndarray:
	"""
	returns the angle between a vector and a plane based on the angle between a vector and the normal
	:param vec: a np array of 1 or more points [x,y,z], shaped (N,3)
	:param plane: a tuple (a,b,c,d) that represents ax+by+cz=d
	"""
	plane_normal = np.array(plane[:3])
	angle_between_vec_and_normal = angle_between_vectors(vec, plane_normal)
	angle = np.pi / 2 - angle_between_vec_and_normal  # right triangle
	return angle


if __name__ == '__main__':
	evaluate_3d_distances(
		proj2_path=r"G:\My Drive\Master\Lab\Thesis\Camera\calibrations\22_09_2023\cameraMatrix1.mat",
		proj3_path=r"G:\My Drive\Master\Lab\Thesis\Camera\calibrations\22_09_2023\cameraMatrix2.mat",
		im2_path=r"E:\Hadar\experiments\22_08_2023\cam2\photos\Img000000.jpg",
		im3_path=r"E:\Hadar\experiments\22_08_2023\cam3\photos\Img000000.jpg"
	)
