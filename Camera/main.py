import os.path
import numpy as np
from loguru import logger

from Camera import to_3d, camera_utils, tracker


def get_angles(
        exp_date: str,
        parent_dirname,
        crop_params_filename,
        photos_sub_dirname: str,
        first_image_name,
        tracking_params,
        add_manual_crop,
        show_wing_tracker,
        show_angle_results
) -> tuple[np.ndarray, np.ndarray]:
    """

    :param exp_date: experiment execution time dd-mm-yy, used as parent dirname
    :param photos_sub_dirname: a sub-dirname for every experiment
    :param use_hard_drive: if true, uses data from external hard drive
    :param show_wing_tracker: for optical flow visualizations
    :param show_angle_results: mostly for debugging
    :return: The tuple of two np arrays (angles, trajectory_3d)
    """
    proj1_path = f"Camera\\calibrations\\{exp_date}\\cameraMatrix1.mat"
    proj2_path = f"Camera\\calibrations\\{exp_date}\\cameraMatrix2.mat"
    camera_numbers = [2, 3]

    optic_flow = tracker.OpticalFlow('blobs', show_wing_tracker)
    trajectories_2d_dict = {cam_num: None for cam_num in camera_numbers}
    save_base_path = f"{parent_dirname}\\experiments\\{exp_date}\\results\\{photos_sub_dirname.replace('Photos', '')}"
    angles_result_path = f"{save_base_path}\\angles.npy"
    trajectory_3d_result_path = f"{save_base_path}\\trajectories.npy"

    if os.path.exists(angles_result_path) and os.path.exists(trajectory_3d_result_path):
        # fast load in case previously computed
        logger.info(f'Loading {angles_result_path} and {trajectory_3d_result_path} from memory...')
        trajectory_3d = np.load(trajectory_3d_result_path)
        angles = np.load(angles_result_path)
        return angles, trajectory_3d

    for cam_num in camera_numbers:
        curr_path = f"{parent_dirname if parent_dirname else 'Camera'}\\experiments\\{exp_date}"
        camera_dirname = f"{curr_path}\\cam{cam_num}"
        crop_params_path = f"{camera_dirname}\\{crop_params_filename}"
        cam_cropping_file = f"{curr_path}\\preset_2030{cam_num}.adj"

        trajectory_2d = optic_flow.run(
            camera_dirname=camera_dirname,
            tracking_params=tracking_params,
            crop_params_filename=crop_params_filename,
            first_image_name=first_image_name,
            photos_sub_dirname=photos_sub_dirname,
            add_manual_crop=add_manual_crop
        )
        trajectory_2d = camera_utils.shift_image_based_on_crop(
            points_2d=trajectory_2d,
            crop_params_path=crop_params_path,
            cam_cropping_file=cam_cropping_file,
            add_manual_crop=add_manual_crop
        )
        # randomly add either +1 or -1 to every entry, to mimic pixel noise:


        trajectories_2d_dict[cam_num] = trajectory_2d

    # making sure the trajectory lengths match (as we return in case of not identifying all points on the wing):
    min_length = min([trajectory.shape[1] for trajectory in trajectories_2d_dict.values()])
    trajectories_2d_dict = {key: val[:, :min_length, :] for key, val in trajectories_2d_dict.items()}

    trajectory_3d = to_3d.triangulate(
        proj_mat1_path=proj1_path,
        proj_mat2_path=proj2_path,
        points1=trajectories_2d_dict[camera_numbers[0]],
        points2=trajectories_2d_dict[camera_numbers[1]]
    )
    angles = to_3d.xyz2euler(
        trajectories_3d=trajectory_3d
    )

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)
    np.save(angles_result_path, angles)
    np.save(trajectory_3d_result_path, trajectory_3d)

    if show_angle_results:
        camera_utils.plot_trajectories(trajectory_3d, wing_plane_jmp=100)
        camera_utils.plot_angles(angles, n_samples=10_000)

    return angles, trajectory_3d
