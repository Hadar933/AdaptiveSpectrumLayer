% Load stereoParams and corresponding points from input_path
traj1 = load('G:\My Drive\Master\Lab\Thesis\Camera\experiments\01_08_2023\cam2\traj.mat').data;
traj2 = load('G:\My Drive\Master\Lab\Thesis\Camera\experiments\01_08_2023\cam3\traj.mat').data;
sp = load('G:\My Drive\Master\Lab\Thesis\Camera\calibrations\01_08_2023\stereoParams.mat').stereoParams;

% Initialize an array to store triangulated 3D points
n_trajectories = size(traj1, 1);
triangulated_points = cell(n_trajectories, 1);

for i = 1:n_trajectories
    % Get points for the current trajectory
    curr_points1 = squeeze(traj1(i, :, :));
    curr_points2 = squeeze(traj2(i, :, :));
    
    % Undistort the points using stereoParams
    curr_points1_undistorted = undistortPoints(curr_points1, sp.CameraParameters1);
    curr_points2_undistorted = undistortPoints(curr_points2, sp.CameraParameters2);
    
    % Perform triangulation using undistorted points and stereoParams
    curr_points_3d = triangulate(curr_points1_undistorted, curr_points2_undistorted, sp);
    
    % Store the triangulated points for the current trajectory
    triangulated_points{i} = curr_points_3d;
end

% Save the triangulated points
save('G:\My Drive\Master\Lab\Thesis\Camera\experiments\01_08_2023\output_points.mat', 'triangulated_points');
