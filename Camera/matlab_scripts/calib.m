clear;
exp_date = '02_10_2023';
use_hard_drive = true;
if use_hard_drive
    folder1 = sprintf('E:\\Hadar\\calibrations\\%s\\cam2cpy', exp_date);
    folder2 = sprintf('E:\\Hadar\\calibrations\\%s\\cam3cpy', exp_date);
else
    folder1 = sprintf('G:\\My Drive\\Master\\Lab\\Thesis\\Camera\\experiments\\%s\\cam2', exp_date);
    folder2 = sprintf('G:\\My Drive\\Master\\Lab\\Thesis\\Camera\\experiments\\%s\\cam3', exp_date);
end

squareSize = 10.0; % use 16.8 for large checkerbox
units = 'millimeters';
stereoCameraCalibrator(folder1,folder2,squareSize,units);

% calculating the projection matrices
%{
cameraMatrix1 = vision.internal.constructCameraMatrix( ...
    eye(3), ...
    [0 0 0], ...
    stereoParams.CameraParameters1.K ...
);
cameraMatrix2 = vision.internal.constructCameraMatrix( ...
    stereoParams.RotationOfCamera2', ...
    stereoParams.TranslationOfCamera2, ...
    stereoParams.CameraParameters2.K ...
);


calibratios_dir = sprintf('G:\\My Drive\\Master\\Lab\\Thesis\\Camera\\calibrations\\%s\\', exp_date);

stereoParams = struct(stereoParams); % for easier loading to python
save(strcat(calibratios_dir,'stereoParams.mat'),'stereoParams')
save(strcat(calibratios_dir,'cameraMatrix1.mat'),'cameraMatrix1');
save(strcat(calibratios_dir,'cameraMatrix2.mat'),'cameraMatrix2');  
%}