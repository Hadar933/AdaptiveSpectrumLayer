% This script measures the 3D distance between two points 
% that were chosen with mouse user input

clear;
stereoParams = load('G:\My Drive\Master\Lab\Thesis\Camera\calibrations\22_08_2023\stereoParams.mat');
stereoParams = stereoParams.stereoParams;

I1 = imread("E:\Hadar\experiments\22_08_2023\cam2\photos\Img000000.jpg");
I2 = imread("E:\Hadar\experiments\22_08_2023\cam3\photos\Img000000.jpg");

%I1 = undistortImage(I1,stereoParams.CameraParameters1);
%I2 = undistortImage(I2,stereoParams.CameraParameters2);

figure;
imshow(I1);
title('Select two points of interest from I1');
% use mouse button to zoom in or out
% Press Enter to get out of the zoom mode
zoom on;
% Wait for the most recent key to become the return/enter key
waitfor(gcf, 'CurrentCharacter', char(13))
zoom reset
zoom off
p1 = ginput(2);
close;

figure;
imshow(I2);
title('Select the corresponding points from I2');
% use mouse button to zoom in or out
% Press Enter to get out of the zoom mode
zoom on;
% Wait for the most recent key to become the return/enter key
waitfor(gcf, 'CurrentCharacter', char(13))
zoom reset
zoom off
p2 = ginput(2);
close;

% add crop params
% p1 = p1 + [889,351];
% p2 = p2 + [27,310];


points_3d = triangulate(p1,p2,stereoParams);
euclid_dist = norm(points_3d(1, :)- points_3d(2, :));
fprintf('Distance in mm: %d\n', euclid_dist);


