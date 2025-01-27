import cv2
from typing import List, Dict, Union, Tuple, Sequence, Literal
import os
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from Camera import camera_gui, camera_utils


class NaiveBlobFinder:
    def __init__(
            self,
            desired_num_blobs: int,
            **kwargs: Dict[str, float]
    ) -> None:
        self.desired_num_blobs = desired_num_blobs
        self._params: Union[cv2.SimpleBlobDetector.Params, None] = None
        self._detector: Union[cv2.SimpleBlobDetector, None] = None
        if kwargs is not None:
            self.set_tracker(override=True, **kwargs)

    @staticmethod
    def plot(
            frame: Union[str, np.ndarray],
            blobs: Sequence[cv2.KeyPoint],
            use_id: bool = False,
            title: str = None,
    ) -> None:
        """ plots blobs on an image, with possible class id text next to it."""
        im_with_blobs = cv2.drawKeypoints(frame, blobs, np.array([]), (255, 0, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if use_id:
            for blob in blobs:
                x, y = int(blob.pt[0]) + 15, int(blob.pt[1])
                cv2.putText(im_with_blobs, str(blob.class_id), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))
        plt.title(title)
        plt.imshow(im_with_blobs)
        plt.show()

    def set_tracker(
            self,
            override: bool,
            **kwargs: Dict[str, float]
    ) -> Tuple[cv2.SimpleBlobDetector.Params, cv2.SimpleBlobDetector]:
        """
        initializes the detector with params provided in kwargs
        :param override: If True, overrides self._params and self._detector
        :param kwargs: dictionary with possible blob detector params
        :return: the params object alongside the blob detector
        """
        params = cv2.SimpleBlobDetector_Params()
        if 'minThreshold' in kwargs: params.minThreshold = kwargs['minThreshold']
        if 'maxThreshold' in kwargs: params.maxThreshold = kwargs['maxThreshold']

        if 'minArea' in kwargs or 'maxArea' in kwargs: params.filterByArea = True
        if 'minArea' in kwargs: params.minArea = kwargs['minArea']
        if 'maxArea' in kwargs: params.maxArea = kwargs['maxArea']

        if 'minCircularity' in kwargs or 'maxCircularity' in kwargs: params.filterByCircularity = True
        if 'minCircularity' in kwargs: params.minCircularity = kwargs['minCircularity']
        if 'maxCircularity' in kwargs: params.maxCircularity = kwargs['maxCircularity']

        if 'minConvexity' in kwargs or 'maxConvexity' in kwargs: params.filterByConvexity = True
        if 'minConvexity' in kwargs: params.minConvexity = kwargs['minConvexity']
        if 'maxConvexity' in kwargs: params.maxConvexity = kwargs['maxConvexity']

        if 'minInertiaRatio' in kwargs or 'maxInertiaRatio' in kwargs: params.filterByInertia = True
        if 'minInertiaRatio' in kwargs: params.minInertiaRatio = kwargs['minInertiaRatio']
        if 'maxInertiaRatio' in kwargs: params.maxInertiaRatio = kwargs['maxInertiaRatio']

        detector = cv2.SimpleBlobDetector_create(params)
        if override:
            self._params = params
            self._detector = detector
        return params, detector

    def run(
            self,
            frame: np.ndarray,
            verbose: bool = False
    ) -> List[cv2.KeyPoint]:
        """
        find blobs in a given image
        :param frame: a (possibly cropped) frame to find blobs in
        :param verbose: if true, plots the found blobs
        :return a list where every item is a sequence blobs in each frame

        """
        # blob detector works on black blobs, so we invert the image:
        orig_image = frame.copy()
        frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)[1]
        found_blobs = self._detector.detect(frame)
        found_blobs = sorted(found_blobs, key=lambda blob: np.sqrt(np.sum((np.array([0, 0]) - np.array(blob.pt)) ** 2)))
        assert len(found_blobs) == self.desired_num_blobs, \
            f"Found {len(found_blobs)}, expected {self.desired_num_blobs}"
        for j, blob in enumerate(found_blobs):  # updating ids and setting first trajectory points for each blob
            blob.class_id = j
        if verbose:
            self.plot(orig_image, found_blobs, True, "Found Blobs")
        return found_blobs


class OpticalFlow:
    def __init__(
            self,
            point_of_interest_algorithm: Literal['gui', 'blobs', 'features'],
            verbose: bool
    ):
        """
        :param point_of_interest_algorithm: few choices for the first points tracking:
            - gui: the user picks the points using mouse
            - blobs: uses open-cvs blob tracking algorithm
            - features: uses open-cvs good-features algorithm
        :param verbose: for plotting purposes.
        """
        self.poi_alg = point_of_interest_algorithm
        self.verbose = verbose

    def run(
            self,
            camera_dirname: str,
            tracking_params: dict,
            crop_params_filename: str,
            first_image_name: str,
            photos_sub_dirname: str,
            add_manual_crop: bool,

    ) -> np.ndarray:
        """
        tracks points of interest using optical flow
        :param camera_dirname: the directory name of the current camera
        :param crop_params_filename: the filename of the crop parameters pickle
        :param first_image_name: the filename of the first image in the directory of images for the current camera
        :param photos_sub_dirname: the name of the directory that contains the images for the current camera
        :return: a np array that represents all trajectories, with shape [n_trajs,n_points_per_traj, 2 (x,y)]
        """

        images_path = os.path.join(camera_dirname, photos_sub_dirname)
        assert os.path.exists(images_path), f'Path {images_path} does not exist.'
        logger.info(f'Calculating Trajectory using optical flow on {images_path}...')
        frame = cv2.imread(os.path.join(images_path, first_image_name))

        if add_manual_crop:
            y, x, h, w = camera_utils.get_crop_params(images_path, crop_params_filename, first_image_name)
            frame = camera_utils.crop(frame, y, x, h, w)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        prev_pts = self._calc_first_trajectory_points(gray_frame, tracking_params)

        trajectories = prev_pts.copy()  # [n_trajs,n_points_per_traj, 2 (x,y)]
        mask = np.zeros_like(frame)  # Create a mask image for drawing purposes
        old_gray_frame = gray_frame.copy()
        color = np.random.randint(0, 255, (trajectories.shape[0], 3))  # random color for each trajectory
        all_images = sorted([im for im in os.listdir(images_path) if any([im.endswith('jpg'), im.endswith('png')])])

        if self.verbose:
            video_output = os.path.join(images_path, 'optical_flow_results.avi')
            frame_size = (frame.shape[1], frame.shape[0])  # Width and height of frames
            fps = 30
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Specify the codec
            out = cv2.VideoWriter(video_output, fourcc, fps, frame_size)

        for image_name in all_images[1:]:
            frame = cv2.imread(os.path.join(images_path, image_name))
            if add_manual_crop:
                frame = camera_utils.crop(frame, y, x, h, w)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray_frame is None:
                break
            next_pts, status, error = cv2.calcOpticalFlowPyrLK(old_gray_frame, gray_frame, prev_pts, None,
                                                               winSize=tracking_params['winSize'],
                                                               maxLevel=tracking_params['maxLevel'],
                                                               criteria=tracking_params['criteria'])
            if next_pts is not None:  # select good points
                good_next_pts = next_pts[status == 1]
                good_prev_pts = prev_pts[status == 1]
                if good_next_pts.shape[0] != trajectories.shape[0]:  # we did not find all points on the wing
                    logger.warning(f'Did not find all {trajectories.shape[0]} points. Finishing Process')
                    return trajectories
                else:
                    trajectories = np.append(trajectories, good_next_pts[:, np.newaxis, :], axis=1)

            if self.verbose:
                for i, (new, old) in enumerate(zip(good_next_pts, good_prev_pts)):
                    x_new, y_new = new.ravel()
                    x_old, y_old = old.ravel()
                    mask = cv2.line(mask, (int(x_new), int(y_new)), (int(x_old), int(y_old)), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (int(x_new), int(y_new)), 5, color[i].tolist(), -1)
                img = cv2.add(frame, mask)
                cv2.imshow('frame', img)
                k = cv2.waitKey(30) & 0xff
                if k == 27: break
                out.write(img)

            old_gray_frame = gray_frame.copy()
            prev_pts = good_next_pts.reshape(-1, 1, 2)

        if self.verbose:
            out.release()
            cv2.destroyAllWindows()

        return trajectories

    def _calc_first_trajectory_points(
            self,
            first_frame: np.ndarray,
            tracking_params
    ) -> np.ndarray:
        """
        calculates the first points on which optical flow will be executed. support few approaches, as described under
        self.poi_alg.
        :param first_frame: the first frame, from which feature points are extracted
        :param tracking_params: contains relevant parameters for features tracking / blobs tracking
        :return: the first point of interest (n_trajectories = n_points, 1, 2 = x,y)
        """
        if self.poi_alg == 'gui':
            p0 = camera_gui.select_points(first_frame)
        elif self.poi_alg == 'blobs':
            blob_detector = NaiveBlobFinder(tracking_params['NumBlobs'], **tracking_params)
            blobs = blob_detector.run(first_frame, self.verbose)
            p0 = np.round(np.array([b.pt for b in blobs]).reshape((len(blobs), 1, 2))).astype(np.float32)
        elif self.poi_alg == 'features':
            feature_params = dict(maxCorners=tracking_params['maxCorners'],
                                  qualityLevel=tracking_params['qualityLevel'],
                                  minDistance=tracking_params['minDistance'], blockSize=tracking_params['blockSize'])
            p0 = cv2.goodFeaturesToTrack(first_frame, mask=None, **feature_params)
        else:
            raise ValueError(f"Unsupported choice {self.poi_alg}. Use either 'gui', 'blobs' or 'features'")
        return p0
