import numpy as np
import torch
import cv2

from ball_tracknet_model import BallTrackNet

def combine_three_frames(frame1, frame2, frame3, width, height):
    # Resize and type converting for each frame
    img = cv2.resize(frame1, (width, height))
    # Input must be float type
    img = img.astype(np.float32)

    # Resize
    img1 = cv2.resize(frame2, (width, height))
    # Input must be float type
    img1 = img1.astype(np.float32)

    # Resize
    img2 = cv2.resize(frame3, (width, height))
    # Input must be float type
    img2 = img2.astype(np.float32)

    # combine three images to (width, height, rgb * 3)
    imgs = np.concatenate((img, img1, img2), axis=2)

    # since the ordering of TrackNet is channels first, we need to change the axis
    imgs = np.rollaxis(imgs, 2, 0)

    return np.array(imgs)

class BallDetector:
    def __init__(self, device, save_state=None, out_channels=2):
        self.device = device

        # Load TrackNet model weights
        self.detector = BallTrackNet(out_channels=out_channels)
        saved_state_dict = torch.load(save_state, map_location=torch.device('cpu'))
        self.detector.load_state_dict(saved_state_dict['model_state'])
        self.detector.eval().to(self.device)

        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None

        self.video_width = None
        self.video_height = None
        self.model_input_width = 640
        self.model_input_height = 360

        self.threshold_dist = 100
        self.xy_coordinates = np.array([[None, None], [None, None]])

        self.bounces_indices = []

    def detect_ball(self, frame):
        # save frame dimensions
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]
        
        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()

        # detect only if 3 frames were given
        if self.last_frame is not None:
            # combine the frames into 1 input tensor
            frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame, self.model_input_width, self.model_input_height)

            frames = (torch.from_numpy(frames) / 255).to(self.device)

            x, y = self.detector.inference(frames)
            if x is not None:
                # rescale the indices to fit dimensions
                x = int(x * (self.video_width / self.model_input_width))
                y = int(y * (self.video_height / self.model_input_height))

                # check the distance from previous location and remove outliers
                if self.xy_coordinates[-1][0] is not None:
                    if np.linalg.norm(np.array([x, y]) - self.xy_coordinates[-1]) > self.threshold_dist:
                        x, y = None, None
        
            self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)

