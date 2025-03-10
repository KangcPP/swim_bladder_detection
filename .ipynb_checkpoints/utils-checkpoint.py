import cv2 as cv
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import seaborn as sns
from scipy import stats
import os
import re
import pickle
import imutils
from scipy.signal import find_peaks, savgol_filter
from pathlib import Path
import glob
from pathlib import Path
import pandas as pd


######################### i/o  
def init_condition(avis, x, y):
    capture = cv.VideoCapture(avis)
    _, frame1 = capture.read()
    
    # Crop to the selected 50x50 region
    frame1 = frame1[y:y+50, x:x+50]
    
    num_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # Set saturation channel
    
    return capture, num_frames, hsv

def select_roi_interactive(avis):
    # Initialize video capture
    capture = cv.VideoCapture(avis)
    ret, frame = capture.read()
    if not ret:
        raise ValueError("Failed to read video")
    
    # Variables to store ROI coordinates
    roi_selected = False
    x_center, y_center = -1, -1

    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_selected, x_center, y_center
        if event == cv.EVENT_LBUTTONDOWN:
            x_center, y_center = x, y
            roi_selected = True

    # Create window and set mouse callback
    cv.namedWindow("Select ROI Center")
    cv.setMouseCallback("Select ROI Center", mouse_callback)

    while True:
        display_frame = frame.copy()
        if roi_selected:
            # Ensure ROI stays within frame boundaries
            x = max(0, min(x_center - 25, frame.shape[1] - 50))
            y = max(0, min(y_center - 25, frame.shape[0] - 50))
            
            # Draw the 50x50 ROI rectangle
            cv.rectangle(display_frame, (x, y), (x+50, y+50), (0, 255, 0), 2)
            
            # Show cropped preview in a new window
            cropped_preview = frame[y:y+50, x:x+50]
            cv.imshow("Cropped Preview", cropped_preview)
        
        cv.imshow("Select ROI Center", display_frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') and roi_selected:  # Press 'q' to confirm
            break
        elif key == 27:  # Press ESC to exit
            cv.destroyAllWindows()
            capture.release()
            return None, None

    cv.destroyAllWindows()
    capture.release()
    
    # Final ROI coordinates (top-left corner)
    x = max(0, min(x_center - 25, frame.shape[1] - 50))
    y = max(0, min(y_center - 25, frame.shape[0] - 50))
    return x, y

def load_significant_frames(video_path, significant_indices):
    """Load frames that exceeded the MSE threshold"""
    cap = cv.VideoCapture(video_path)
    significant_frames = []
    
    for idx in tqdm(range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[y_cord:y_cord+50, x_cord:x_cord+50]
        if idx in significant_indices:
            significant_frames.append(frame)
    
    cap.release()
    return significant_frames



######################### signal processing
def noise_detection(smoothed_signal, noise_window=10,
                             min_duration=3, prominence_factor=2.0):
    # 1. Estimate noise characteristics using rolling baseline
    noise_estimates = []
    for i in range(len(smoothed_signal)):
        start = max(0, i - noise_window)
        end = min(len(smoothed_signal), i + noise_window)
        noise_estimates.append(np.std(smoothed_signal[start:end]))
    
    noise_std = np.median(noise_estimates)  # Robust noise estimate
    return noise_std

from scipy import signal
from numpy.lib.stride_tricks import sliding_window_view
def cfar_fast(
    x: np.ndarray,
    num_ref_cells: int,
    num_guard_cells: int,
    bias: float = 1,
    method=np.mean,
):
    pad = int((num_ref_cells + num_guard_cells))
    # fmt: off
    window_mean = np.pad(                                                                   # Pad front/back since n_windows < n_points
        method(                                                                             # Apply input method to remaining compute cells
            np.delete(                                                                      # Remove guard cells, CUT from computation
                sliding_window_view(x, (num_ref_cells * 2) + (num_guard_cells * 2)),        # Windows of x including CUT, guard cells, and compute cells
                np.arange(int(num_ref_cells), num_ref_cells + (num_guard_cells * 2) + 1),   # Get indices of guard cells, CUT
                axis=1),
            axis=1
        ), (pad - 1, pad),
        constant_values=(np.nan, np.nan)                                                    # Fill with NaNs
    ) + bias                                                                             
    # fmt: on
    return window_mean


########### episode detection
def find_periods(ori_up_idx, ori_down_idx):
    ori_up = sorted(ori_up_idx)
    ori_down = sorted(ori_down_idx)
    used_up = set()
    used_down = set()
    periods = []
    i_up = 0

    while i_up < len(ori_up):
        current_up = ori_up[i_up]
        if current_up in used_up:
            i_up += 1
            continue

        # Expand up chain
        up_chain = [current_up]
        used_up.add(current_up)
        j_up = i_up + 1
        while j_up < len(ori_up):
            next_up = ori_up[j_up]
            if next_up - current_up > 40:
                break
            # Check if any down exists between current_up and next_up
            has_down = False
            for d in ori_down:
                if current_up < d < next_up and d not in used_down:
                    has_down = True
                    break
            if has_down:
                break
            up_chain.append(next_up)
            used_up.add(next_up)
            current_up = next_up
            j_up += 1

        # Find down chain
        down_chain = []
        current_down = None
        # Find the first down after current_up and within 40 frames
        for d in ori_down:
            if d > current_up and d <= current_up + 40 and d not in used_down:
                current_down = d
                break
        if current_down is None:
            i_up += 1
            continue

        down_chain.append(current_down)
        used_down.add(current_down)
        # Expand down chain
        while True:
            next_down = None
            for d in ori_down:
                if d > current_down and d <= current_down + 40 and d not in used_down:
                    next_down = d
                    break
            if next_down is None:
                break
            # Check if any up exists between current_down and next_down
            has_up = False
            for u in ori_up:
                if current_down < u < next_down and u not in used_up:
                    has_up = True
                    break
            if has_up:
                break
            down_chain.append(next_down)
            used_down.add(next_down)
            current_down = next_down

        periods.append([up_chain[0], down_chain[-1]])

        # Move i_up to the next unused up index
        while i_up < len(ori_up) and ori_up[i_up] in used_up:
            i_up += 1

    return periods




####################### behaviour plottings
def get_behavior_timestamps(filename):
    t = np.loadtxt(filename)[:, 1]
    t = (t - t[0]) / 3515839
    return t


def get_tail_angles(df_tail, heading):
    xy = df_tail.values[:, ::2] + df_tail.values[:, 1::2] * 1j
    midline = -np.exp(1j * np.deg2rad(np.asarray(heading)))
    return -np.angle(np.diff(xy, axis=1) / midline[:, None])

def low_pass_filt(x, fs, cutoff, axis=0, order=2):
    from scipy.signal import butter, filtfilt

    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, x, axis=axis)

def load_data(h5_path, fs=200):
    from scipy.interpolate import interp1d

    t = get_behavior_timestamps(Path(h5_path).with_suffix(".txt"))
    df_eye = pd.read_hdf(h5_path, "eye")
    eye_angles = df_eye[[("left_eye", "angle"), ("right_eye", "angle")]].values
    t_new = np.arange(int(t[-1]) * fs) / fs
    eye_angles = interp1d(t, eye_angles, axis=0, kind="cubic")(t_new)
    df_tail = pd.read_hdf(h5_path, "tail")
    tail_angles = get_tail_angles(df_tail, df_eye["heading"].values)
    tail_angles = interp1d(t, tail_angles, axis=0, kind="cubic")(t_new)

    return t_new, eye_angles, tail_angles


