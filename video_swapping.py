import time
import cv2
from swappr.utils import parse_timestamp_to_frame
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle
import os
import torch
from typing import TypedDict, Optional, List, Tuple, Union
import logging
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

# Add the necessary paths
sys.path.append(str(Path(__file__).parent))

from imcui.ui.utils import load_config, get_matcher_zoo, run_matching
from imcui.ui.viz import display_keypoints, display_matches
from typing import TypedDict, Optional
import numpy as np
import time
import torch
from typing import TypedDict, Optional
from imcui.hloc.matchers.matchanything import MatchAnything
from imcui.hloc import match_dense
from imcui.ui.utils import (
    get_model, proc_ransac_matches, set_null_pred,
    DEFAULT_MIN_NUM_MATCHES, ransac_zoo, DEFAULT_RANSAC_METHOD
)

class DebugVisualizer:
    """
    Debug visualization class for logo replacement pipeline.

    Handles all debugging visualizations including:
    - Bounding boxes on original frame
    - Raw and filtered keypoints from MatchAnything
    - Person mask overlays
    - Logo comparison with RANSAC-filtered match lines only
    """

    def __init__(self, enable_debug: bool = True):
        """
        Initialize the debug visualizer.

        Args:
            enable_debug: Whether to enable detailed debug visualizations
        """
        self.enable_debug = enable_debug
        self.fig = None
        self.axes = None
        self.match_ax = None
        self.axes_restructured = False

        if self.enable_debug:
            self._setup_debug_display()

    def _setup_debug_display(self):
        """Set up the matplotlib figure for debug display."""
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(20, 15))

        # Configure axes
        self.axes[0, 0].set_title('Original Frame with Debug Info')
        self.axes[0, 0].axis('off')

        self.axes[0, 1].set_title('Logo Replacement Result')
        self.axes[0, 1].axis('off')

        self.axes[1, 0].set_title('Cropped Logo (Physical)')
        self.axes[1, 0].axis('off')

        self.axes[1, 1].set_title('Reference Logo (Digital)')
        self.axes[1, 1].axis('off')

        plt.tight_layout()

    def _draw_bounding_box(self, frame: np.ndarray, bbox: np.ndarray,
                          color: tuple = (0, 255, 0), thickness: int = 3,
                          label: str = "") -> np.ndarray:
        """
        Draw bounding box on frame.

        Args:
            frame: Input frame
            bbox: Bounding box as [x1, y1, x2, y2]
            color: BGR color tuple
            thickness: Line thickness
            label: Optional label text

        Returns:
            Frame with bounding box drawn
        """
        frame_copy = frame.copy()
        x1, y1, x2, y2 = bbox.astype(int)

        # Draw rectangle
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)

        # Draw label if provided
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Draw background rectangle
            cv2.rectangle(frame_copy, (x1, y1 - text_height - 10),
                         (x1 + text_width, y1), color, -1)

            # Draw text
            cv2.putText(frame_copy, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

        return frame_copy

    def _draw_keypoints_dual(self, frame: np.ndarray,
                            raw_keypoints: Optional[np.ndarray] = None,
                            filtered_keypoints: Optional[np.ndarray] = None,
                            raw_color: tuple = (100, 150, 255),  # Light blue for raw
                            filtered_color: tuple = (255, 0, 0),  # Red for filtered
                            radius: int = 3) -> np.ndarray:
        """
        Draw both raw and filtered keypoints on frame with different colors.

        Args:
            frame: Input frame
            raw_keypoints: Raw keypoints from MatchAnything
            filtered_keypoints: RANSAC-filtered keypoints
            raw_color: BGR color for raw keypoints
            filtered_color: BGR color for filtered keypoints
            radius: Circle radius

        Returns:
            Frame with keypoints drawn
        """
        frame_copy = frame.copy()

        # Draw raw keypoints first (underneath)
        if raw_keypoints is not None and len(raw_keypoints) > 0:
            for kp in raw_keypoints:
                x, y = kp.astype(int)
                cv2.circle(frame_copy, (x, y), radius, raw_color, -1)

        # Draw filtered keypoints on top
        if filtered_keypoints is not None and len(filtered_keypoints) > 0:
            for kp in filtered_keypoints:
                x, y = kp.astype(int)
                cv2.circle(frame_copy, (x, y), radius, filtered_color, -1)

        return frame_copy

    def _overlay_mask(self, frame: np.ndarray, mask: np.ndarray,
                     color: tuple = (0, 0, 255), alpha: float = 0.3) -> np.ndarray:
        """
        Overlay mask on frame with transparency.

        Args:
            frame: Input frame
            mask: Binary mask
            color: BGR color tuple
            alpha: Transparency factor

        Returns:
            Frame with mask overlay
        """
        frame_copy = frame.copy()

        # Create colored mask
        colored_mask = np.zeros_like(frame_copy)
        colored_mask[mask > 0] = color

        # Blend with original frame
        result = cv2.addWeighted(frame_copy, 1 - alpha, colored_mask, alpha, 0)

        return result

    def _draw_matches_filtered_only(self, img1: np.ndarray, img2: np.ndarray,
                                   raw_kpts1: np.ndarray, raw_kpts2: np.ndarray,
                                   filtered_kpts1: np.ndarray, filtered_kpts2: np.ndarray) -> np.ndarray:
        """
        Draw matches between two images showing raw and filtered keypoints with match lines only for filtered.

        Args:
            img1: First image (cropped logo)
            img2: Second image (reference logo)
            raw_kpts1: Raw keypoints in first image
            raw_kpts2: Raw keypoints in second image
            filtered_kpts1: RANSAC-filtered keypoints in first image
            filtered_kpts2: RANSAC-filtered keypoints in second image

        Returns:
            Combined image with keypoints and filtered match lines
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Create combined image
        combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        combined[:h1, :w1] = img1
        combined[:h2, w1:w1+w2] = img2

        # Draw raw keypoints first (light blue)
        for kp in raw_kpts1:
            x, y = kp.astype(int)
            cv2.circle(combined, (x, y), 2, (100, 150, 255), -1)  # Light blue

        for kp in raw_kpts2:
            x, y = kp.astype(int)
            cv2.circle(combined, (x + w1, y), 2, (100, 150, 255), -1)  # Light blue

        # Draw filtered keypoints on top (red)
        for kp in filtered_kpts1:
            x, y = kp.astype(int)
            cv2.circle(combined, (x, y), 3, (255, 0, 0), -1)  # Red

        for kp in filtered_kpts2:
            x, y = kp.astype(int)
            cv2.circle(combined, (x + w1, y), 3, (255, 0, 0), -1)  # Red

        # Draw match lines only for filtered keypoints (yellow)
        for i in range(min(len(filtered_kpts1), len(filtered_kpts2))):
            x1, y1 = filtered_kpts1[i].astype(int)
            x2, y2 = filtered_kpts2[i].astype(int)
            cv2.line(combined, (x1, y1), (x2 + w1, y2), (0, 255, 255), 1)

        return combined

    def _restructure_for_matches(self):
        """Restructure the layout to show match visualization."""
        if not self.axes_restructured and self.fig is not None and self.axes is not None:
            try:
                # Remove bottom axes
                self.fig.delaxes(self.axes[1, 0])
                self.fig.delaxes(self.axes[1, 1])

                # Create new axis spanning both bottom plots
                self.match_ax = self.fig.add_subplot(2, 1, 2)
                self.match_ax.axis('off')

                self.axes_restructured = True
                plt.tight_layout()
            except (KeyError, ValueError):
                # Axes may already be removed, ignore error
                pass

    def _reset_layout(self):
        """Reset to original 2x2 layout."""
        if self.axes_restructured and self.fig is not None and self.axes is not None:
            try:
                if self.match_ax is not None:
                    self.fig.delaxes(self.match_ax)
                    self.match_ax = None

                # Recreate bottom axes
                self.axes = self.axes.tolist()  # Convert to list to modify
                self.axes[1] = [
                    self.fig.add_subplot(2, 2, 3),
                    self.fig.add_subplot(2, 2, 4)
                ]
                self.axes = np.array(self.axes)  # Convert back to numpy array

                self.axes[1, 0].axis('off')
                self.axes[1, 1].axis('off')

                self.axes_restructured = False
                plt.tight_layout()
            except (KeyError, ValueError, AttributeError):
                # Handle any errors gracefully
                pass

    def update_display(self, frame_number: int, original_frame: np.ndarray,
                      result_frame: np.ndarray, debug_info: Optional[dict] = None):
        """
        Update the display with current frame information.

        Args:
            frame_number: Current frame number
            original_frame: Original input frame
            result_frame: Processed result frame
            debug_info: Optional debug information dictionary
        """
        if not self.enable_debug:
            # Simple OpenCV display
            cv2.imshow('Logo Replacement Result', cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            return

        if self.axes is None or self.fig is None:
            return

        # Prepare debug frame
        debug_frame = original_frame.copy()

        if debug_info:
            # Draw bounding box if available
            if 'bbox' in debug_info and debug_info['bbox'] is not None:
                debug_frame = self._draw_bounding_box(
                    debug_frame, debug_info['bbox'],
                    color=(0, 255, 0), label="Budlight Logo"
                )

            # Draw both raw and filtered keypoints if available
            raw_keypoints = debug_info.get('raw_keypoints')
            filtered_keypoints = debug_info.get('filtered_keypoints')

            if raw_keypoints is not None or filtered_keypoints is not None:
                debug_frame = self._draw_keypoints_dual(
                    debug_frame,
                    raw_keypoints=raw_keypoints,
                    filtered_keypoints=filtered_keypoints,
                    raw_color=(100, 150, 255),  # Light blue for raw
                    filtered_color=(255, 0, 0)  # Red for filtered
                )

            # Overlay person mask if available
            if 'person_mask' in debug_info and debug_info['person_mask'] is not None:
                debug_frame = self._overlay_mask(
                    debug_frame, debug_info['person_mask'],
                    color=(0, 0, 255), alpha=0.3
                )

        # Check if we need to show matches
        show_matches = (debug_info and
                       'cropped_logo' in debug_info and
                       'reference_logo' in debug_info and
                       'raw_match_keypoints1' in debug_info and
                       'raw_match_keypoints2' in debug_info and
                       'filtered_match_keypoints1' in debug_info and
                       'filtered_match_keypoints2' in debug_info and
                       len(debug_info['filtered_match_keypoints1']) > 0 and
                       len(debug_info['filtered_match_keypoints2']) > 0)

        if show_matches and not self.axes_restructured:
            self._restructure_for_matches()
        elif not show_matches and self.axes_restructured:
            self._reset_layout()

        # Clear and update top axes
        self.axes[0, 0].clear()
        self.axes[0, 0].imshow(debug_frame)
        self.axes[0, 0].set_title(f'Original Frame {frame_number} with Debug Info')
        self.axes[0, 0].axis('off')

        self.axes[0, 1].clear()
        self.axes[0, 1].imshow(result_frame)
        self.axes[0, 1].set_title(f'Logo Replacement Result {frame_number}')
        self.axes[0, 1].axis('off')

        # Update display based on layout
        if show_matches and debug_info is not None:
            # Show enhanced match visualization
            cropped_logo = debug_info['cropped_logo']
            reference_logo = debug_info['reference_logo']
            raw_kpts1 = debug_info['raw_match_keypoints1']
            raw_kpts2 = debug_info['raw_match_keypoints2']
            filtered_kpts1 = debug_info['filtered_match_keypoints1']
            filtered_kpts2 = debug_info['filtered_match_keypoints2']

            match_vis = self._draw_matches_filtered_only(
                cropped_logo, reference_logo,
                raw_kpts1, raw_kpts2,
                filtered_kpts1, filtered_kpts2
            )

            if self.match_ax is not None:
                self.match_ax.clear()
                self.match_ax.imshow(match_vis)
                self.match_ax.set_title(f'Logo Matches: {len(raw_kpts1)} raw → {len(filtered_kpts1)} filtered (RANSAC lines only)')
                self.match_ax.axis('off')
        else:
            # Show individual logos
            if debug_info and 'cropped_logo' in debug_info and 'reference_logo' in debug_info:
                if not self.axes_restructured:
                    self.axes[1, 0].clear()
                    self.axes[1, 0].imshow(debug_info['cropped_logo'])
                    self.axes[1, 0].set_title('Cropped Logo (Physical)')
                    self.axes[1, 0].axis('off')

                    self.axes[1, 1].clear()
                    self.axes[1, 1].imshow(debug_info['reference_logo'])
                    self.axes[1, 1].set_title('Reference Logo (Digital)')
                    self.axes[1, 1].axis('off')

        # Add frame statistics
        if debug_info and 'stats' in debug_info:
            stats = debug_info['stats']
            stats_text = f"Frame: {frame_number}\n"
            if 'num_raw_matches' in stats and 'num_filtered_matches' in stats:
                stats_text += f"Matches: {stats['num_raw_matches']} raw → {stats['num_filtered_matches']} filtered\n"
            if 'ekf_info' in stats:
                ekf_info = stats['ekf_info']
                if ekf_info is not None:
                    stats_text += f"EKF Covariance: {ekf_info.get('covariance_trace', 0):.6f}\n"
            if 'processing_time' in stats:
                stats_text += f"Processing: {stats['processing_time']:.3f}s\n"

            self.fig.suptitle(stats_text, fontsize=10, ha='left', va='top')

        plt.draw()
        plt.pause(0.0005)  # Small pause to allow matplotlib to update

    def close(self):
        """Close the debug display."""
        if self.enable_debug:
            plt.ioff()
            if self.fig:
                plt.close(self.fig)
        else:
            cv2.destroyAllWindows()

class HomographyEKF:
    """
    Extended Kalman Filter for homography matrix stabilization.

    This class implements an EKF to smooth homography matrices over time,
    reducing jitter and improving temporal consistency in video logo replacement.

    The state vector represents the 8 independent parameters of the homography matrix
    (excluding the last element which is normalized to 1) and their velocities.

    State vector: [h00, h01, h02, h10, h11, h12, h20, h21,
                   h00_vel, h01_vel, h02_vel, h10_vel, h11_vel, h12_vel, h20_vel, h21_vel]
    """

    def __init__(self,
                 dt: float = 1.0,
                 process_noise_std: float = 0.01,
                 measurement_noise_std: float = 0.1,
                 initial_covariance: float = 1.0):
        """
        Initialize the Homography EKF.

        Args:
            dt: Time step between frames
            process_noise_std: Standard deviation of process noise
            measurement_noise_std: Standard deviation of measurement noise
            initial_covariance: Initial state covariance
        """
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.initial_covariance = initial_covariance

        # State dimension: 8 homography parameters + 8 velocities = 16
        self.state_dim = 16
        # Measurement dimension: 8 homography parameters
        self.measurement_dim = 8

        # Initialize EKF
        self.ekf = ExtendedKalmanFilter(dim_x=self.state_dim, dim_z=self.measurement_dim)

        # Initialize state vector (identity homography + zero velocities)
        self.ekf.x = np.zeros(self.state_dim)
        self.ekf.x[0] = 1.0  # h00 = 1
        self.ekf.x[4] = 1.0  # h11 = 1
        # h22 is implicitly 1 (not in state vector)

        # Initialize covariance matrix
        self.ekf.P = np.eye(self.state_dim) * self.initial_covariance

        # State transition matrix (constant velocity model)
        self.ekf.F = np.eye(self.state_dim)
        # Position = position + velocity * dt
        for i in range(8):
            self.ekf.F[i, i + 8] = self.dt

        # Process noise covariance matrix
        self.ekf.Q = np.eye(self.state_dim) * (self.process_noise_std ** 2)

        # Measurement noise covariance matrix
        self.ekf.R = np.eye(self.measurement_dim) * (self.measurement_noise_std ** 2)

        self.initialized = False

    def measurement_function(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function that maps state to measurement.
        For homography, we directly observe the first 8 parameters.

        Args:
            x: State vector

        Returns:
            Expected measurement vector
        """
        return x[:8]  # First 8 elements are the homography parameters

    def measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of the measurement function.

        Args:
            x: State vector

        Returns:
            Jacobian matrix
        """
        H = np.zeros((self.measurement_dim, self.state_dim))
        # Identity for the first 8 parameters, zero for velocities
        for i in range(8):
            H[i, i] = 1.0
        return H

    def homography_to_vector(self, H: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert 3x3 homography matrix to 8-element vector.

        Args:
            H: 3x3 homography matrix

        Returns:
            8-element vector [h00, h01, h02, h10, h11, h12, h20, h21] or None if invalid
        """
        if H is None or H.size == 0:
            return None

        # Normalize by h22 to ensure h22 = 1
        H_normalized = H / H[2, 2]

        # Extract 8 parameters (excluding h22 which is 1)
        h_vector = np.array([
            H_normalized[0, 0], H_normalized[0, 1], H_normalized[0, 2],
            H_normalized[1, 0], H_normalized[1, 1], H_normalized[1, 2],
            H_normalized[2, 0], H_normalized[2, 1]
        ])

        return h_vector

    def vector_to_homography(self, h_vector: np.ndarray) -> np.ndarray:
        """
        Convert 8-element vector to 3x3 homography matrix.

        Args:
            h_vector: 8-element vector [h00, h01, h02, h10, h11, h12, h20, h21]

        Returns:
            3x3 homography matrix
        """
        H = np.array([
            [h_vector[0], h_vector[1], h_vector[2]],
            [h_vector[3], h_vector[4], h_vector[5]],
            [h_vector[6], h_vector[7], 1.0]
        ])

        return H

    def predict(self) -> np.ndarray:
        """
        Predict the next state using the constant velocity model.

        Returns:
            Predicted homography matrix
        """
        self.ekf.predict()

        # Extract homography parameters from state
        h_vector = self.ekf.x[:8]
        return self.vector_to_homography(h_vector)

    def update(self, H_measured: np.ndarray) -> np.ndarray:
        """
        Update the filter with a new homography measurement.

        Args:
            H_measured: 3x3 measured homography matrix

        Returns:
            Smoothed homography matrix
        """
        if H_measured is None or H_measured.size == 0:
            # No measurement available, return prediction
            return self.predict()

        # Convert homography to measurement vector
        h_vector = self.homography_to_vector(H_measured)

        if h_vector is None:
            return self.predict()

        if not self.initialized:
            # Initialize state with first measurement
            self.ekf.x[:8] = h_vector
            self.ekf.x[8:] = 0.0  # Zero initial velocities
            self.initialized = True
            return H_measured

        # Predict step
        self.ekf.predict()

        # Update step with measurement function and Jacobian
        self.ekf.update(h_vector, self.measurement_jacobian, self.measurement_function)

        # Extract smoothed homography parameters
        h_smoothed = self.ekf.x[:8]
        H_smoothed = self.vector_to_homography(h_smoothed)

        return H_smoothed

    def get_state_info(self) -> dict:
        """
        Get current state information for debugging.

        Returns:
            Dictionary containing state information
        """
        return {
            'state': self.ekf.x.copy(),
            'covariance_trace': np.trace(self.ekf.P),
            'homography_params': self.ekf.x[:8].copy(),
            'velocities': self.ekf.x[8:].copy(),
            'initialized': self.initialized
        }

# Simple configuration for frame-by-frame processing
MATCHING_CONFIDENCE_THRESHOLD = 0.1
MAX_KEYPOINTS = 6000
MIN_KP_FOR_HOMOGRAPHY = 200
MIN_BBOX_W = 200
MIN_BBOX_H = 50
RANSAC_THRESHOLD = 30
DEBUG = True

# Optical Flow Tracking Configuration
KEYFRAME_INTERVAL = 60  # Run MatchAnything every N frames
MIN_TRACKING_POINTS = 150  # Minimum points to continue tracking
MAX_FB_ERROR = 1.0  # Forward-backward error threshold (pixels) - stricter for better quality
LK_WIN_SIZE = (15, 15)  # LK window size
LK_MAX_LEVEL = 3  # Pyramid levels
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

# EKF Configuration for homography stabilization
EKF_ENABLED = True
EKF_PROCESS_NOISE_STD = 0.01    # Lower = smoother but slower response
EKF_MEASUREMENT_NOISE_STD = 0.1   # Lower = trust measurements more
EKF_INITIAL_COVARIANCE = 0.1     # Initial uncertainty

# Load overlay components
# Configuration for transparency
is_transparent = False  # Set to True for transparent SPATEN logo, False for floor color

# Determine filename based on transparency flag
if is_transparent:
    components_filename = "overlay_components_transparent.pkl"
else:
    components_filename = "overlay_components_not_transparent.pkl"

overlay_components_path = Path(__file__).parent / "overlay_components" / components_filename
print(f"Loading overlay components from: {overlay_components_path}")

with open(overlay_components_path, "rb") as f:
    overlay_components = pickle.load(f)

# Extract components for global use
budlight_downsampled = overlay_components["budlight_downsampled"]
spaten_resized = overlay_components["spaten_resized"]
center_x = overlay_components["center_x"]
center_y = overlay_components["center_y"]
loaded_is_transparent = overlay_components.get("is_transparent", False)

print(f"Loaded overlay components:")
print(f"  - Budlight reference: {budlight_downsampled.shape}")
print(f"  - Spaten replacement: {spaten_resized.shape}")
print(f"  - Overlay offset: ({center_x}, {center_y})")
print(f"  - Transparency mode: {'Transparent' if loaded_is_transparent else 'Non-transparent (floor color)'}")

# Verify configuration matches loaded components
if is_transparent != loaded_is_transparent:
    print(f"WARNING: Configuration mismatch!")
    print(f"  Script is_transparent: {is_transparent}")
    print(f"  Loaded is_transparent: {loaded_is_transparent}")
    print(f"  Using loaded configuration: {loaded_is_transparent}")
    is_transparent = loaded_is_transparent

# Load person segmentation model
seg_model_path = "/home/sebastiangarcia/projects/swappr/yolo11n-seg.pt"
print(f"Loading YOLO segmentation model from: {seg_model_path}")
person_seg_model = YOLO(seg_model_path)
print(f"YOLO segmentation model loaded successfully!")

# Person class IDs in COCO dataset (0 = person)
PERSON_CLASS_IDS = [0]

# Raw match prediction
class MatchPrediction(TypedDict):
    # Matched points before filtering with RANSAC
    # the points x,y coordinate must match with the shape of the
    # input images ("mkeypoints0_orig", "mkeypoints1_orig")
    mkpts0: np.ndarray # matched points in image0 as Nx2 numpy array.
    mkpts1: np.ndarray # matched points in image1 as Nx2 numpy array.
    # Additional required keys for RANSAC filtering
    mkeypoints0_orig: np.ndarray  # matched points in original image0 coordinates
    mkeypoints1_orig: np.ndarray  # matched points in original image1 coordinates
    mconf: np.ndarray  # confidence scores for matches
    image0_orig: np.ndarray  # original image0
    image1_orig: np.ndarray  # original image1

class FilteredMatchPrediction(MatchPrediction):
    # Homography matrix estimated after RANSAC filtering
    H: np.ndarray
    # Matched points after filtering with RANSAC
    # the points x,y coordinate must match with the shape of the
    # input images ("mkeypoints0_orig", "mkeypoints1_orig")
    mmkpts0: np.ndarray # matched points in image0 as Nx2 numpy array.
    mmkpts1: np.ndarray # matched points in image1 as Nx2 numpy array.
    mmkeypoints0_orig: np.ndarray  # RANSAC filtered matches in original image0 coordinates
    mmkeypoints1_orig: np.ndarray  # RANSAC filtered matches in original image1 coordinates
    mmconf: np.ndarray  # confidence scores for RANSAC filtered matches

def load_matchanything_model(
    model_name: str = "matchanything_eloftr",
    match_threshold: float = 0.01,
    extract_max_keypoints: int = 2000,
    log_timing: bool = False
) -> tuple[MatchAnything, dict]:
    """
    Load and return a ready-to-use MatchAnything model instance with its preprocessing config.

    Args:
        model_name: Either "matchanything_eloftr" or "matchanything_roma"
        match_threshold: Matching threshold for the model
        extract_max_keypoints: Maximum number of keypoints to extract
        log_timing: Whether to log loading time

    Returns:
        Tuple of (loaded MatchAnything model instance, preprocessing configuration)
    """
    if log_timing:
        t0 = time.time()

    # Load configuration and matcher zoo
    config_path = Path(__file__).parent / "config/config.yaml"
    cfg = load_config(config_path)
    matcher_zoo = get_matcher_zoo(cfg["matcher_zoo"])

    # Get model configuration
    model_config = matcher_zoo[model_name]
    match_conf = model_config["matcher"]

    # Update model configuration
    match_conf["model"]["match_threshold"] = match_threshold
    match_conf["model"]["max_keypoints"] = extract_max_keypoints

    # Load the model
    model = get_model(match_conf)

    # Get the original preprocessing configuration (which has the correct force_resize setting)
    preprocessing_conf = match_conf["preprocessing"].copy()

    if log_timing:
        print(f"Model {model_name} loaded in {time.time() - t0:.3f}s")
        print(f"Using preprocessing config: {preprocessing_conf}")

    return model, preprocessing_conf

def run_matching_simple(
    model: MatchAnything,
    img0_frame_logo: np.ndarray,
    img1_ref_logo: np.ndarray,
    preprocessing_conf: Optional[dict] = None,
    match_threshold: float = 0.01,
    extract_max_keypoints: int = 2000,
    keypoint_threshold: float = 0.05,
    log_timing: bool = False
) -> MatchPrediction:
    """
    Runs MatchAnything model to estimate matches between
    a logo that appears with an arbitrary viewpoint from the frame
    of a UFC fight and the same logo observed from the bird's eye
    view of the octagon.

    The matches that this function outputs will be used to estimate
    a homography matrix for logo replacement.

    Parameters:
        model: MatchAnything: instantiated ROMA or ELoFTR model.
        img0_frame_logo: is the cropped logo that was cropped in a frame during the fight,
        img1_ref_logo: is the image of the logo taken from a top view before the UFC fight began
        preprocessing_conf: preprocessing configuration from load_matchanything_model
        match_threshold: matching threshold for the model
        extract_max_keypoints: maximum number of keypoints to extract
        keypoint_threshold: keypoint detection threshold
        log_timing: whether to log inference time

    Returns:
        MatchPrediction with matched keypoints and metadata
    """
    if log_timing:
        t0 = time.time()

    # Use the original preprocessing configuration or create a default one
    if preprocessing_conf is None:
        preprocessing_conf = {
            "grayscale": False,
            "resize_max": 832,
            "dfactor": 32,
            "width": 640,
            "height": 480,
            "force_resize": True,  # Use True by default to match original config
        }

    # Update model thresholds
    model.conf["match_threshold"] = match_threshold
    model.conf["max_keypoints"] = extract_max_keypoints

    # Run inference using the same pipeline as the original code
    with torch.no_grad():
        pred = match_dense.match_images(
            model, img0_frame_logo, img1_ref_logo, preprocessing_conf, device="cuda" if torch.cuda.is_available() else "cpu"
        )

    # Extract the required information for MatchPrediction
    result = MatchPrediction(
        mkpts0=pred["mkeypoints0_orig"],
        mkpts1=pred["mkeypoints1_orig"],
        mkeypoints0_orig=pred["mkeypoints0_orig"],
        mkeypoints1_orig=pred["mkeypoints1_orig"],
        mconf=pred["mconf"],
        image0_orig=pred["image0_orig"],
        image1_orig=pred["image1_orig"]
    )

    if log_timing:
        print(f"Matching inference completed in {time.time() - t0:.3f}s")

    return result

def filter_matches_ransac(
    prediction: MatchPrediction,
    ransac_method: str = "CV2_USAC_MAGSAC_PLUS",
    ransac_reproj_threshold: float = 8.0,
    ransac_confidence: float = 0.999,
    ransac_max_iter: int = 10000,
    log_timing: bool = False
) -> FilteredMatchPrediction:
    """
    Filter matches using RANSAC and estimate Homography matrix.

    Args:
        prediction: MatchPrediction from run_matching_simple
        ransac_method: RANSAC method to use
        ransac_reproj_threshold: RANSAC reprojection threshold
        ransac_confidence: RANSAC confidence level
        ransac_max_iter: RANSAC maximum iterations
        log_timing: whether to log processing time

    Returns:
        FilteredMatchPrediction with RANSAC-filtered matches and homography matrix
    """
    if log_timing:
        t0 = time.time()

    mkpts0 = prediction["mkeypoints0_orig"]
    mkpts1 = prediction["mkeypoints1_orig"]
    mconf = prediction["mconf"]

    # Check if we have enough matches
    if len(mkpts0) < DEFAULT_MIN_NUM_MATCHES:
        if log_timing:
            print(f"Not enough matches ({len(mkpts0)} < {DEFAULT_MIN_NUM_MATCHES}), returning empty result")

        return FilteredMatchPrediction(
            **prediction,  # Keep original data
            H=np.array([]),
            mmkpts0=np.array([]).reshape(0, 2),
            mmkpts1=np.array([]).reshape(0, 2),
            mmkeypoints0_orig=np.array([]).reshape(0, 2),
            mmkeypoints1_orig=np.array([]).reshape(0, 2),
            mmconf=np.array([])
        )

    # Validate ransac method
    if ransac_method not in ransac_zoo.keys():
        ransac_method = DEFAULT_RANSAC_METHOD

    try:
        # Compute Homography using RANSAC
        H, mask_h = proc_ransac_matches(
            mkpts1,  # Note: swapped order as in original code
            mkpts0,
            ransac_method,
            ransac_reproj_threshold,
            ransac_confidence,
            ransac_max_iter,
            geometry_type="Homography",
        )

        if H is not None and mask_h is not None:
            # Filter matches using the RANSAC mask
            filtered_mkpts0 = mkpts0[mask_h]
            filtered_mkpts1 = mkpts1[mask_h]
            filtered_mconf = mconf[mask_h]

            result = FilteredMatchPrediction(
                **prediction,  # Keep original data
                H=H,
                mmkpts0=filtered_mkpts0,
                mmkpts1=filtered_mkpts1,
                mmkeypoints0_orig=filtered_mkpts0,
                mmkeypoints1_orig=filtered_mkpts1,
                mmconf=filtered_mconf
            )
        else:
            # RANSAC failed, return empty filtered results
            result = FilteredMatchPrediction(
                **prediction,  # Keep original data
                H=np.array([]),
                mmkpts0=np.array([]).reshape(0, 2),
                mmkpts1=np.array([]).reshape(0, 2),
                mmkeypoints0_orig=np.array([]).reshape(0, 2),
                mmkeypoints1_orig=np.array([]).reshape(0, 2),
                mmconf=np.array([])
            )

    except Exception as e:
        print(f"RANSAC failed with error: {e}")
        # Return empty filtered results on any error
        result = FilteredMatchPrediction(
            **prediction,  # Keep original data
            H=np.array([]),
            mmkpts0=np.array([]).reshape(0, 2),
            mmkpts1=np.array([]).reshape(0, 2),
            mmkeypoints0_orig=np.array([]).reshape(0, 2),
            mmkeypoints1_orig=np.array([]).reshape(0, 2),
            mmconf=np.array([])
        )

    if log_timing:
        num_filtered = len(result["mmkpts0"]) if len(result["H"]) > 0 else 0
        print(f"RANSAC filtering completed in {time.time() - t0:.3f}s")
        print(f"Filtered {len(mkpts0)} → {num_filtered} matches")

    return result

def expand_box(bbox: np.ndarray, expansion_factor: float, img_width: Optional[int] = None, img_height: Optional[int] = None) -> tuple[int, int, int, int]:
    """
    Expand a bounding box by a given percentage.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2] numpy array
        expansion_factor: Factor to expand the box by (e.g., 0.1 for 10% expansion)
        img_width: Maximum width to clamp the box to (optional)
        img_height: Maximum height to clamp the box to (optional)

    Returns:
        Tuple of (x1, y1, x2, y2) expanded coordinates
    """
    x1, y1, x2, y2 = bbox

    # Calculate current width and height
    width = x2 - x1
    height = y2 - y1

    # Calculate expansion amounts
    expand_w = int(width * expansion_factor / 2)  # Divide by 2 since we expand both sides
    expand_h = int(height * expansion_factor / 2)

    # Expand the box
    new_x1 = x1 - expand_w
    new_y1 = y1 - expand_h
    new_x2 = x2 + expand_w
    new_y2 = y2 + expand_h

    # Clamp to image boundaries if provided
    if img_width is not None:
        new_x1 = max(0, new_x1)
        new_x2 = min(img_width, new_x2)

    if img_height is not None:
        new_y1 = max(0, new_y1)
        new_y2 = min(img_height, new_y2)

    return new_x1, new_y1, new_x2, new_y2

def get_person_masks(frame: np.ndarray,
                    segmentation_model,
                    confidence_threshold: float = 0.5,
                    log_timing: bool = False) -> np.ndarray:
    """
    Get combined segmentation masks for all people in the frame.

    Args:
        frame: Input frame (RGB format)
        segmentation_model: YOLO segmentation model
        confidence_threshold: Confidence threshold for person detection
        log_timing: Whether to log processing time

    Returns:
        Combined binary mask for all detected people (same size as frame)
    """
    if log_timing:
        t0 = time.time()

    frame_height, frame_width = frame.shape[:2]

    # Initialize combined mask
    combined_person_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # Run segmentation inference
    results = segmentation_model(frame, conf=confidence_threshold, verbose=False)

    # Process results
    if results[0].masks is not None:
        boxes = results[0].boxes
        masks = results[0].masks

        # Loop through all detections
        for box, mask in zip(boxes, masks):
            # Check if it's a person (class ID 0 in COCO)
            if int(box.cls) in PERSON_CLASS_IDS:
                # Get mask array
                seg_mask = mask.data.cpu().numpy()
                seg_mask = np.squeeze(seg_mask)

                # Resize mask to match frame dimensions
                if seg_mask.shape[0] > 0 and seg_mask.shape[1] > 0:
                    seg_mask = cv2.resize(seg_mask, (frame_width, frame_height))

                    # Add to combined mask (union operation)
                    person_area = (seg_mask > 0.5).astype(np.uint8)
                    combined_person_mask = np.logical_or(combined_person_mask, person_area).astype(np.uint8)

    if log_timing:
        num_people = len([box for box in results[0].boxes if int(box.cls) in PERSON_CLASS_IDS]) if results[0].boxes is not None else 0
        print(f"Person segmentation completed in {time.time() - t0:.3f}s, detected {num_people} people")

    return combined_person_mask

def apply_person_occlusion(frame_with_logo: np.ndarray,
                          original_frame: np.ndarray,
                          person_mask: np.ndarray,
                          log_timing: bool = False) -> np.ndarray:
    """
    Apply person occlusion to bring people in front of the replaced logo.

    Args:
        frame_with_logo: Frame with logo replacement applied
        original_frame: Original frame without logo replacement
        person_mask: Binary mask indicating person locations
        log_timing: Whether to log processing time

    Returns:
        Frame with people brought forward in front of logo
    """
    if log_timing:
        t0 = time.time()

    # Create result frame starting with logo-replaced frame
    result_frame = frame_with_logo.copy()

    # Create 3-channel mask for RGB application
    person_mask_3d = np.stack([person_mask, person_mask, person_mask], axis=-1).astype(bool)

    # Apply person pixels from original frame wherever people are detected
    result_frame[person_mask_3d] = original_frame[person_mask_3d]

    if log_timing:
        person_pixel_count = np.sum(person_mask)
        print(f"Person occlusion applied in {time.time() - t0:.3f}s, restored {person_pixel_count} person pixels")

    return result_frame

def track_keypoints_lk(prev_gray: np.ndarray,
                      curr_gray: np.ndarray,
                      prev_keypoints: np.ndarray,
                      log_timing: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Track keypoints using Lucas-Kanade optical flow with forward-backward consistency check.

    Args:
        prev_gray: Previous frame in grayscale
        curr_gray: Current frame in grayscale
        prev_keypoints: Previous keypoints as Nx2 array
        log_timing: Whether to log processing time

    Returns:
        Tuple of (tracked_keypoints, good_mask, tracking_stats)
    """
    if log_timing:
        t0 = time.time()

    # Convert keypoints to the format expected by OpenCV (Nx1x2, float32)
    p0 = prev_keypoints.reshape(-1, 1, 2).astype(np.float32)

    # Forward tracking: prev_frame -> curr_frame
    p1, st_forward, err_forward = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0,
        nextPts=np.empty_like(p0),
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA
    )

    # Backward tracking: curr_frame -> prev_frame (for consistency check)
    p0_back, st_backward, err_backward = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, p1,
        nextPts=np.empty_like(p1),
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA
    )

    # Forward-backward consistency check
    fb_error = np.linalg.norm(p0_back - p0, axis=2).flatten()

    # Combine all quality checks
    good_forward = st_forward.flatten() == 1
    good_backward = st_backward.flatten() == 1
    good_fb_error = fb_error < MAX_FB_ERROR

    # Final good mask: all checks must pass
    good_mask = good_forward & good_backward & good_fb_error

    # Extract good tracked points
    tracked_keypoints = p1[good_mask].reshape(-1, 2)

    # Tracking statistics
    tracking_stats = {
        'total_points': len(prev_keypoints),
        'tracked_points': len(tracked_keypoints),
        'survival_rate': len(tracked_keypoints) / len(prev_keypoints) if len(prev_keypoints) > 0 else 0.0,
        'avg_fb_error': np.mean(fb_error[good_mask]) if np.any(good_mask) else float('inf'),
        'max_fb_error': np.max(fb_error[good_mask]) if np.any(good_mask) else float('inf')
    }

    if log_timing:
        print(f"LK tracking completed in {time.time() - t0:.3f}s")
        print(f"  Tracked {tracking_stats['tracked_points']}/{tracking_stats['total_points']} points ({tracking_stats['survival_rate']:.1%})")
        print(f"  Avg FB error: {tracking_stats['avg_fb_error']:.2f}px")

    return tracked_keypoints, good_mask, tracking_stats

def should_reseed_keyframe(tracking_stats: dict, frame_count: int) -> tuple[bool, str]:
    """
    Determine if we should reseed with MatchAnything based on tracking quality.

    Args:
        tracking_stats: Statistics from LK tracking
        frame_count: Current frame count since last keyframe

    Returns:
        Tuple of (should_reseed, reason)
    """
    # Regular keyframe interval
    if frame_count >= KEYFRAME_INTERVAL:
        return True, f"regular_interval_{KEYFRAME_INTERVAL}"

    # Not enough tracking points survived
    if tracking_stats['tracked_points'] < MIN_TRACKING_POINTS:
        return True, f"insufficient_points_{tracking_stats['tracked_points']}"

    # Poor survival rate (< 60%)
    if tracking_stats['survival_rate'] < 0.6:
        return True, f"poor_survival_{tracking_stats['survival_rate']:.1%}"

    # High forward-backward error (> 1.5px average)
    if tracking_stats['avg_fb_error'] > MAX_FB_ERROR:
        return True, f"high_fb_error_{tracking_stats['avg_fb_error']:.2f}px"

    return False, "continue_tracking"

def filter_keypoints_for_tracking(keypoints: np.ndarray, max_points: int = 800) -> np.ndarray:
    """
    Filter keypoints to the strongest corners for better LK tracking.

    Args:
        keypoints: Input keypoints as Nx2 array
        max_points: Maximum number of points to keep

    Returns:
        Filtered keypoints as Nx2 array
    """
    if len(keypoints) <= max_points:
        return keypoints

    # For now, just take the first max_points (could improve with corner strength)
    return keypoints[:max_points]

def map_budlight_to_spaten_coordinates(budlight_points: np.ndarray,
                                     center_x: int, center_y: int) -> np.ndarray:
    """
    Map coordinates from Budlight logo space to SPATEN logo space.

    This function takes keypoints detected in the Budlight logo and maps them
    to the corresponding coordinates in the SPATEN logo, accounting for the
    overlay offset used when positioning SPATEN on top of Budlight.

    Args:
        budlight_points: Array of shape (N, 2) with (x, y) coordinates in Budlight logo space
        center_x: X offset used to center SPATEN on Budlight
        center_y: Y offset used to center SPATEN on Budlight

    Returns:
        Array of shape (N, 2) with corresponding (x, y) coordinates in SPATEN logo space
    """
    # Convert Budlight coordinates to SPATEN coordinates
    # Since SPATEN was placed at (center_x, center_y) on Budlight,
    # we need to subtract these offsets to get SPATEN-local coordinates
    spaten_points = budlight_points.copy()
    spaten_points[:, 0] -= center_x  # Adjust x coordinates
    spaten_points[:, 1] -= center_y  # Adjust y coordinates

    return spaten_points

def replace_logo_in_frame(video_frame: np.ndarray,
                        budlight_bbox: np.ndarray,
                        roma_model,
                        preprocessing_conf: dict,
                        homography_ekf: Optional[HomographyEKF] = None,
                        expansion_factor: float = 0.1,
                        collect_debug_info: bool = False) -> tuple[np.ndarray, Optional[dict]]:
    """
    Replace Budlight logo with SPATEN in a video frame.

    Args:
        video_frame: Original video frame with Budlight logo
        budlight_bbox: Bounding box of Budlight logo [x1, y1, x2, y2]
        roma_model: Loaded ROMA model for matching
        preprocessing_conf: Preprocessing configuration
        homography_ekf: Optional EKF for homography stabilization
        expansion_factor: Factor to expand bounding box
        collect_debug_info: Whether to collect debug information

    Returns:
        Tuple of (result_frame, debug_info_dict)
    """
    debug_info: dict = {} if collect_debug_info else {}

    # Expand bounding box
    img_height, img_width = video_frame.shape[:2]
    x1, y1, x2, y2 = expand_box(budlight_bbox, expansion_factor, img_width, img_height)

    # Crop the physical logo from video frame
    physical_logo_cropped = video_frame[y1:y2, x1:x2]

    if collect_debug_info:
        debug_info['cropped_logo'] = physical_logo_cropped
        debug_info['reference_logo'] = budlight_downsampled
        debug_info['bbox'] = budlight_bbox

    # Step 1: Match physical logo with digital Budlight
    match_pred = run_matching_simple(
        roma_model,
        physical_logo_cropped,
        budlight_downsampled,  # Use the same Budlight logo used in overlay
        preprocessing_conf=preprocessing_conf,
        match_threshold=MATCHING_CONFIDENCE_THRESHOLD,
        extract_max_keypoints=MAX_KEYPOINTS,
        log_timing=False
    )

    # Step 2: Filter matches with RANSAC
    match_filtered = filter_matches_ransac(
        match_pred,
        ransac_reproj_threshold=RANSAC_THRESHOLD,
        log_timing=False
    )

    if len(match_filtered['H']) == 0 or len(match_filtered['mmkpts0']) < MIN_KP_FOR_HOMOGRAPHY:
        print(f"Failed to find sufficient matches ({len(match_filtered.get('mmkpts0', []))} < {MIN_KP_FOR_HOMOGRAPHY}), returning original frame")
        return video_frame, debug_info if collect_debug_info else None

    # Step 3: Map Budlight keypoints to SPATEN keypoints
    budlight_keypoints = match_filtered['mmkpts1']  # Keypoints in digital Budlight
    spaten_keypoints = map_budlight_to_spaten_coordinates(budlight_keypoints, center_x, center_y)

    # Step 4: Compute homography using physical logo keypoints -> SPATEN keypoints
    physical_keypoints = match_filtered['mmkpts0']  # Keypoints in physical logo (cropped space)

    # Transform physical keypoints to full frame coordinates
    physical_keypoints_full_frame = physical_keypoints.copy()
    physical_keypoints_full_frame[:, 0] += x1  # Add x offset
    physical_keypoints_full_frame[:, 1] += y1  # Add y offset

    if collect_debug_info:
        # Raw keypoints (all matches from MatchAnything)
        raw_physical_keypoints = match_pred['mkpts0'].copy()
        raw_physical_keypoints[:, 0] += x1  # Transform to full frame coordinates
        raw_physical_keypoints[:, 1] += y1

        debug_info['raw_keypoints'] = raw_physical_keypoints  # All raw keypoints in full frame
        debug_info['filtered_keypoints'] = physical_keypoints_full_frame  # RANSAC filtered keypoints in full frame

        # For match visualization (in cropped logo space)
        debug_info['raw_match_keypoints1'] = match_pred['mkpts0']  # Raw keypoints in cropped logo
        debug_info['raw_match_keypoints2'] = match_pred['mkpts1']  # Raw keypoints in reference logo
        debug_info['filtered_match_keypoints1'] = physical_keypoints  # RANSAC filtered keypoints in cropped logo
        debug_info['filtered_match_keypoints2'] = budlight_keypoints  # RANSAC filtered keypoints in reference logo

    # Compute homography: SPATEN -> full frame coordinates
    H_spaten, mask = cv2.findHomography(
        spaten_keypoints,  # Source: SPATEN coordinates
        physical_keypoints_full_frame,  # Target: full frame coordinates
        cv2.USAC_MAGSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        confidence=0.999,
        maxIters=10000
    )

    if H_spaten is None:
        print("Failed to compute SPATEN homography, returning original frame")
        return video_frame, debug_info if collect_debug_info else None

    # Step 5: Apply EKF stabilization to homography matrix
    if homography_ekf is not None:
        H_spaten_stabilized = homography_ekf.update(H_spaten)

        # Get EKF state information for debugging
        ekf_info = homography_ekf.get_state_info()
        if collect_debug_info:
            debug_info['ekf_info'] = ekf_info
        print(f"EKF stabilization - covariance trace: {ekf_info['covariance_trace']:.6f}")

        H_spaten = H_spaten_stabilized
    else:
        print("EKF stabilization disabled, using raw homography")

    # Step 6: Warp SPATEN logo to full frame size
    frame_h, frame_w = video_frame.shape[:2]

    if is_transparent:
        # For transparent logo, remove alpha channel for warping
        spaten_warped = cv2.warpPerspective(
            spaten_resized[:,:,:3],  # Remove alpha channel for warping
            H_spaten,
            (frame_w, frame_h)
        )
    else:
        # For non-transparent logo, use all channels
        spaten_warped = cv2.warpPerspective(
            spaten_resized,
            H_spaten,
            (frame_w, frame_h)
        )

    # Step 7: Create mask for entire frame and replace logo
    if is_transparent:
        # For transparent logo, use alpha channel for masking
        spaten_alpha_warped = cv2.warpPerspective(
            spaten_resized[:,:,3],  # Alpha channel only
            H_spaten,
            (frame_w, frame_h)
        )
        mask = spaten_alpha_warped > 0
    else:
        # For non-transparent logo, use grayscale intensity for masking
        spaten_gray = cv2.cvtColor(spaten_warped, cv2.COLOR_RGB2GRAY)
        mask = spaten_gray > 0

    mask_3d = np.stack([mask, mask, mask], axis=-1)

    # Replace logo in entire frame
    frame_with_logo = video_frame.copy()
    frame_with_logo[mask_3d] = spaten_warped[mask_3d]

    # Step 8: Get person masks and apply occlusion (bring people forward)
    person_mask = get_person_masks(
        video_frame,
        person_seg_model,
        confidence_threshold=0.5,
        log_timing=False
    )

    if collect_debug_info:
        debug_info['person_mask'] = person_mask

    # Apply person occlusion to bring people in front of logo
    result_frame = apply_person_occlusion(
        frame_with_logo,
        video_frame,  # Original frame
        person_mask,
        log_timing=False
    )

    print(f"Logo replacement successful with {len(match_filtered['mmkpts0'])} keypoints")

    if collect_debug_info:
        debug_info['stats'] = {
            'num_raw_matches': len(match_pred['mkpts0']),
            'num_filtered_matches': len(match_filtered['mmkpts0']),
            'ekf_info': ekf_info if homography_ekf is not None else None
        }

    return result_frame, debug_info if collect_debug_info else None

# Load models once (this should be done at startup)
print("Loading MatchAnything models...")

model_type = "roma"

if model_type == "roma":
    model, preprocessing_conf = load_matchanything_model("matchanything_roma", log_timing=True)
else:
    model, preprocessing_conf = load_matchanything_model("matchanything_eloftr", log_timing=True)

# Initialize EKF for homography stabilization
homography_ekf = None
if EKF_ENABLED:
    homography_ekf = HomographyEKF(
        dt=1.0,  # Assuming 1 frame time step
        process_noise_std=EKF_PROCESS_NOISE_STD,
        measurement_noise_std=EKF_MEASUREMENT_NOISE_STD,
        initial_covariance=EKF_INITIAL_COVARIANCE
    )
    print("Homography EKF initialized for stabilization")

print("Simple frame-by-frame processing ready!")

# Video processing parameters
video_path = "/home/sebastiangarcia/projects/swappr/data/poc/UFC317/BrazilPriEncode_swappr_317.ts"

# Test segments with good logo visibility
# 01:26:05-01:26:17 (good logo visibility)
# 01:53:12-01:53:15 (shorter test)
# 00:50:42-00:50:48 (stable scene)
# 01:55:12-01:55:35 (longer test)
# 00:50:42_00:50:48 (good conditions)
# 35:39-00:35:43
start_timestamp = "00:35:39"
end_timestamp = "00:36:05"

output_video_path = f"swapped_{model_type}_hybrid_lk_{start_timestamp}_{end_timestamp}.mp4"
yolo_model_path = "/home/sebastiangarcia/projects/swappr/models/poc/v2_budlight_logo_detection/weights/best.pt"
tracker_config_path = "/home/sebastiangarcia/projects/swappr/configs/trackers/bytetrack.yaml"

det_model_budlight = YOLO(yolo_model_path)
video_stream = cv2.VideoCapture(video_path)
video_fps: float = video_stream.get(cv2.CAP_PROP_FPS)

start_frame = parse_timestamp_to_frame(start_timestamp, video_fps)
end_frame = parse_timestamp_to_frame(end_timestamp, video_fps)

video_stream.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
current_frame_number = start_frame

# Get video properties for output
frame_width: int = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height: int = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    output_video_path,
    fourcc,
    video_fps,
    (frame_width, frame_height)
)

# Initialize debug visualizer
debug_visualizer = DebugVisualizer(enable_debug=DEBUG)

# Tracking state variables for hybrid MA + LK approach
tracking_state = {
    'prev_frame_gray': None,
    'prev_keypoints_physical': None,  # Physical logo keypoints (full frame coords)
    'prev_keypoints_spaten': None,    # Corresponding SPATEN logo keypoints
    'prev_bbox': None,                # Previous bounding box
    'frame_count_since_keyframe': 0,  # Frames since last MA keyframe
    'last_homography': None,          # Last successful homography for fallback
    'is_tracking': False              # Whether we're currently tracking vs using MA
}

print(f"Starting hybrid MatchAnything + LK tracking from frame {start_frame} to {end_frame}")
print(f"Video FPS: {video_fps}")
print(f"Keyframe interval: every {KEYFRAME_INTERVAL} frames")
print(f"Min tracking points: {MIN_TRACKING_POINTS}")
print(f"Max forward-backward error: {MAX_FB_ERROR}px")
print(f"Debug mode: {'ON' if DEBUG else 'OFF'}")

# Performance tracking
total_times = []
ma_calls = 0  # Count MatchAnything calls
lk_calls = 0  # Count LK tracking calls

while video_stream.isOpened():
    start_time_frame: float = time.time()

    ret, frame = video_stream.read()
    if not ret or frame is None:
        print("End of video stream or error reading frame.")
        break

    # Check if we've reached the end timestamp
    if end_frame is not None and current_frame_number >= end_frame:
        print(f"Reached end timestamp at frame {current_frame_number}")
        break

    # Convert BGR to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result_frame = frame_rgb.copy()
    debug_info: Optional[dict] = None

    # Detect Budlight logo
    results: list[Results] = det_model_budlight.track(
        frame, conf=0.8,
        persist=True,
        tracker=tracker_config_path,
        stream=False,
        verbose=False,
    )

    # Check if we have detection results
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy
        confidences = results[0].boxes.conf
    else:
        boxes = None
        confidences = None

    if boxes is not None and boxes.shape[0] > 0:
        # Handle multiple detections by selecting the one with highest confidence
        if hasattr(boxes, 'cpu') and callable(getattr(boxes, 'cpu', None)):
            # It's a tensor
            boxes_np = boxes.cpu().numpy().astype(int)
            confidences_np = confidences.cpu().numpy() if confidences is not None else None
        else:
            # It's already a numpy array
            boxes_np = np.array(boxes).astype(int)
            confidences_np = np.array(confidences) if confidences is not None else None

        # Select the bounding box with highest confidence
        if len(boxes_np.shape) == 2 and boxes_np.shape[0] > 1:
            # Multiple detections - select the one with highest confidence
            if confidences_np is not None:
                best_idx = np.argmax(confidences_np)
                budlight_bbox = boxes_np[best_idx]
                best_confidence = confidences_np[best_idx]
                bbox_w = budlight_bbox[2] - budlight_bbox[0]
                bbox_h = budlight_bbox[3] - budlight_bbox[1]
                print(f"Frame {current_frame_number}: Detected {boxes_np.shape[0]} Budlight logos, using highest confidence (conf={best_confidence:.3f}, {bbox_w}x{bbox_h})")
            else:
                # No confidence scores available, use the first detection
                budlight_bbox = boxes_np[0]
                bbox_w = budlight_bbox[2] - budlight_bbox[0]
                bbox_h = budlight_bbox[3] - budlight_bbox[1]
                print(f"Frame {current_frame_number}: Detected {boxes_np.shape[0]} Budlight logos, using first detection ({bbox_w}x{bbox_h})")
        else:
            # Single detection or squeeze to 1D
            budlight_bbox = boxes_np.squeeze()
            bbox_w = budlight_bbox[2] - budlight_bbox[0]
            bbox_h = budlight_bbox[3] - budlight_bbox[1]
            print(f"Frame {current_frame_number}: Detected Budlight logo ({bbox_w}x{bbox_h})")

        # Ensure budlight_bbox is 1D array with 4 elements
        if budlight_bbox.shape != (4,):
            print(f"Frame {current_frame_number}: Invalid bounding box shape {budlight_bbox.shape}, expected (4,)")
            print(f"Frame {current_frame_number}: ❌ Bounding box extraction failed")
            # Reset tracking state on invalid detection
            tracking_state['is_tracking'] = False
            tracking_state['frame_count_since_keyframe'] = 0
            current_frame_number += 1
            continue

        # Calculate bbox dimensions for validation (recalculate to ensure consistency)
        bbox_w = budlight_bbox[2] - budlight_bbox[0]  # x2 - x1
        bbox_h = budlight_bbox[3] - budlight_bbox[1]  # y2 - y1

        if bbox_w < MIN_BBOX_W:
            print(f"Frame {current_frame_number}: ❌ Bounding box width too small ({bbox_w} < {MIN_BBOX_W})")
            # Reset tracking state on invalid detection
            tracking_state['is_tracking'] = False
            tracking_state['frame_count_since_keyframe'] = 0
            current_frame_number += 1
            continue

        if bbox_h < MIN_BBOX_H:
            print(f"Frame {current_frame_number}: ❌ Bounding box height too small ({bbox_h} < {MIN_BBOX_H})")
            # Reset tracking state on invalid detection
            tracking_state['is_tracking'] = False
            tracking_state['frame_count_since_keyframe'] = 0
            current_frame_number += 1
            continue

        # Decide whether to use MatchAnything or LK tracking
        use_matchanything = False
        reseed_reason = "first_frame"

        if not tracking_state['is_tracking'] or tracking_state['prev_frame_gray'] is None:
            # First frame or tracking was reset
            use_matchanything = True
            reseed_reason = "first_frame_or_reset"
        elif tracking_state['prev_keypoints_physical'] is not None and len(tracking_state['prev_keypoints_physical']) > 0:
            # We have previous keypoints, try LK tracking first
            tracked_keypoints, good_mask, tracking_stats = track_keypoints_lk(
                tracking_state['prev_frame_gray'],
                frame_gray,
                tracking_state['prev_keypoints_physical'],
                log_timing=False
            )

            # Check if we should reseed with MatchAnything
            should_reseed, reseed_reason = should_reseed_keyframe(
                tracking_stats, tracking_state['frame_count_since_keyframe']
            )

            if should_reseed:
                use_matchanything = True
                print(f"Frame {current_frame_number}: 🔄 Reseeding with MatchAnything - {reseed_reason}")
            else:
                # Continue with LK tracking
                use_matchanything = False
                # Update tracking state
                tracking_state['prev_keypoints_physical'] = tracked_keypoints
                # Update corresponding SPATEN keypoints (maintain the same valid points)
                if tracking_state['prev_keypoints_spaten'] is not None:
                    tracking_state['prev_keypoints_spaten'] = tracking_state['prev_keypoints_spaten'][good_mask]

                lk_calls += 1
                print(f"Frame {current_frame_number}: 🎯 LK tracking - {tracking_stats['tracked_points']} points, {tracking_stats['survival_rate']:.1%} survival")
        else:
            # No previous keypoints available
            use_matchanything = True
            reseed_reason = "no_previous_keypoints"

        if use_matchanything:
            # Use full MatchAnything pipeline
            result_frame, debug_info = replace_logo_in_frame(
                frame_rgb,
                budlight_bbox,
                model,
                preprocessing_conf,
                homography_ekf,
                expansion_factor=0.1,
                collect_debug_info=DEBUG
            )

            # Extract keypoints for future tracking
            if debug_info and 'filtered_match_keypoints1' in debug_info and 'filtered_match_keypoints2' in debug_info:
                # Get keypoints in full frame coordinates (already transformed in replace_logo_in_frame)
                if 'filtered_keypoints' in debug_info:
                    physical_keypoints_full = debug_info['filtered_keypoints']
                    # Filter keypoints for better LK tracking
                    physical_keypoints_filtered = filter_keypoints_for_tracking(physical_keypoints_full, max_points=800)

                    # Get corresponding SPATEN keypoints
                    budlight_keypoints = debug_info['filtered_match_keypoints2']  # In Budlight logo space
                    spaten_keypoints = map_budlight_to_spaten_coordinates(budlight_keypoints, center_x, center_y)

                    # Filter SPATEN keypoints to match the filtered physical keypoints
                    if len(spaten_keypoints) > len(physical_keypoints_filtered):
                        spaten_keypoints = spaten_keypoints[:len(physical_keypoints_filtered)]

                    # Update tracking state
                    tracking_state['prev_keypoints_physical'] = physical_keypoints_filtered
                    tracking_state['prev_keypoints_spaten'] = spaten_keypoints
                    tracking_state['is_tracking'] = True
                    tracking_state['frame_count_since_keyframe'] = 0

                    print(f"Frame {current_frame_number}: 🎯 MA keyframe - extracted {len(physical_keypoints_filtered)} points for tracking")
                else:
                    tracking_state['is_tracking'] = False
            else:
                tracking_state['is_tracking'] = False

            ma_calls += 1
            print(f"Frame {current_frame_number}: 🔍 MatchAnything - {reseed_reason}")
        else:
            # Use LK tracking result to compute homography and replace logo
            if len(tracking_state['prev_keypoints_physical']) >= MIN_KP_FOR_HOMOGRAPHY and len(tracking_state['prev_keypoints_spaten']) >= MIN_KP_FOR_HOMOGRAPHY:
                # Compute homography using tracked keypoints
                H_spaten, mask = cv2.findHomography(
                    tracking_state['prev_keypoints_spaten'],  # Source: SPATEN coordinates
                    tracking_state['prev_keypoints_physical'],  # Target: full frame coordinates
                    cv2.USAC_MAGSAC,
                    ransacReprojThreshold=RANSAC_THRESHOLD,
                    confidence=0.999,
                    maxIters=10000
                )

                if H_spaten is not None:
                    # Apply EKF stabilization
                    if homography_ekf is not None:
                        H_spaten = homography_ekf.update(H_spaten)

                    # Store for potential fallback
                    tracking_state['last_homography'] = H_spaten

                    # Warp and replace logo using the computed homography
                    frame_h, frame_w = frame_rgb.shape[:2]

                    if is_transparent:
                        spaten_warped = cv2.warpPerspective(
                            spaten_resized[:,:,:3],  # Remove alpha channel for warping
                            H_spaten,
                            (frame_w, frame_h)
                        )
                        spaten_alpha_warped = cv2.warpPerspective(
                            spaten_resized[:,:,3],  # Alpha channel only
                            H_spaten,
                            (frame_w, frame_h)
                        )
                        mask = spaten_alpha_warped > 0
                    else:
                        spaten_warped = cv2.warpPerspective(
                            spaten_resized,
                            H_spaten,
                            (frame_w, frame_h)
                        )
                        spaten_gray = cv2.cvtColor(spaten_warped, cv2.COLOR_RGB2GRAY)
                        mask = spaten_gray > 0

                    mask_3d = np.stack([mask, mask, mask], axis=-1)

                    # Replace logo in frame
                    frame_with_logo = frame_rgb.copy()
                    frame_with_logo[mask_3d] = spaten_warped[mask_3d]

                    # Apply person occlusion
                    person_mask = get_person_masks(frame_rgb, person_seg_model, confidence_threshold=0.5, log_timing=False)
                    result_frame = apply_person_occlusion(frame_with_logo, frame_rgb, person_mask, log_timing=False)

                    if DEBUG:
                        # Prepare debug info for LK tracking frames
                        debug_info = {
                            'bbox': budlight_bbox,
                            'filtered_keypoints': tracking_state['prev_keypoints_physical'],
                            'person_mask': person_mask,
                            'stats': tracking_stats
                        }
                else:
                    print(f"Frame {current_frame_number}: ❌ LK homography computation failed")
                    result_frame = frame_rgb  # Use original frame
            else:
                print(f"Frame {current_frame_number}: ❌ Not enough LK points for homography")
                result_frame = frame_rgb  # Use original frame

        # Update tracking state
        tracking_state['prev_frame_gray'] = frame_gray.copy()
        tracking_state['prev_bbox'] = budlight_bbox
        tracking_state['frame_count_since_keyframe'] += 1

        print(f"Frame {current_frame_number}: ✅ Logo processing completed")
    else:
        print(f"Frame {current_frame_number}: No Budlight logo detected")
        # Reset tracking state when no logo is detected
        tracking_state['is_tracking'] = False
        tracking_state['frame_count_since_keyframe'] = 0

    # Convert result frame back to BGR for video writing
    result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
    out.write(result_frame_bgr)

    # Performance metrics
    frame_time = time.time() - start_time_frame
    total_times.append(frame_time)
    fps = 1.0 / frame_time if frame_time > 0 else 0

    # Add processing time to debug info
    if debug_info is not None and isinstance(debug_info, dict) and 'stats' in debug_info and debug_info['stats'] is not None:
        debug_info['stats']['processing_time'] = frame_time

    # Update debug display
    debug_visualizer.update_display(
        current_frame_number,
        frame_rgb,
        result_frame,
        debug_info
    )

    print(f"Frame {current_frame_number}: Processing time: {frame_time:.3f}s, FPS: {fps:.1f}")

    current_frame_number += 1

video_stream.release()
out.release()
debug_visualizer.close()

print("\n" + "="*50)
print("HYBRID MATCHANYTHING + LK TRACKING PERFORMANCE STATISTICS")
print("="*50)

if total_times:
    avg_frame_time = np.mean(total_times)
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    total_frames = len(total_times)

    print(f"Average frame time: {avg_frame_time:.3f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Total frames processed: {total_frames}")
    print(f"MatchAnything calls: {ma_calls} ({ma_calls/total_frames:.1%} of frames)")
    print(f"LK tracking calls: {lk_calls} ({lk_calls/total_frames:.1%} of frames)")
    print(f"Efficiency improvement: {(total_frames-ma_calls)/total_frames:.1%} frames used lightweight LK tracking")
    print(f"Expected speedup vs full MA: ~{total_frames/max(ma_calls,1):.1f}x (theoretical)")

print("Hybrid MatchAnything + LK tracking video processing completed!")
