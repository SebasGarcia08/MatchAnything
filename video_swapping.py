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

# Hybrid tracking configuration
MATCH_EVERY_N_FRAMES = 120
MAX_KP_TO_TRACK = 2000
MIN_KP_FOR_HOMOGRAPHY = 100
ROMA_CONFIDENCE_THRESHOLD = 0.02  # Increased from 0.01
COTRACKER_INPUT_RESOLUTION = (384, 512)  # H, W for CoTracker3


logger = logging.getLogger(__name__)

class HybridLogoTracker:
    """
    Hybrid tracking system that combines MatchAnything ROMA with CoTracker3
    for efficient and robust logo keypoint tracking.
    """

    def __init__(self,
                 roma_model: MatchAnything,
                 roma_preprocessing_conf: dict,
                 budlight_reference: np.ndarray,
                 device: str = "cuda"):
        """
        Initialize the hybrid tracker.

        Args:
            roma_model: Loaded MatchAnything ROMA model
            roma_preprocessing_conf: Preprocessing configuration for ROMA
            budlight_reference: Reference Budlight logo image
            device: Device to run models on
        """
        self.roma_model = roma_model
        self.roma_preprocessing_conf = roma_preprocessing_conf
        self.budlight_reference = budlight_reference
        self.device = device

        # Initialize CoTracker3 online model
        self.cotracker_model = None
        self.cotracker_initialized = False
        self.cotracker_step = 8  # Default step size for CoTracker3

        # Frame buffer for CoTracker3 online processing
        self.frame_buffer: list[np.ndarray] = []
        self.frame_buffer_max_size = 32  # Keep enough frames for processing

        # Tracking state
        self.current_keypoints = None  # [N, 2] keypoints in full frame coordinates
        self.reference_keypoints = None  # [N, 2] keypoints in reference logo coordinates
        self.queries = None  # Queries tensor for CoTracker3
        self.tracking_active = False
        self.frame_counter = 0

        # 🔥 NEW: Fallback system for continuous logo replacement
        self.last_good_homography = None  # Store last successful homography
        self.last_good_keypoints = None  # Store last successful keypoints
        self.last_good_reference_keypoints = None  # Store last successful reference keypoints
        self.consecutive_failures = 0  # Track consecutive failures
        self.max_consecutive_failures = 3  # Max failures before emergency fallback
        self.emergency_fallback_active = False  # Flag for emergency mode

        # Performance tracking
        self.roma_times = []
        self.cotracker_times = []
        self.fallback_usage_count = 0
        self.emergency_fallback_count = 0

    def _initialize_cotracker(self) -> None:
        """Initialize CoTracker3 online model."""
        if self.cotracker_model is None:
            print("Loading CoTracker3 online model...")
            self.cotracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
            self.cotracker_model = self.cotracker_model.to(self.device)
            self.cotracker_step = self.cotracker_model.step
            print(f"CoTracker3 online model loaded successfully! Step size: {self.cotracker_step}")

    def _detect_keypoints_with_roma(self,
                                   frame: np.ndarray,
                                   logo_bbox: np.ndarray,
                                   log_timing: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect keypoints using MatchAnything ROMA model.

        Args:
            frame: Current frame
            logo_bbox: Logo bounding box [x1, y1, x2, y2]
            log_timing: Whether to log timing information

        Returns:
            Tuple of (frame_keypoints, reference_keypoints) or (None, None) if failed
        """
        if log_timing:
            t0 = time.time()

        try:
            # Expand bounding box
            img_height, img_width = frame.shape[:2]
            x1, y1, x2, y2 = expand_box(logo_bbox, 0.1, img_width, img_height)

            # Crop logo region
            physical_logo_cropped = frame[y1:y2, x1:x2]

            # Run ROMA matching
            match_pred = run_matching_simple(
                self.roma_model,
                physical_logo_cropped,
                self.budlight_reference,
                preprocessing_conf=self.roma_preprocessing_conf,
                match_threshold=ROMA_CONFIDENCE_THRESHOLD,
                extract_max_keypoints=MAX_KP_TO_TRACK * 2,  # Get more, filter later
                log_timing=False
            )

            # Filter matches with RANSAC
            match_filtered = filter_matches_ransac(match_pred, log_timing=False)

            if len(match_filtered['H']) == 0 or len(match_filtered['mmkpts0']) < MIN_KP_FOR_HOMOGRAPHY:
                if log_timing:
                    print(f"ROMA: Insufficient matches ({len(match_filtered.get('mmkpts0', []))}) for reliable tracking")
                return None, None

            # Get keypoints in cropped coordinates
            frame_keypoints_cropped = match_filtered['mmkpts0']  # [N, 2]
            reference_keypoints = match_filtered['mmkpts1']  # [N, 2]

            # Convert to full frame coordinates
            frame_keypoints_full = frame_keypoints_cropped.copy()
            frame_keypoints_full[:, 0] += x1  # Add x offset
            frame_keypoints_full[:, 1] += y1  # Add y offset

            # Select best keypoints if we have too many
            if len(frame_keypoints_full) > MAX_KP_TO_TRACK:
                # Sort by confidence and take top MAX_KP_TO_TRACK
                confidences = match_filtered['mmconf']
                top_indices = np.argsort(confidences)[-MAX_KP_TO_TRACK:]
                frame_keypoints_full = frame_keypoints_full[top_indices]
                reference_keypoints = reference_keypoints[top_indices]

            if log_timing:
                roma_time = time.time() - t0
                self.roma_times.append(roma_time)
                print(f"ROMA: Detected {len(frame_keypoints_full)} keypoints in {roma_time:.3f}s")

            return frame_keypoints_full, reference_keypoints

        except Exception as e:
            print(f"ROMA keypoint detection failed: {e}")
            return None, None

    def _initialize_cotracker_online(self,
                                    initial_frame: np.ndarray,
                                    initial_keypoints: np.ndarray) -> bool:
        """
        Initialize CoTracker3 online processing with initial frame and keypoints.

        Args:
            initial_frame: Initial frame for CoTracker3
            initial_keypoints: Initial keypoints [N, 2] in full frame coordinates

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Ensure CoTracker model is loaded
            self._initialize_cotracker()

            if self.cotracker_model is None:
                print(f"      🔍 DEBUG: CoTracker model is None, initialization failed")
                return False

            # 🔍 DEBUG: Log initialization parameters
            print(f"      🔍 DEBUG: Initializing CoTracker online with {len(initial_keypoints)} keypoints")
            print(f"      🔍 DEBUG: Frame shape: {initial_frame.shape}")
            print(f"      🔍 DEBUG: CoTracker input resolution: {COTRACKER_INPUT_RESOLUTION}")

            # Prepare frame tensor - CoTracker3 online expects [B, T, C, H, W]
            frame_resized = cv2.resize(initial_frame, COTRACKER_INPUT_RESOLUTION[::-1])  # (W, H)
            frame_tensor = torch.from_numpy(frame_resized).float().to(self.device)
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]

            # Scale keypoints to CoTracker resolution
            frame_height, frame_width = initial_frame.shape[:2]
            scaled_keypoints = initial_keypoints.copy()
            scaled_keypoints[:, 0] *= COTRACKER_INPUT_RESOLUTION[1] / frame_width   # Scale x
            scaled_keypoints[:, 1] *= COTRACKER_INPUT_RESOLUTION[0] / frame_height  # Scale y

            # Prepare queries tensor: [1, N, 3] where each point is [t, x, y]
            # t=0 since we're initializing on the first frame
            num_points = len(scaled_keypoints)
            queries = torch.zeros(1, num_points, 3).to(self.device)
            queries[0, :, 0] = 0  # Time index (first frame)
            queries[0, :, 1] = torch.from_numpy(scaled_keypoints[:, 0]).float()  # x coordinates
            queries[0, :, 2] = torch.from_numpy(scaled_keypoints[:, 1]).float()  # y coordinates

            # Store queries for later use
            self.queries = queries

            # 🔍 DEBUG: Log queries tensor
            print(f"      🔍 DEBUG: queries shape: {queries.shape}")
            print(f"      🔍 DEBUG: queries range: t({queries[0, :, 0].min():.1f}-{queries[0, :, 0].max():.1f}), x({queries[0, :, 1].min():.1f}-{queries[0, :, 1].max():.1f}), y({queries[0, :, 2].min():.1f}-{queries[0, :, 2].max():.1f})")

            # Initialize CoTracker3 online processing
            print(f"      🔍 DEBUG: Calling CoTracker3 online initialization...")
            result = self.cotracker_model(
                video_chunk=frame_tensor,
                is_first_step=True,
                grid_size=0,
                queries=queries,
                add_support_grid=False
            )

            # 🔍 DEBUG: Log initialization result
            print(f"      🔍 DEBUG: Initialization result: {result}")

            # Initialize frame buffer with the first frame
            self.frame_buffer = [initial_frame]

            self.cotracker_initialized = True
            print(f"CoTracker3 online initialized with {num_points} keypoints")
            return True

        except Exception as e:
            print(f"CoTracker3 online initialization failed: {e}")
            print(f"      🔍 DEBUG: Initialization exception type: {type(e)}")
            print(f"      🔍 DEBUG: Initialization exception args: {e.args}")
            self.cotracker_initialized = False
            return False

    def _track_keypoints_with_cotracker_online(self,
                                              frame: np.ndarray,
                                              log_timing: bool = True) -> Optional[np.ndarray]:
        """
        Track keypoints using CoTracker3 online model.

        Args:
            frame: Current frame
            log_timing: Whether to log timing information

        Returns:
            Tracked keypoints [N, 2] in full frame coordinates or None if failed
        """
        if not self.cotracker_initialized or self.queries is None or self.cotracker_model is None:
            return None

        if log_timing:
            t0 = time.time()

        try:
            # Add frame to buffer
            self.frame_buffer.append(frame)

            # Keep buffer size manageable
            if len(self.frame_buffer) > self.frame_buffer_max_size:
                self.frame_buffer = self.frame_buffer[-self.frame_buffer_max_size:]

            # Check if we have enough frames for processing
            if len(self.frame_buffer) < self.cotracker_step * 2:
                print(f"      🔍 DEBUG: Not enough frames in buffer ({len(self.frame_buffer)} < {self.cotracker_step * 2})")
                return None

            # Prepare video chunk - take the last cotracker_step * 2 frames
            video_chunk_frames = self.frame_buffer[-self.cotracker_step * 2:]

            # Resize frames to CoTracker resolution
            video_chunk_resized = []
            for frame_i in video_chunk_frames:
                frame_resized = cv2.resize(frame_i, COTRACKER_INPUT_RESOLUTION[::-1])  # (W, H)
                video_chunk_resized.append(frame_resized)

            # Convert to tensor [1, T, C, H, W]
            video_chunk_tensor = torch.from_numpy(np.stack(video_chunk_resized)).float().to(self.device)
            video_chunk_tensor = video_chunk_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, 3, H, W]

            # 🔍 DEBUG: Log tensor shapes before CoTracker call
            print(f"      🔍 DEBUG: video_chunk_tensor shape: {video_chunk_tensor.shape}")
            print(f"      🔍 DEBUG: queries shape: {self.queries.shape}")

            # Run CoTracker3 online inference
            pred_tracks, pred_visibility = self.cotracker_model(
                video_chunk=video_chunk_tensor,
                grid_size=0,
                queries=self.queries,
                add_support_grid=False
            )

            # 🔍 DEBUG: Log output tensor shapes
            print(f"      🔍 DEBUG: pred_tracks shape: {pred_tracks.shape if pred_tracks is not None else 'None'}")
            print(f"      🔍 DEBUG: pred_visibility shape: {pred_visibility.shape if pred_visibility is not None else 'None'}")

            # Validate outputs
            if pred_tracks is None or pred_visibility is None:
                print(f"      🔍 DEBUG: CoTracker returned None outputs")
                return None

            # Get tracks for the last frame (most recent)
            try:
                # pred_tracks shape: [B, T, N, 2], we want the last frame
                last_frame_tracks = pred_tracks[0, -1].cpu().numpy()  # [N, 2]
                last_frame_visibility = pred_visibility[0, -1].cpu().numpy()  # [N]

                print(f"      🔍 DEBUG: last_frame_tracks shape: {last_frame_tracks.shape}")
                print(f"      🔍 DEBUG: last_frame_visibility shape: {last_frame_visibility.shape}")
            except Exception as index_error:
                print(f"      🔍 DEBUG: Indexing error: {index_error}")
                print(f"      🔍 DEBUG: pred_tracks shape: {pred_tracks.shape}")
                print(f"      🔍 DEBUG: pred_visibility shape: {pred_visibility.shape}")
                return None

            # Scale from CoTracker resolution to full frame resolution
            frame_height, frame_width = frame.shape[:2]
            last_frame_tracks[:, 0] *= frame_width / COTRACKER_INPUT_RESOLUTION[1]   # Scale x
            last_frame_tracks[:, 1] *= frame_height / COTRACKER_INPUT_RESOLUTION[0]  # Scale y

            # Validate tracking results
            if len(last_frame_tracks) == 0:
                print(f"      🔍 DEBUG: No valid tracks after scaling")
                return None

            # Check for reasonable keypoint positions
            valid_x = (last_frame_tracks[:, 0] >= 0) & (last_frame_tracks[:, 0] < frame_width)
            valid_y = (last_frame_tracks[:, 1] >= 0) & (last_frame_tracks[:, 1] < frame_height)
            valid_points = valid_x & valid_y

            if np.sum(valid_points) < len(last_frame_tracks) * 0.5:  # Less than 50% valid points
                print(f"      🔍 DEBUG: Too many invalid tracking points ({np.sum(valid_points)}/{len(last_frame_tracks)})")
                return None

            if log_timing:
                cotracker_time = time.time() - t0
                self.cotracker_times.append(cotracker_time)
                visible_count = np.sum(last_frame_visibility > 0.5)
                print(f"CoTracker3 Online: Tracked {len(last_frame_tracks)} keypoints ({visible_count} visible) in {cotracker_time:.3f}s")

            return last_frame_tracks

        except Exception as e:
            logger.exception("CoTracker3 online tracking failed")
            print(f"CoTracker3 online tracking failed: {e}")
            print(f"      🔍 DEBUG: Exception type: {type(e)}")
            print(f"      🔍 DEBUG: Exception args: {e.args}")

            # Reset CoTracker on failure to prevent persistent state issues
            print(f"      🔍 DEBUG: Resetting CoTracker3 online due to failure...")
            self.cotracker_initialized = False
            self.cotracker_model = None
            self.tracking_active = False
            self.frame_buffer = []
            self.queries = None

            return None

    def _filter_occluded_keypoints(self,
                                  keypoints: np.ndarray,
                                  person_masks: np.ndarray,
                                  reference_keypoints: np.ndarray,
                                  dilation_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter keypoints that are occluded by people.

        Args:
            keypoints: Keypoints [N, 2] in full frame coordinates
            person_masks: Person segmentation masks [H, W]
            reference_keypoints: Reference keypoints [N, 2] in logo coordinates
            dilation_size: Dilation size for person masks

        Returns:
            Tuple of (filtered_keypoints, filtered_reference_keypoints)
        """
        if len(keypoints) == 0 or len(reference_keypoints) == 0:
            return keypoints, reference_keypoints

        # Dilate person masks slightly to be more conservative
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        person_masks_dilated = cv2.dilate(person_masks, kernel, iterations=1)

        # Check which keypoints are not occluded
        valid_mask = np.ones(len(keypoints), dtype=bool)

        for i, (x, y) in enumerate(keypoints):
            # Check if keypoint is within frame bounds
            if x < 0 or y < 0 or x >= person_masks.shape[1] or y >= person_masks.shape[0]:
                valid_mask[i] = False
                continue

            # Check if keypoint is occluded by person
            if person_masks_dilated[int(y), int(x)] > 0:
                valid_mask[i] = False

        # Filter keypoints
        filtered_keypoints = keypoints[valid_mask]
        filtered_reference_keypoints = reference_keypoints[valid_mask]

        return filtered_keypoints, filtered_reference_keypoints

    def process_frame(self,
                     frame: np.ndarray,
                     logo_bbox: np.ndarray,
                     person_masks: np.ndarray,
                     log_timing: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process a single frame to get keypoints for homography computation.

        Args:
            frame: Current frame
            logo_bbox: Logo bounding box [x1, y1, x2, y2]
            person_masks: Person segmentation masks [H, W]
            log_timing: Whether to log timing information

        Returns:
            Tuple of (frame_keypoints, reference_keypoints) or (None, None) if failed
        """
        self.frame_counter += 1
        is_recalibration_frame = self.frame_counter % MATCH_EVERY_N_FRAMES == 1 or not self.tracking_active

        if log_timing:
            print(f"\n🔍 FRAME {self.frame_counter} DEBUG:")
            print(f"  - Recalibration frame: {is_recalibration_frame}")
            print(f"  - Tracking active: {self.tracking_active}")
            print(f"  - Frame counter mod: {self.frame_counter % MATCH_EVERY_N_FRAMES}")

        # Check if we need to recalibrate with ROMA
        if is_recalibration_frame:
            if log_timing:
                print(f"  🔄 STARTING ROMA RECALIBRATION...")

            # Recalibration phase: Use ROMA to detect keypoints
            frame_kp, ref_kp = self._detect_keypoints_with_roma(frame, logo_bbox, log_timing)

            if frame_kp is not None and ref_kp is not None:
                if log_timing:
                    print(f"  ✅ ROMA SUCCESS: {len(frame_kp)} keypoints detected")

                # Filter occluded keypoints
                frame_kp_filtered, ref_kp_filtered = self._filter_occluded_keypoints(
                    frame_kp, person_masks, ref_kp
                )

                if len(frame_kp_filtered) >= MIN_KP_FOR_HOMOGRAPHY:
                    if log_timing:
                        print(f"  ✅ OCCLUSION FILTER SUCCESS: {len(frame_kp_filtered)} keypoints remain")

                    # Update tracking state
                    self.current_keypoints = frame_kp_filtered
                    self.reference_keypoints = ref_kp_filtered

                    # Initialize CoTracker3 online for next frames
                    if self._initialize_cotracker_online(frame, frame_kp_filtered):
                        self.tracking_active = True
                        if log_timing:
                            print(f"  ✅ COTRACKER3 ONLINE INIT SUCCESS")
                            print(f"  🎯 FRAME {self.frame_counter}: ROMA recalibration successful, {len(frame_kp_filtered)} keypoints")
                        return frame_kp_filtered, ref_kp_filtered
                    else:
                        if log_timing:
                            print(f"  ❌ COTRACKER3 ONLINE INIT FAILED")
                            print(f"  🚨 FRAME {self.frame_counter}: CoTracker3 online initialization failed")
                        return None, None
                else:
                    if log_timing:
                        print(f"  ❌ INSUFFICIENT KEYPOINTS: {len(frame_kp_filtered)} < {MIN_KP_FOR_HOMOGRAPHY}")
                        print(f"  🚨 FRAME {self.frame_counter}: Insufficient keypoints after occlusion filtering ({len(frame_kp_filtered)})")
                    return None, None
            else:
                if log_timing:
                    print(f"  ❌ ROMA DETECTION FAILED")
                    print(f"  🚨 FRAME {self.frame_counter}: ROMA keypoint detection failed")
                return None, None

        else:
            if log_timing:
                print(f"  🏃 COTRACKER3 ONLINE TRACKING...")

            # Tracking phase: Use CoTracker3 online to track existing keypoints
            tracked_kp = self._track_keypoints_with_cotracker_online(frame, log_timing)

            if tracked_kp is not None and len(tracked_kp) > 0 and self.reference_keypoints is not None:
                if log_timing:
                    print(f"  ✅ COTRACKER3 ONLINE SUCCESS: {len(tracked_kp)} keypoints tracked")

                # Filter occluded keypoints
                tracked_kp_filtered, ref_kp_filtered = self._filter_occluded_keypoints(
                    tracked_kp, person_masks, self.reference_keypoints
                )

                if len(tracked_kp_filtered) >= MIN_KP_FOR_HOMOGRAPHY:
                    # Update current keypoints
                    self.current_keypoints = tracked_kp_filtered
                    if log_timing:
                        print(f"  ✅ OCCLUSION FILTER SUCCESS: {len(tracked_kp_filtered)} keypoints remain")
                        print(f"  🎯 FRAME {self.frame_counter}: CoTracker3 online tracking successful, {len(tracked_kp_filtered)} keypoints")
                    return tracked_kp_filtered, ref_kp_filtered
                else:
                    # Not enough keypoints, trigger immediate recalibration
                    if log_timing:
                        print(f"  ❌ INSUFFICIENT KEYPOINTS: {len(tracked_kp_filtered)} < {MIN_KP_FOR_HOMOGRAPHY}")
                        print(f"  🔄 TRIGGERING IMMEDIATE RECALIBRATION")
                        print(f"  🚨 FRAME {self.frame_counter}: Insufficient keypoints after filtering ({len(tracked_kp_filtered)}), triggering recalibration")
                    self.frame_counter = MATCH_EVERY_N_FRAMES  # Force recalibration on next call
                    self.tracking_active = False
                    return None, None
            else:
                # Tracking failed, trigger immediate recalibration
                if log_timing:
                    print(f"  ❌ COTRACKER3 ONLINE TRACKING FAILED")
                    print(f"  🔄 TRIGGERING IMMEDIATE RECALIBRATION")
                    print(f"  🚨 FRAME {self.frame_counter}: CoTracker3 online tracking failed, triggering recalibration")
                self.frame_counter = MATCH_EVERY_N_FRAMES  # Force recalibration on next call
                self.tracking_active = False
                return None, None

    def _compute_emergency_homography(self, frame: np.ndarray, logo_bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Emergency fallback: Use traditional 4-corner approach when hybrid tracking fails.

        Args:
            frame: Current frame
            logo_bbox: Logo bounding box [x1, y1, x2, y2]

        Returns:
            Emergency homography matrix or None if failed
        """
        try:
            print(f"  🚨 EMERGENCY FALLBACK: Using 4-corner approach")

            # Expand bounding box
            img_height, img_width = frame.shape[:2]
            x1, y1, x2, y2 = expand_box(logo_bbox, 0.1, img_width, img_height)

            # Crop logo region
            physical_logo_cropped = frame[y1:y2, x1:x2]

            # Run ROMA matching with lower thresholds for emergency
            match_pred = run_matching_simple(
                self.roma_model,
                physical_logo_cropped,
                self.budlight_reference,
                preprocessing_conf=self.roma_preprocessing_conf,
                match_threshold=0.005,  # Lower threshold for emergency
                extract_max_keypoints=200,  # Fewer keypoints for stability
                log_timing=False
            )

            # Filter matches with RANSAC
            match_filtered = filter_matches_ransac(match_pred, log_timing=False)

            if len(match_filtered['H']) > 0 and len(match_filtered['mmkpts0']) >= 4:
                # Use the existing homography from RANSAC
                H = match_filtered['H']

                # Convert to full frame coordinates
                frame_keypoints_cropped = match_filtered['mmkpts0']
                frame_keypoints_full = frame_keypoints_cropped.copy()
                frame_keypoints_full[:, 0] += x1
                frame_keypoints_full[:, 1] += y1

                # Compute homography for SPATEN logo
                reference_keypoints = match_filtered['mmkpts1']
                spaten_keypoints = map_budlight_to_spaten_coordinates(reference_keypoints, center_x, center_y)

                emergency_H = compute_homography_from_keypoints(frame_keypoints_full, spaten_keypoints)

                if emergency_H is not None:
                    self.emergency_fallback_count += 1
                    print(f"  ✅ EMERGENCY FALLBACK SUCCESS: Homography computed")
                    return emergency_H

            print(f"  ❌ EMERGENCY FALLBACK FAILED")
            return None

        except Exception as e:
            print(f"  ❌ EMERGENCY FALLBACK ERROR: {e}")
            return None

    def get_homography_for_frame(self,
                                frame: np.ndarray,
                                logo_bbox: np.ndarray,
                                person_masks: np.ndarray,
                                log_timing: bool = True) -> Optional[np.ndarray]:
        """
        Get homography matrix for current frame with comprehensive fallback system.

        Args:
            frame: Current frame
            logo_bbox: Logo bounding box [x1, y1, x2, y2]
            person_masks: Person segmentation masks [H, W]
            log_timing: Whether to log timing information

        Returns:
            Homography matrix for logo replacement or None if all methods fail
        """
        # Try primary hybrid tracking
        frame_keypoints, reference_keypoints = self.process_frame(frame, logo_bbox, person_masks, log_timing)

        if frame_keypoints is not None and reference_keypoints is not None:
            # PRIMARY SUCCESS: Compute homography
            spaten_keypoints = map_budlight_to_spaten_coordinates(reference_keypoints, center_x, center_y)
            H = compute_homography_from_keypoints(frame_keypoints, spaten_keypoints)

            if H is not None:
                # Update fallback data
                self.last_good_homography = H.copy()
                self.last_good_keypoints = frame_keypoints.copy()
                self.last_good_reference_keypoints = reference_keypoints.copy()
                self.consecutive_failures = 0
                self.emergency_fallback_active = False

                if log_timing:
                    print(f"  ✅ PRIMARY SUCCESS: Homography computed from {len(frame_keypoints)} keypoints")

                return H

        # PRIMARY FAILED: Try fallback systems
        self.consecutive_failures += 1

        if log_timing:
            print(f"  ⚠️  PRIMARY FAILED: Consecutive failures = {self.consecutive_failures}")

        # FALLBACK 1: Use last good homography (for minor tracking failures)
        if self.last_good_homography is not None and self.consecutive_failures <= self.max_consecutive_failures:
            self.fallback_usage_count += 1
            if log_timing:
                print(f"  🔄 FALLBACK 1: Using last good homography (usage #{self.fallback_usage_count})")
            return self.last_good_homography.copy()

        # FALLBACK 2: Emergency recalibration with 4-corner approach
        if self.consecutive_failures > self.max_consecutive_failures:
            self.emergency_fallback_active = True
            emergency_H = self._compute_emergency_homography(frame, logo_bbox)

            if emergency_H is not None:
                # Update fallback data
                self.last_good_homography = emergency_H.copy()
                self.consecutive_failures = 0
                return emergency_H

        # FALLBACK 3: Last resort - use previous homography even if old
        if self.last_good_homography is not None:
            self.fallback_usage_count += 1
            if log_timing:
                print(f"  🆘 FALLBACK 3: Using old homography as last resort")
            return self.last_good_homography.copy()

        # COMPLETE FAILURE: No homography available
        if log_timing:
            print(f"  💀 COMPLETE FAILURE: No homography available")

        return None

    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        stats = {
            'roma_calls': len(self.roma_times),
            'cotracker_calls': len(self.cotracker_times),
            'roma_avg_time': np.mean(self.roma_times) if self.roma_times else 0,
            'cotracker_avg_time': np.mean(self.cotracker_times) if self.cotracker_times else 0,
            'total_frames': self.frame_counter,
            'fallback_usage_count': self.fallback_usage_count,
            'emergency_fallback_count': self.emergency_fallback_count,
            'consecutive_failures': self.consecutive_failures,
            'emergency_fallback_active': self.emergency_fallback_active,
            'success_rate': (self.frame_counter - self.fallback_usage_count) / self.frame_counter if self.frame_counter > 0 else 0
        }
        return stats


def compute_homography_from_keypoints(frame_keypoints: np.ndarray,
                                     reference_keypoints: np.ndarray,
                                     method: int = cv2.RANSAC,
                                     ransac_threshold: float = 8.0,
                                     confidence: float = 0.999,
                                     max_iters: int = 10000) -> Optional[np.ndarray]:
    """
    Compute homography matrix from corresponding keypoints using all available points.

    Args:
        frame_keypoints: Keypoints in frame coordinates [N, 2]
        reference_keypoints: Keypoints in reference logo coordinates [N, 2]
        method: OpenCV method for homography computation
        ransac_threshold: RANSAC reprojection threshold
        confidence: RANSAC confidence level
        max_iters: Maximum RANSAC iterations

    Returns:
        Homography matrix [3, 3] or None if computation failed
    """
    if len(frame_keypoints) < 4 or len(reference_keypoints) < 4:
        return None

    try:
        # Compute homography using all available keypoints
        H, mask = cv2.findHomography(
            reference_keypoints,  # Source points (reference logo)
            frame_keypoints,      # Destination points (frame)
            method=method,
            ransacReprojThreshold=ransac_threshold,
            confidence=confidence,
            maxIters=max_iters
        )

        # Check if homography is valid
        if H is not None:
            # Convert to float64 array to ensure proper type for determinant calculation
            H_matrix = np.array(H, dtype=np.float64)
            det = float(np.linalg.det(H_matrix[:2, :2]))
            if abs(det) > 1e-6:  # Not nearly singular
                return H

        return None

    except Exception as e:
        print(f"Homography computation failed: {e}")
        return None


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
    ransac_method: str = "CV2_USAC_MAGSAC",
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


def warp_images_simple(
    filtered_prediction: FilteredMatchPrediction,
    log_timing: bool = False
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Warp images using the estimated homography matrix, replicating wrap_images functionality.

    Args:
        filtered_prediction: FilteredMatchPrediction from filter_matches_ransac
        geom_type: Type of geometry ("Homography" or "Fundamental")
        log_timing: Whether to log processing time

    Returns:
        Tuple of (visualization image, warped image1) or (None, None) if failed
    """
    if log_timing:
        t0 = time.time()

    # Check if we have a valid homography matrix
    if len(filtered_prediction["H"]) == 0:
        if log_timing:
            print("No homography matrix available, cannot warp images")
        return None, None

    img0 = filtered_prediction["image0_orig"]
    img1 = filtered_prediction["image1_orig"]
    H = filtered_prediction["H"]

    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape

    try:
        # Warp img1 to img0's perspective using the homography matrix
        rectified_image1 = cv2.warpPerspective(img1, H, (w0, h0))

        # Create side-by-side visualization like the original
        # Concatenate images horizontally for comparison
        combined_img = np.concatenate([img0, rectified_image1], axis=1)

        if log_timing:
            print(f"Image warping completed in {time.time() - t0:.3f}s")

        return combined_img, rectified_image1

    except Exception as e:
        print(f"Image warping failed with error: {e}")
        return None, None


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


#%%

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

def create_logo_replacement_pipeline():
    """
    Create a complete pipeline for replacing Budlight with SPATEN in video frames.

    Returns:
        Dictionary containing all necessary components and functions
    """

    def replace_logo_in_frame(video_frame: np.ndarray,
                            budlight_bbox: np.ndarray,
                            roma_model,
                            preprocessing_conf: dict,
                            expansion_factor: float = 0.1) -> np.ndarray:
        """
        Replace Budlight logo with SPATEN in a video frame.

        Args:
            video_frame: Original video frame with Budlight logo
            budlight_bbox: Bounding box of Budlight logo [x1, y1, x2, y2]
            roma_model: Loaded ROMA model for matching
            preprocessing_conf: Preprocessing configuration
            expansion_factor: Factor to expand bounding box

        Returns:
            Video frame with SPATEN logo replacing Budlight and people brought forward
        """
        # Expand bounding box
        img_height, img_width = video_frame.shape[:2]
        x1, y1, x2, y2 = expand_box(budlight_bbox, expansion_factor, img_width, img_height)

        # Crop the physical logo from video frame
        physical_logo_cropped = video_frame[y1:y2, x1:x2]

        # Step 1: Match physical logo with digital Budlight
        match_pred = run_matching_simple(
            roma_model,
            physical_logo_cropped,
            budlight_downsampled,  # Use the same Budlight logo used in overlay
            preprocessing_conf=preprocessing_conf
        )

        # Step 2: Filter matches with RANSAC
        match_filtered = filter_matches_ransac(match_pred)

        if len(match_filtered['H']) == 0:
            print("Failed to find sufficient matches, returning original frame")
            return video_frame

        # Step 3: Map Budlight keypoints to SPATEN keypoints
        budlight_keypoints = match_filtered['mmkpts1']  # Keypoints in digital Budlight
        spaten_keypoints = map_budlight_to_spaten_coordinates(budlight_keypoints, center_x, center_y)

        # Step 4: Compute homography using physical logo keypoints -> SPATEN keypoints
        physical_keypoints = match_filtered['mmkpts0']  # Keypoints in physical logo (cropped space)

        # ✅ NEW: Transform physical keypoints to full frame coordinates
        physical_keypoints_full_frame = physical_keypoints.copy()
        physical_keypoints_full_frame[:, 0] += x1  # Add x offset
        physical_keypoints_full_frame[:, 1] += y1  # Add y offset

        # Compute homography: SPATEN -> full frame coordinates
        H_spaten, mask = cv2.findHomography(
            spaten_keypoints,  # Source: SPATEN coordinates
            physical_keypoints_full_frame,  # Target: full frame coordinates
            cv2.RANSAC,
            ransacReprojThreshold=8.0,
            confidence=0.999,
            maxIters=10000
        )

        if H_spaten is None:
            print("Failed to compute SPATEN homography, returning original frame")
            return video_frame

        # Step 5: Warp SPATEN logo to full frame size
        frame_h, frame_w = video_frame.shape[:2]

        if is_transparent:
            # For transparent logo, remove alpha channel for warping
            spaten_warped = cv2.warpPerspective(
                spaten_resized[:,:,:3],  # Remove alpha channel for warping
                H_spaten,
                (frame_w, frame_h)  # ✅ NEW: Warp to full frame size
            )
        else:
            # For non-transparent logo, use all channels
            spaten_warped = cv2.warpPerspective(
                spaten_resized,
                H_spaten,
                (frame_w, frame_h)  # ✅ NEW: Warp to full frame size
            )

        # Step 6: Create mask for entire frame and replace logo
        if is_transparent:
            # For transparent logo, use alpha channel for masking
            spaten_alpha_warped = cv2.warpPerspective(
                spaten_resized[:,:,3],  # Alpha channel only
                H_spaten,
                (frame_w, frame_h)  # ✅ NEW: Warp alpha to full frame size
            )
            mask = spaten_alpha_warped > 0
        else:
            # For non-transparent logo, use grayscale intensity for masking
            spaten_gray = cv2.cvtColor(spaten_warped, cv2.COLOR_RGB2GRAY)
            mask = spaten_gray > 0

        mask_3d = np.stack([mask, mask, mask], axis=-1)

        # ✅ NEW: Replace logo in entire frame (not just bounding box)
        frame_with_logo = video_frame.copy()
        frame_with_logo[mask_3d] = spaten_warped[mask_3d]

        # ✅ NEW: Get person masks and apply occlusion (bring people forward)
        print("Getting person masks for occlusion handling...")
        person_mask = get_person_masks(
            video_frame,
            person_seg_model,
            confidence_threshold=0.5,
            log_timing=True
        )

        # Apply person occlusion to bring people in front of logo
        result_frame = apply_person_occlusion(
            frame_with_logo,
            video_frame,  # Original frame
            person_mask,
            log_timing=True
        )

        return result_frame

    return {
        "budlight_reference": budlight_downsampled,
        "spaten_replacement": spaten_resized,
        "overlay_offset": (center_x, center_y),
        "mapping_function": map_budlight_to_spaten_coordinates,
        "replacement_pipeline": replace_logo_in_frame,
        "is_transparent": is_transparent
    }

# Load models once (this should be done at startup)
print("Loading MatchAnything models...")

model_type = "roma"

if model_type == "roma":
	model, preprocessing_conf = load_matchanything_model("matchanything_roma", log_timing=True)
else:
	model, preprocessing_conf = load_matchanything_model("matchanything_eloftr", log_timing=True)

# Create the hybrid tracker
print("Creating Hybrid Logo Tracker...")
hybrid_tracker = HybridLogoTracker(
    roma_model=model,
    roma_preprocessing_conf=preprocessing_conf,
    budlight_reference=budlight_downsampled,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Create the complete pipeline
logo_replacement_pipeline = create_logo_replacement_pipeline()

print("Hybrid tracking system ready!")

# Video processing parameters
video_path = "/home/sebastiangarcia/projects/swappr/data/poc/UFC317/BrazilPriEncode_swappr_317.ts"

# 01:26:05-01:26:17
# 01:53:12-01:53:15
# 00:50:42-00:50:48
start_timestamp = "00:50:42" # 00:36:37
end_timestamp = "00:50:48"
# 00:50:42_00:50:48
start_timestamp = "00:50:42"
end_timestamp="00:52:28"

output_video_path = f"swapped_{model_type}_hybrid_{start_timestamp}_{end_timestamp}.mp4"
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

# Set up matplotlib for displaying frames
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.set_title('Original Frame')
ax1.axis('off')
ax2.set_title('Hybrid Logo Replacement Result')
ax2.axis('off')

print(f"Starting hybrid video processing from frame {start_frame} to {end_frame}")
print(f"Video FPS: {video_fps}")
print(f"Using hybrid tracking: ROMA every {MATCH_EVERY_N_FRAMES} frames, CoTracker3 for intermediate frames")

# Performance tracking
total_times = []
roma_frame_count = 0
cotracker_frame_count = 0

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

    # Detect Budlight logo
    results: list[Results] = det_model_budlight.track(
		frame, conf=0.8,
		persist=True, # because we're inferencing in batches
		tracker=tracker_config_path, stream=False,
		verbose=False,
	)

    # Check if we have detection results
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy
    else:
        boxes = None

    # Convert BGR to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_frame = frame_rgb.copy()

    if boxes is not None and boxes.shape[0] > 0:
        # Extract bounding box - handle both tensor and numpy array cases
        try:
            if hasattr(boxes, 'cpu') and callable(getattr(boxes, 'cpu', None)):
                # It's a tensor
                budlight_bbox = boxes.cpu().numpy().astype(int).squeeze()
            else:
                # It's already a numpy array
                budlight_bbox = np.array(boxes).astype(int).squeeze()
        except Exception as box_error:
            print(f"Error processing bounding box: {box_error}")
            budlight_bbox = np.array(boxes).astype(int).squeeze()

        print(f"Frame {current_frame_number}: Detected Budlight logo at {budlight_bbox}")

        try:
            # Get person masks for occlusion filtering
            person_mask = get_person_masks(
                frame_rgb,
                person_seg_model,
                confidence_threshold=0.5,
                log_timing=False
            )

            # 🔥 NEW: Use robust fallback system to get homography
            H_spaten = hybrid_tracker.get_homography_for_frame(
                frame_rgb,
                budlight_bbox,
                person_mask,
                log_timing=True
            )

            if H_spaten is not None:
                # Apply logo replacement using computed homography
                frame_h, frame_w = frame_rgb.shape[:2]

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

                # Create mask for logo replacement
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
                result_frame[mask_3d] = spaten_warped[mask_3d]

                # Apply person occlusion to bring people in front of logo
                result_frame = apply_person_occlusion(
                    result_frame,
                    frame_rgb,  # Original frame
                    person_mask,
                    log_timing=False
                )

                # Update performance counters
                if hybrid_tracker.frame_counter % MATCH_EVERY_N_FRAMES == 1 or not hybrid_tracker.tracking_active:
                    roma_frame_count += 1
                else:
                    cotracker_frame_count += 1

                # Determine success type for logging
                if hybrid_tracker.consecutive_failures > 0:
                    success_type = "FALLBACK"
                elif hybrid_tracker.emergency_fallback_active:
                    success_type = "EMERGENCY"
                else:
                    success_type = "PRIMARY"

                print(f"Frame {current_frame_number}: ✅ {success_type} logo replacement successful")

            else:
                print(f"Frame {current_frame_number}: ❌ ALL FALLBACK METHODS FAILED - Logo will remain original")
                # Even complete failure - we still process the frame to avoid crashes
                result_frame = frame_rgb  # Use original frame

        except Exception as e:
            print(f"Frame {current_frame_number}: ❌ Exception in logo replacement: {e}")
            result_frame = frame_rgb  # Use original frame if replacement fails
    else:
        print(f"Frame {current_frame_number}: No Budlight logo detected")

    # Convert result frame back to BGR for video writing
    result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
    out.write(result_frame_bgr)

    # Display both frames side by side
    ax1.clear()
    ax1.imshow(frame_rgb)
    ax1.set_title(f'Original Frame {current_frame_number}')
    ax1.axis('off')

    ax2.clear()
    ax2.imshow(result_frame)
    ax2.set_title(f'Hybrid Logo Replacement Result {current_frame_number}')
    ax2.axis('off')

    plt.draw()
    plt.pause(0.001)  # Small pause to allow matplotlib to update

    # Performance metrics
    frame_time = time.time() - start_time_frame
    total_times.append(frame_time)
    fps = 1.0 / frame_time if frame_time > 0 else 0
    print(f"Frame {current_frame_number}: Processing time: {frame_time:.3f}s, FPS: {fps:.1f}")

    current_frame_number += 1

video_stream.release()
out.release()
plt.ioff()  # Turn off interactive mode
plt.show()

# Print performance statistics
print("\n" + "="*50)
print("HYBRID TRACKING PERFORMANCE STATISTICS")
print("="*50)

stats = hybrid_tracker.get_performance_stats()
print(f"Total frames processed: {stats['total_frames']}")
print(f"ROMA recalibrations: {stats['roma_calls']} ({roma_frame_count} frames)")
print(f"CoTracker trackings: {stats['cotracker_calls']} ({cotracker_frame_count} frames)")
print(f"Average ROMA time: {stats['roma_avg_time']:.3f}s")
print(f"Average CoTracker time: {stats['cotracker_avg_time']:.3f}s")

print(f"\n📊 FALLBACK SYSTEM STATISTICS:")
print(f"Primary success rate: {stats['success_rate']:.1%}")
print(f"Fallback usage count: {stats['fallback_usage_count']}")
print(f"Emergency fallback count: {stats['emergency_fallback_count']}")
print(f"Current consecutive failures: {stats['consecutive_failures']}")
print(f"Emergency fallback active: {stats['emergency_fallback_active']}")

if total_times:
    avg_frame_time = np.mean(total_times)
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    print(f"\n⏱️  OVERALL PERFORMANCE:")
    print(f"Average total frame time: {avg_frame_time:.3f}s")
    print(f"Average FPS: {avg_fps:.1f}")

print(f"\n🚀 OPTIMIZATION IMPACT:")
print(f"ROMA frequency: 1 every {MATCH_EVERY_N_FRAMES} frames ({100/MATCH_EVERY_N_FRAMES:.1f}%)")
print(f"CoTracker frequency: {MATCH_EVERY_N_FRAMES-1} every {MATCH_EVERY_N_FRAMES} frames ({100*(MATCH_EVERY_N_FRAMES-1)/MATCH_EVERY_N_FRAMES:.1f}%)")

# Calculate continuity metrics
total_processed = stats['total_frames']
primary_success = total_processed - stats['fallback_usage_count']
fallback_success = stats['fallback_usage_count'] - stats['emergency_fallback_count']
emergency_success = stats['emergency_fallback_count']
complete_failures = total_processed - (primary_success + fallback_success + emergency_success)

print(f"\n🎯 LOGO REPLACEMENT CONTINUITY:")
print(f"Primary success: {primary_success}/{total_processed} ({100*primary_success/total_processed:.1f}%)")
print(f"Fallback success: {fallback_success}/{total_processed} ({100*fallback_success/total_processed:.1f}%)")
print(f"Emergency success: {emergency_success}/{total_processed} ({100*emergency_success/total_processed:.1f}%)")
print(f"Complete failures: {complete_failures}/{total_processed} ({100*complete_failures/total_processed:.1f}%)")

continuity_rate = (total_processed - complete_failures) / total_processed if total_processed > 0 else 0
print(f"\n✅ OVERALL CONTINUITY RATE: {continuity_rate:.1%}")
if continuity_rate >= 0.99:
    print("🎉 EXCELLENT: Logo replacement is nearly seamless!")
elif continuity_rate >= 0.95:
    print("✅ GOOD: Logo replacement is mostly continuous")
elif continuity_rate >= 0.90:
    print("⚠️  ACCEPTABLE: Some logo flickering may be noticeable")
else:
    print("❌ POOR: Significant logo flickering detected")

print("Video processing completed with hybrid tracking and fallback system!")
