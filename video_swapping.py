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
MATCH_EVERY_N_FRAMES = 160
MAX_KP_TO_TRACK = 50
MIN_KP_FOR_HOMOGRAPHY = 100  # Increased from default
ROMA_CONFIDENCE_THRESHOLD = 0.05  # Increased from 0.02
COTRACKER_INPUT_RESOLUTION = (384, 512)  # H, W for CoTracker3

# Memory optimization settings
USE_MIXED_PRECISION = True  # Enable half precision for memory savings
REDUCE_FRAME_BUFFER_SIZE = True  # Use minimal frame buffer
COTRACKER_FRAME_BUFFER_SIZE = 16  # Reduced from 32 (model.step * 2)
PROCESS_RESOLUTION_SCALE = 0.8  # Process at 80% resolution to save memory
UNLOAD_MODELS_BETWEEN_FRAMES = False  # Extreme memory saving (slower)
ENABLE_MEMORY_MONITORING = True  # Monitor GPU memory usage

# Global variables for tracking state
cotracker_model = None
cotracker_initialized = False
cotracker_queries = None
frame_buffer = []
current_keypoints = None
reference_keypoints = None
frame_counter = 0

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0

def log_memory_usage(stage: str):
    """Log GPU memory usage at different stages."""
    if ENABLE_MEMORY_MONITORING and torch.cuda.is_available():
        memory_mb = get_gpu_memory_usage()
        print(f"GPU Memory [{stage}]: {memory_mb:.1f} MB")

def clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def unload_cotracker_model():
    """Unload CoTracker model to free memory (extreme memory saving)."""
    global cotracker_model, cotracker_initialized
    if cotracker_model is not None:
        del cotracker_model
        cotracker_model = None
        cotracker_initialized = False
        clear_gpu_memory()
        print("CoTracker model unloaded to save memory")

def initialize_cotracker():
    """Initialize CoTracker3 online model with memory optimizations."""
    global cotracker_model
    if cotracker_model is None:
        log_memory_usage("Before CoTracker loading")
        print("Loading CoTracker3 online model with memory optimizations...")
        with torch.inference_mode():  # Memory optimization
            cotracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cotracker_model = cotracker_model.to(device)
            
            # Use mixed precision for memory savings
            if USE_MIXED_PRECISION and torch.cuda.is_available():
                cotracker_model = cotracker_model.half()  # Convert to half precision
                print("CoTracker3 converted to half precision for memory savings")
            
        print(f"CoTracker3 online model loaded successfully! Step size: {cotracker_model.step}")
        clear_gpu_memory()  # Clear memory after loading
        log_memory_usage("After CoTracker loading")

def detect_keypoints_with_roma(frame: np.ndarray, 
                              logo_bbox: np.ndarray,
                              roma_model,
                              preprocessing_conf: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect keypoints using MatchAnything ROMA model with memory optimizations.
    
    Args:
        frame: Current frame
        logo_bbox: Logo bounding box [x1, y1, x2, y2]
        roma_model: ROMA model
        preprocessing_conf: Preprocessing configuration
        
    Returns:
        Tuple of (frame_keypoints, reference_keypoints) or (None, None) if failed
    """
    # Memory optimization: Use inference mode for all operations
    with torch.inference_mode():
        log_memory_usage("Before ROMA processing")
        
        # Expand bounding box
        img_height, img_width = frame.shape[:2]
        x1, y1, x2, y2 = expand_box(logo_bbox, 0.1, img_width, img_height)
        
        # Crop logo region
        physical_logo_cropped = frame[y1:y2, x1:x2]
        
        # Memory optimization: Reduce processing resolution if needed
        if PROCESS_RESOLUTION_SCALE < 1.0:
            new_h = int(physical_logo_cropped.shape[0] * PROCESS_RESOLUTION_SCALE)
            new_w = int(physical_logo_cropped.shape[1] * PROCESS_RESOLUTION_SCALE)
            physical_logo_cropped = cv2.resize(physical_logo_cropped, (new_w, new_h))
            # Scale coordinates back later
            scale_factor = 1.0 / PROCESS_RESOLUTION_SCALE
        else:
            scale_factor = 1.0
        
        # Run ROMA matching
        match_pred = run_matching_simple(
            roma_model,
            physical_logo_cropped,
            budlight_downsampled,
            preprocessing_conf=preprocessing_conf,
            match_threshold=ROMA_CONFIDENCE_THRESHOLD,
            extract_max_keypoints=MAX_KP_TO_TRACK,
            log_timing=False
        )
        
        # Filter matches with RANSAC
        match_filtered = filter_matches_ransac(match_pred, log_timing=False)
        
        if len(match_filtered['H']) == 0 or len(match_filtered['mmkpts0']) < MIN_KP_FOR_HOMOGRAPHY:
            print(f"ROMA: Insufficient matches ({len(match_filtered.get('mmkpts0', []))}) for reliable tracking")
            return None, None
        
        # Get keypoints in cropped coordinates
        frame_keypoints_cropped = match_filtered['mmkpts0']  # [N, 2]
        reference_keypoints = match_filtered['mmkpts1']  # [N, 2]
        
        # Scale keypoints back if resolution was reduced
        if scale_factor != 1.0:
            frame_keypoints_cropped = frame_keypoints_cropped * scale_factor
        
        # Convert to full frame coordinates
        frame_keypoints_full = frame_keypoints_cropped.copy()
        frame_keypoints_full[:, 0] += x1  # Add x offset
        frame_keypoints_full[:, 1] += y1  # Add y offset
        
        print(f"ROMA: Detected {len(frame_keypoints_full)} keypoints")
        
        # Memory cleanup
        del match_pred, match_filtered, physical_logo_cropped
        clear_gpu_memory()
        log_memory_usage("After ROMA processing")
        
        return frame_keypoints_full, reference_keypoints

def initialize_cotracker_online(initial_frame: np.ndarray, 
                               initial_keypoints: np.ndarray) -> bool:
    """
    Initialize CoTracker3 online processing with memory optimizations.
    
    Args:
        initial_frame: Initial frame for CoTracker3
        initial_keypoints: Initial keypoints [N, 2] in full frame coordinates
        
    Returns:
        True if initialization successful, False otherwise
    """
    global cotracker_initialized, cotracker_queries, frame_buffer
    
    # Memory optimization: Use inference mode
    with torch.inference_mode():
        # Ensure CoTracker model is loaded
        initialize_cotracker()
        
        if cotracker_model is None:
            return False
        
        # Prepare frame tensor - CoTracker3 online expects [B, T, C, H, W]
        frame_resized = cv2.resize(initial_frame, COTRACKER_INPUT_RESOLUTION[::-1])  # (W, H)
        frame_tensor = torch.from_numpy(frame_resized).float()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        frame_tensor = frame_tensor.to(device)
        
        # Use mixed precision if enabled
        if USE_MIXED_PRECISION and torch.cuda.is_available():
            frame_tensor = frame_tensor.half()
        
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
        
        # Scale keypoints to CoTracker resolution
        frame_height, frame_width = initial_frame.shape[:2]
        scaled_keypoints = initial_keypoints.copy()
        scaled_keypoints[:, 0] *= COTRACKER_INPUT_RESOLUTION[1] / frame_width   # Scale x
        scaled_keypoints[:, 1] *= COTRACKER_INPUT_RESOLUTION[0] / frame_height  # Scale y
        
        # Prepare queries tensor: [1, N, 3] where each point is [t, x, y]
        num_points = len(scaled_keypoints)
        queries = torch.zeros(1, num_points, 3).to(device)
        
        # Use mixed precision if enabled
        if USE_MIXED_PRECISION and torch.cuda.is_available():
            queries = queries.half()
        
        queries[0, :, 0] = 0  # Time index (first frame)
        queries[0, :, 1] = torch.from_numpy(scaled_keypoints[:, 0]).float().to(device)
        queries[0, :, 2] = torch.from_numpy(scaled_keypoints[:, 1]).float().to(device)
        
        # Use mixed precision if enabled
        if USE_MIXED_PRECISION and torch.cuda.is_available():
            queries[0, :, 1] = queries[0, :, 1].half()
            queries[0, :, 2] = queries[0, :, 2].half()
        
        # Store queries for later use
        cotracker_queries = queries
        
        # Initialize CoTracker3 online processing
        cotracker_model(
            video_chunk=frame_tensor,
            is_first_step=True,
            grid_size=0,
            queries=queries,
            add_support_grid=False
        )
        
        # Memory optimization: Use smaller frame buffer
        buffer_size = COTRACKER_FRAME_BUFFER_SIZE if REDUCE_FRAME_BUFFER_SIZE else (cotracker_model.step * 2)
        frame_buffer = [initial_frame] * buffer_size
        
        cotracker_initialized = True
        print(f"CoTracker3 online initialized with {num_points} keypoints (buffer size: {buffer_size})")
        
        # Memory cleanup
        del frame_tensor, queries
        clear_gpu_memory()
        
        return True

def track_keypoints_with_cotracker(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Track keypoints using CoTracker3 online model with memory optimizations.
    
    Args:
        frame: Current frame
        
    Returns:
        Tracked keypoints [N, 2] in full frame coordinates or None if failed
    """
    global frame_buffer, cotracker_queries
    
    if not cotracker_initialized or cotracker_queries is None or cotracker_model is None:
        return None
    
    # Memory optimization: Use inference mode
    with torch.inference_mode():
        # Add new frame to buffer and remove oldest
        frame_buffer.append(frame)
        
        # Memory optimization: Keep smaller buffer size
        buffer_size = COTRACKER_FRAME_BUFFER_SIZE if REDUCE_FRAME_BUFFER_SIZE else (cotracker_model.step * 2)
        frame_buffer = frame_buffer[-buffer_size:]
        
        # Prepare video chunk - use available frames
        video_chunk_frames = frame_buffer[-min(len(frame_buffer), cotracker_model.step * 2):]
        
        # Resize frames to CoTracker resolution
        video_chunk_resized = []
        for frame_i in video_chunk_frames:
            frame_resized = cv2.resize(frame_i, COTRACKER_INPUT_RESOLUTION[::-1])  # (W, H)
            video_chunk_resized.append(frame_resized)
        
        # Convert to tensor [1, T, C, H, W]
        video_chunk_tensor = torch.from_numpy(np.stack(video_chunk_resized)).float()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        video_chunk_tensor = video_chunk_tensor.to(device)
        
        # Use mixed precision if enabled
        if USE_MIXED_PRECISION and torch.cuda.is_available():
            video_chunk_tensor = video_chunk_tensor.half()
        
        video_chunk_tensor = video_chunk_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, 3, H, W]
        
        # Run CoTracker3 online inference
        pred_tracks, pred_visibility = cotracker_model(
            video_chunk=video_chunk_tensor,
            is_first_step=False,
            grid_size=0,
            queries=cotracker_queries,
            add_support_grid=False
        )
        
        # Get tracks for the last frame (most recent)
        last_frame_tracks = pred_tracks[0, -1].cpu().numpy()  # [N, 2]
        last_frame_visibility = pred_visibility[0, -1].cpu().numpy()  # [N]
        
        # Scale from CoTracker resolution to full frame resolution
        frame_height, frame_width = frame.shape[:2]
        last_frame_tracks[:, 0] *= frame_width / COTRACKER_INPUT_RESOLUTION[1]   # Scale x
        last_frame_tracks[:, 1] *= frame_height / COTRACKER_INPUT_RESOLUTION[0]  # Scale y
        
        print(f"CoTracker3: Tracked {len(last_frame_tracks)} keypoints")
        
        # Memory cleanup
        del video_chunk_tensor, pred_tracks, pred_visibility
        clear_gpu_memory()
        
        return last_frame_tracks

def filter_occluded_keypoints(keypoints: np.ndarray,
                             person_masks: np.ndarray,
                             reference_keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter keypoints that are occluded by people, but keep tracking them.
    
    Args:
        keypoints: Keypoints [N, 2] in full frame coordinates
        person_masks: Person segmentation masks [H, W]
        reference_keypoints: Reference keypoints [N, 2] in logo coordinates
        
    Returns:
        Tuple of (visible_keypoints, visible_reference_keypoints) for homography computation
    """
    if len(keypoints) == 0 or len(reference_keypoints) == 0:
        return keypoints, reference_keypoints
    
    # Check which keypoints are not occluded
    visible_mask = np.ones(len(keypoints), dtype=bool)
    
    for i, (x, y) in enumerate(keypoints):
        # Check if keypoint is within frame bounds
        if x < 0 or y < 0 or x >= person_masks.shape[1] or y >= person_masks.shape[0]:
            visible_mask[i] = False
            continue
        
        # Check if keypoint is occluded by person
        if person_masks[int(y), int(x)] > 0:
            visible_mask[i] = False
    
    # Filter keypoints for homography computation (only visible ones)
    visible_keypoints = keypoints[visible_mask]
    visible_reference_keypoints = reference_keypoints[visible_mask]
    
    print(f"Occlusion filter: {len(visible_keypoints)}/{len(keypoints)} keypoints visible")
    return visible_keypoints, visible_reference_keypoints

def get_keypoints_for_frame(frame: np.ndarray,
                           logo_bbox: np.ndarray,
                           person_masks: np.ndarray,
                           roma_model,
                           preprocessing_conf: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get keypoints for current frame using hybrid tracking approach.
    
    Args:
        frame: Current frame
        logo_bbox: Logo bounding box [x1, y1, x2, y2]
        person_masks: Person segmentation masks [H, W]
        roma_model: ROMA model
        preprocessing_conf: Preprocessing configuration
        
    Returns:
        Tuple of (frame_keypoints, reference_keypoints) or (None, None) if failed
    """
    global frame_counter, current_keypoints, reference_keypoints, cotracker_initialized
    
    frame_counter += 1
    is_recalibration_frame = frame_counter % MATCH_EVERY_N_FRAMES == 1
    
    print(f"Frame {frame_counter}: {'ROMA recalibration' if is_recalibration_frame else 'CoTracker tracking'}")
    
    if is_recalibration_frame or not cotracker_initialized:
        # Recalibration phase: Use ROMA to detect keypoints
        frame_kp, ref_kp = detect_keypoints_with_roma(frame, logo_bbox, roma_model, preprocessing_conf)
        
        if frame_kp is not None and ref_kp is not None:
            # Filter occluded keypoints
            frame_kp_filtered, ref_kp_filtered = filter_occluded_keypoints(frame_kp, person_masks, ref_kp)
            
            if len(frame_kp_filtered) >= MIN_KP_FOR_HOMOGRAPHY:
                # Update tracking state
                current_keypoints = frame_kp
                reference_keypoints = ref_kp
                
                # Initialize CoTracker3 online for next frames
                initialize_cotracker_online(frame, frame_kp)
                
                print(f"Frame {frame_counter}: ROMA success with {len(frame_kp_filtered)} visible keypoints")
                return frame_kp_filtered, ref_kp_filtered
            else:
                print(f"Frame {frame_counter}: Insufficient visible keypoints after filtering ({len(frame_kp_filtered)})")
                return None, None
        else:
            print(f"Frame {frame_counter}: ROMA detection failed")
            return None, None
    
    else:
        # Tracking phase: Use CoTracker3 online to track existing keypoints
        tracked_kp = track_keypoints_with_cotracker(frame)
        
        if tracked_kp is not None and reference_keypoints is not None:
            # Filter occluded keypoints for homography computation
            tracked_kp_filtered, ref_kp_filtered = filter_occluded_keypoints(tracked_kp, person_masks, reference_keypoints)
            
            if len(tracked_kp_filtered) >= MIN_KP_FOR_HOMOGRAPHY:
                print(f"Frame {frame_counter}: CoTracker success with {len(tracked_kp_filtered)} visible keypoints")
                return tracked_kp_filtered, ref_kp_filtered
            else:
                print(f"Frame {frame_counter}: Insufficient visible keypoints after tracking ({len(tracked_kp_filtered)})")
                # Reset for immediate recalibration
                cotracker_initialized = False
                frame_counter = MATCH_EVERY_N_FRAMES  # Force recalibration
                return None, None
        else:
            print(f"Frame {frame_counter}: CoTracker tracking failed")
            # Reset for immediate recalibration
            cotracker_initialized = False
            frame_counter = MATCH_EVERY_N_FRAMES  # Force recalibration
            return None, None


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
# Remove the HybridLogoTracker instantiation since we're using simple functions now

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
        if hasattr(boxes, 'cpu') and callable(getattr(boxes, 'cpu', None)):
            # It's a tensor
            budlight_bbox = boxes.cpu().numpy().astype(int).squeeze()
        else:
            # It's already a numpy array
            budlight_bbox = np.array(boxes).astype(int).squeeze()

        print(f"Frame {current_frame_number}: Detected Budlight logo at {budlight_bbox}")

        # Get person masks for occlusion filtering
        person_mask = get_person_masks(
            frame_rgb,
            person_seg_model,
            confidence_threshold=0.5,
            log_timing=False
        )

        # Get keypoints using hybrid tracking approach
        frame_keypoints, reference_keypoints = get_keypoints_for_frame(
            frame_rgb,
            budlight_bbox,
            person_mask,
            model,
            preprocessing_conf
        )

        if frame_keypoints is not None and reference_keypoints is not None:
            # Compute homography for logo replacement
            spaten_keypoints = map_budlight_to_spaten_coordinates(reference_keypoints, center_x, center_y)
            H_spaten = compute_homography_from_keypoints(frame_keypoints, spaten_keypoints)

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

                print(f"Frame {current_frame_number}: ✅ Logo replacement successful")
            else:
                print(f"Frame {current_frame_number}: ❌ Homography computation failed")
        else:
            print(f"Frame {current_frame_number}: ❌ Keypoint detection/tracking failed")
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

if total_times:
    avg_frame_time = np.mean(total_times)
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    print(f"Average frame time: {avg_frame_time:.3f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Total frames processed: {len(total_times)}")
    print(f"ROMA recalibrations: {len(total_times) // MATCH_EVERY_N_FRAMES + 1}")
    print(f"CoTracker trackings: {len(total_times) - (len(total_times) // MATCH_EVERY_N_FRAMES + 1)}")

print("Video processing completed with simplified hybrid tracking!")
