import time
import cv2
from swappr.utils import parse_timestamp_to_frame
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle
import os

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

# Load overlay components
overlay_components_path = Path(__file__).parent / "overlay_components" / "overlay_components.pkl"
print(f"Loading overlay components from: {overlay_components_path}")

with open(overlay_components_path, "rb") as f:
    overlay_components = pickle.load(f)

# Extract components for global use
budlight_downsampled = overlay_components["budlight_downsampled"]
spaten_resized = overlay_components["spaten_resized"]
center_x = overlay_components["center_x"]
center_y = overlay_components["center_y"]

print(f"Loaded overlay components:")
print(f"  - Budlight reference: {budlight_downsampled.shape}")
print(f"  - Spaten replacement: {spaten_resized.shape}")
print(f"  - Overlay offset: ({center_x}, {center_y})")


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
            Video frame with SPATEN logo replacing Budlight
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
        physical_keypoints = match_filtered['mmkpts0']  # Keypoints in physical logo

        # Compute homography: physical logo -> SPATEN
        H_spaten, mask = cv2.findHomography(
            spaten_keypoints,  # Source: SPATEN coordinates
            physical_keypoints,  # Target: physical logo coordinates
            cv2.RANSAC,
            ransacReprojThreshold=8.0,
            confidence=0.999,
            maxIters=10000
        )

        if H_spaten is None:
            print("Failed to compute SPATEN homography, returning original frame")
            return video_frame

        # Step 5: Warp SPATEN logo to match physical logo perspective
        crop_h, crop_w = physical_logo_cropped.shape[:2]
        spaten_warped = cv2.warpPerspective(
            spaten_resized[:,:,:3],  # Remove alpha channel for warping
            H_spaten,
            (crop_w, crop_h)
        )

        # Step 6: Create mask and replace logo in video frame
        spaten_gray = cv2.cvtColor(spaten_warped, cv2.COLOR_RGB2GRAY)
        mask = spaten_gray > 0
        mask_3d = np.stack([mask, mask, mask], axis=-1)

        # Replace the logo in the video frame
        result_frame = video_frame.copy()
        result_frame[y1:y2, x1:x2][mask_3d] = spaten_warped[mask_3d]

        return result_frame

    return {
        "budlight_reference": budlight_downsampled,
        "spaten_replacement": spaten_resized,
        "overlay_offset": (center_x, center_y),
        "mapping_function": map_budlight_to_spaten_coordinates,
        "replacement_pipeline": replace_logo_in_frame
    }

# Load models once (this should be done at startup)
print("Loading MatchAnything models...")

# eloftr_model, eloftr_preprocessing_conf = load_matchanything_model("matchanything_eloftr", log_timing=True)
roma_model, roma_preprocessing_conf = load_matchanything_model("matchanything_roma", log_timing=True)

# Create the complete pipeline
logo_replacement_pipeline = create_logo_replacement_pipeline()

print("Logo replacement pipeline created successfully!")

# Video processing parameters
video_path = "/home/sebastiangarcia/projects/swappr/data/poc/UFC317/BrazilPriEncode_swappr_317.ts"
# 01:26:05-01:26:17
# 01:53:12-01:53:15
# 00:50:42-00:50:48
start_timestamp = "00:50:42" # 00:36:37
end_timestamp = "00:50:48"
yolo_model_path = "/home/sebastiangarcia/projects/swappr/models/poc/v2_budlight_logo_detection/weights/best.pt"

det_model_budlight = YOLO(yolo_model_path)
video_stream = cv2.VideoCapture(video_path)
video_fps: float = video_stream.get(cv2.CAP_PROP_FPS)

start_frame = parse_timestamp_to_frame(start_timestamp, video_fps)
end_frame = parse_timestamp_to_frame(end_timestamp, video_fps)

video_stream.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
current_frame_number = start_frame

# Set up matplotlib for displaying frames
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.set_title('Original Frame')
ax1.axis('off')
ax2.set_title('Logo Replacement Result')
ax2.axis('off')

print(f"Starting video processing from frame {start_frame} to {end_frame}")
print(f"Video FPS: {video_fps}")

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
    results = det_model_budlight(frame)
    boxes = results[0].boxes.xyxy
    
    # Convert BGR to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_frame = frame_rgb.copy()
    
    if boxes.shape[0] > 0:
        budlight_bbox = boxes.cpu().numpy().astype("int").squeeze()
        print(f"Frame {current_frame_number}: Detected Budlight logo at {budlight_bbox}")
        
        try:
            # Apply logo replacement pipeline
            result_frame = logo_replacement_pipeline["replacement_pipeline"](
                frame_rgb,
                budlight_bbox,
                roma_model,
                roma_preprocessing_conf,
                expansion_factor=0.1
            )
            print(f"Frame {current_frame_number}: Logo replacement applied successfully")
        except Exception as e:
            print(f"Frame {current_frame_number}: Logo replacement failed: {e}")
            result_frame = frame_rgb  # Use original frame if replacement fails
    else:
        print(f"Frame {current_frame_number}: No Budlight logo detected")
    
    # Display both frames side by side
    ax1.clear()
    ax1.imshow(frame_rgb)
    ax1.set_title(f'Original Frame {current_frame_number}')
    ax1.axis('off')
    
    ax2.clear()
    ax2.imshow(result_frame)
    ax2.set_title(f'Logo Replacement Result {current_frame_number}')
    ax2.axis('off')
    
    plt.draw()
    plt.pause(0.001)  # Small pause to allow matplotlib to update
    
    # Performance metrics
    frame_time = time.time() - start_time_frame
    fps = 1.0 / frame_time if frame_time > 0 else 0
    print(f"Frame {current_frame_number}: Processing time: {frame_time:.3f}s, FPS: {fps:.1f}")
    
    current_frame_number += 1
    
    # Optional: Add a small delay to control playback speed
    # time.sleep(1/video_fps)  # Uncomment to play at original speed

video_stream.release()
plt.ioff()  # Turn off interactive mode
plt.show()

print("Video processing completed!")
