#%% [markdown]
# # Homography estimation of UFC logos

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the necessary paths
sys.path.append(str(Path(__file__).parent))

# Import the required modules
from imcui.ui.utils import load_config, get_matcher_zoo, run_matching
from imcui.ui.viz import display_keypoints, display_matches
from typing import TypedDict, Optional

# Load images
image0 = cv2.imread("/home/sebastiangarcia/projects/swappr/data/legacy/dataset_specific_v2/train/images/converted_clip_14_frame_000000.jpg")
image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)[0: 600, 900:, ...]

image1 = cv2.imread('/home/sebastiangarcia/projects/swappr/logo_id_data/physical/bud_light.jpg')
# image1 = cv2.imread('/home/sebastiangarcia/projects/swappr/logo_id_data/physical/black_rifle_coffe.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

print(f"Image 0 shape: {image0.shape}")
print(f"Image 1 shape: {image1.shape}")

#%% [markdown]
# ## Setup Configuration and Matcher Zoo

#%%
# Load configuration
config_path = Path(__file__).parent / "config/config.yaml"
cfg = load_config(config_path)

# Get matcher zoo
matcher_zoo = get_matcher_zoo(cfg["matcher_zoo"])

print("Available matchers:")
for key in matcher_zoo.keys():
    print(f"  - {key}")

#%% [markdown]
# ## Instantiate Match Anything ELoFTR

#%%
# Parameters for ELoFTR
model_key = "matchanything_eloftr"
match_threshold = 0.01
extract_max_keypoints = 2000
keypoint_threshold = 0.05
ransac_method = "CV2_USAC_MAGSAC"
ransac_reproj_threshold = 8
ransac_confidence = 0.999
ransac_max_iter = 10000
choice_geometry_type = "Homography"

print(f"Running matching with {model_key}...")

#%%

# Run matching
results_eloftr = run_matching(
    image0=image0,
    image1=image1,
    match_threshold=match_threshold,
    extract_max_keypoints=extract_max_keypoints,
    keypoint_threshold=keypoint_threshold,
    key=model_key,
    ransac_method=ransac_method,
    ransac_reproj_threshold=ransac_reproj_threshold,
    ransac_confidence=ransac_confidence,
    ransac_max_iter=ransac_max_iter,
    choice_geometry_type=choice_geometry_type,
    matcher_zoo=matcher_zoo,
    force_resize=False,
    image_width=640,
    image_height=480,
    use_cached_model=False,
)

# Extract results
(output_keypoints_eloftr, output_matches_raw_eloftr, output_matches_ransac_eloftr,
 num_matches_eloftr, configs_eloftr, geom_info_eloftr, output_wrapped_eloftr,
 state_cache_eloftr, output_pred_eloftr) = results_eloftr

print(f"ELoFTR Results:")
print(f"  - Raw matches: {num_matches_eloftr['num_raw_matches']}")
print(f"  - RANSAC matches: {num_matches_eloftr['num_ransac_matches']}")

#%% [markdown]
# ## Display ELoFTR Results

#%%
# Display the three plots for ELoFTR
fig, axes = plt.subplots(3, 1, figsize=(15, 20))

# Plot 1: Keypoints
axes[0].imshow(output_keypoints_eloftr)
axes[0].set_title("ELoFTR - Keypoints Detection", fontsize=16)
axes[0].axis('off')

# Plot 2: Raw Matches
axes[1].imshow(output_matches_raw_eloftr)
axes[1].set_title("ELoFTR - Raw Matches", fontsize=16)
axes[1].axis('off')

# Plot 3: RANSAC Matches
axes[2].imshow(output_matches_ransac_eloftr)
axes[2].set_title("ELoFTR - RANSAC Matches", fontsize=16)
axes[2].axis('off')

plt.tight_layout()
plt.show()

#%% [markdown]
# ## Instantiate Match Anything ROMA

#%%
# Parameters for ROMA
model_key = "matchanything_roma"
match_threshold = 0.01
extract_max_keypoints = 2000
keypoint_threshold = 0.05
ransac_method = "CV2_USAC_MAGSAC"
ransac_reproj_threshold = 8
ransac_confidence = 0.999
ransac_max_iter = 10000
choice_geometry_type = "Homography"

print(f"Running matching with {model_key}...")

# Run matching
results_roma = run_matching(
    image0=image0,
    image1=image1,
    match_threshold=match_threshold,
    extract_max_keypoints=extract_max_keypoints,
    keypoint_threshold=keypoint_threshold,
    key=model_key,
    ransac_method=ransac_method,
    ransac_reproj_threshold=ransac_reproj_threshold,
    ransac_confidence=ransac_confidence,
    ransac_max_iter=ransac_max_iter,
    choice_geometry_type=choice_geometry_type,
    matcher_zoo=matcher_zoo,
    force_resize=False,
    image_width=640,
    image_height=480,
    use_cached_model=False,
)

# Extract results
(output_keypoints_roma, output_matches_raw_roma, output_matches_ransac_roma,
 num_matches_roma, configs_roma, geom_info_roma, output_wrapped_roma,
 state_cache_roma, output_pred_roma) = results_roma

print(f"ROMA Results:")
print(f"  - Raw matches: {num_matches_roma['num_raw_matches']}")
print(f"  - RANSAC matches: {num_matches_roma['num_ransac_matches']}")

#%% [markdown]
# ## Display ROMA Results

#%%
# Display the three plots for ROMA
fig, axes = plt.subplots(3, 1, figsize=(15, 20))

# Plot 1: Keypoints
axes[0].imshow(output_keypoints_roma)
axes[0].set_title("ROMA - Keypoints Detection", fontsize=16)
axes[0].axis('off')

# Plot 2: Raw Matches
axes[1].imshow(output_matches_raw_roma)
axes[1].set_title("ROMA - Raw Matches", fontsize=16)
axes[1].axis('off')

# Plot 3: RANSAC Matches
axes[2].imshow(output_matches_ransac_roma)
axes[2].set_title("ROMA - RANSAC Matches", fontsize=16)
axes[2].axis('off')

plt.tight_layout()
plt.show()

#%% [markdown]
# ## Compare Results

#%%
# Compare the results between ELoFTR and ROMA
comparison_data = {
    'Model': ['ELoFTR', 'ROMA'],
    'Raw Matches': [num_matches_eloftr['num_raw_matches'], num_matches_roma['num_raw_matches']],
    'RANSAC Matches': [num_matches_eloftr['num_ransac_matches'], num_matches_roma['num_ransac_matches']]
}

import pandas as pd
df = pd.DataFrame(comparison_data)
print("\nComparison Results:")
print(df.to_string(index=False))

#%% [markdown]
# ## Display Warped Images

#%%
# Display the warped images for both models
fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# ELoFTR warped result
if output_wrapped_eloftr is not None:
    axes[0].imshow(output_wrapped_eloftr)
    axes[0].set_title("ELoFTR - Warped Images (Homography)", fontsize=16)
    axes[0].axis('off')
else:
    axes[0].text(0.5, 0.5, "ELoFTR - No warped image available", 
                ha='center', va='center', transform=axes[0].transAxes, fontsize=14)
    axes[0].axis('off')

# ROMA warped result
if output_wrapped_roma is not None:
    axes[1].imshow(output_wrapped_roma)
    axes[1].set_title("ROMA - Warped Images (Homography)", fontsize=16)
    axes[1].axis('off')
else:
    axes[1].text(0.5, 0.5, "ROMA - No warped image available", 
                ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
    axes[1].axis('off')

plt.tight_layout()
plt.show()

print("\nHomography estimation completed!")
print("The warped images show how well the second image (Bud Light logo) can be")
print("aligned with the first image (UFC frame) using the estimated homography matrix.")

#%% [markdown]
# # Simplifying model inference

#%%
# TODO: 
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


#%% [markdown]
# ## Test the simplified pipeline

#%%
# Load models once (this should be done at startup)
print("Loading MatchAnything models...")

eloftr_model, eloftr_preprocessing_conf = load_matchanything_model("matchanything_eloftr", log_timing=True)
roma_model, roma_preprocessing_conf = load_matchanything_model("matchanything_roma", log_timing=True)

print("Models loaded successfully!")

#%%
# Test with ELoFTR
print("\n=== Testing ELoFTR Simplified Pipeline ===")

# Run matching with the original preprocessing configuration
eloftr_prediction = run_matching_simple(
    eloftr_model,
    image0,  # UFC frame
    image1,  # Bud Light logo
    preprocessing_conf=eloftr_preprocessing_conf,  # Use the original config!
    log_timing=True
)

print(f"ELoFTR raw matches: {len(eloftr_prediction['mkpts0'])}")

# Filter with RANSAC
eloftr_filtered = filter_matches_ransac(
    eloftr_prediction,
    log_timing=True
)

print(f"ELoFTR filtered matches: {len(eloftr_filtered['mmkpts0'])}")
if len(eloftr_filtered['H']) > 0:
    print(f"Homography matrix shape: {eloftr_filtered['H'].shape}")

#%%
# Test with ROMA
print("\n=== Testing ROMA Simplified Pipeline ===")

# Run matching with the original preprocessing configuration
roma_prediction = run_matching_simple(
    roma_model,
    image0,  # UFC frame
    image1,  # Bud Light logo
    preprocessing_conf=roma_preprocessing_conf,  # Use the original config!
    log_timing=True
)

print(f"ROMA raw matches: {len(roma_prediction['mkpts0'])}")

# Filter with RANSAC
roma_filtered = filter_matches_ransac(
    roma_prediction,
    log_timing=True
)

print(f"ROMA filtered matches: {len(roma_filtered['mmkpts0'])}")
if len(roma_filtered['H']) > 0:
    print(f"Homography matrix shape: {roma_filtered['H'].shape}")

#%% [markdown]
# ## Performance Comparison

#%%
# Compare with original pipeline timing
print("\n=== Performance Comparison ===")

# Time the original run_matching function for comparison
print("Original pipeline timing (ELoFTR):")
t0 = time.time()
original_results = run_matching(
    image0=image0,
    image1=image1,
    match_threshold=0.01,
    extract_max_keypoints=2000,
    keypoint_threshold=0.05,
    key="matchanything_eloftr",
    ransac_method="CV2_USAC_MAGSAC",
    ransac_reproj_threshold=8,
    ransac_confidence=0.999,
    ransac_max_iter=10000,
    choice_geometry_type="Homography",
    matcher_zoo=matcher_zoo,
    force_resize=False,
    image_width=640,
    image_height=480,
    use_cached_model=False,
)
original_time = time.time() - t0
print(f"Original pipeline total time: {original_time:.3f}s")

# Time our simplified pipeline (using the original preprocessing config)
print("\nSimplified pipeline timing (ELoFTR):")
t0 = time.time()
simple_prediction = run_matching_simple(
    eloftr_model, 
    image0, 
    image1, 
    preprocessing_conf=eloftr_preprocessing_conf,  # Use original config!
    log_timing=False
)
simple_filtered = filter_matches_ransac(simple_prediction, log_timing=False)
simplified_time = time.time() - t0
print(f"Simplified pipeline total time: {simplified_time:.3f}s")

print(f"\nSpeedup: {original_time/simplified_time:.2f}x faster")
print("Note: Simplified pipeline excludes visualization generation and model reloading")

print("\nPipeline ready for semi-real-time usage!")
print("Recommended usage:")
print("1. Load models once at startup with load_matchanything_model()")
print("2. For each frame pair, call run_matching_simple() + filter_matches_ransac()")
print("3. Use the homography matrix for logo replacement")

# %%
