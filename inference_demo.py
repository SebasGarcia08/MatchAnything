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
