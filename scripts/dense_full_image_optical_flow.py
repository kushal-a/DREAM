import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Use 'Agg' backend to save plots without needing a display window
import matplotlib
matplotlib.use('Agg')

def save_flow_heatmap_with_bar(flow, save_path):
    """
    Plots the magnitude of optical flow as a heatmap with a color bar.
    """
    # 1. Calculate Magnitude (Speed of motion)
    # flow shape is (H, W, 2) -> mag shape is (H, W)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 2. Setup Plot
    plt.figure(figsize=(10, 6))
    
    # Use a colormap like 'inferno', 'hot', or 'viridis'
    # vmin=0 ensures the scale starts at no motion
    # vmax can be dynamic (max of current frame) or fixed (e.g., 20) to compare across frames
    plt.imshow(mag, cmap='inferno', aspect='auto') 
    
    # 3. Add Reference Bar (Colorbar)
    cbar = plt.colorbar()
    cbar.set_label('Motion Magnitude (pixels per frame)')
    
    plt.title("Optical Flow Magnitude")
    plt.axis('off') # Hide axes for cleaner image
    
    # 4. Save and Close
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

# ------------------------------------------------------------
# Main Dense Optical Flow Pipeline
# ------------------------------------------------------------
def calculate_and_save_dense_flow(input_dir, output_dir):
    
    # --- CONFIGURATION ---
    PYR_SCALE = 0.5   
    LEVELS = 3        
    WIN_SIZE = 15     
    ITERATIONS = 3    
    POLY_N = 5        
    POLY_SIGMA = 1.2  
    FLAGS = 0         
    
    START_INDEX = 1   

    # Output directories
    VISUAL_DIR = os.path.join(output_dir, "flow_visualizations_dense")
    HEATMAP_DIR = os.path.join(output_dir, "flow_heatmaps") # <--- NEW DIR
    # DATA_DIR = os.path.join(output_dir, "flow_data_dense")
    DATA_DIR = output_dir
    
    os.makedirs(VISUAL_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True) # <--- Create it
    os.makedirs(DATA_DIR, exist_ok=True)

    # Robust numeric sorting for filenames
    def numeric_key(fname):
        try:
            return int(fname.split('.')[0])
        except:
            return fname

    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")],
        key=numeric_key
    )

    if len(image_files) < 2:
        print("Need at least two frames to compute optical flow.")
        return

    # --- INITIALIZATION ---
    prev_idx = START_INDEX - 1
    prev_frame_path = os.path.join(input_dir, image_files[prev_idx])
    prev_frame = cv2.imread(prev_frame_path)
    
    if prev_frame is None: return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    hsv_buffer = np.zeros_like(prev_frame)
    hsv_buffer[..., 1] = 255

    print(f" Starting Processing...")
    print(f"  Heatmaps will be saved to: {HEATMAP_DIR}")

    # --- MAIN LOOP ---
    for i in range(START_INDEX, len(image_files)):

        image_name = image_files[i]
        curr_frame_path = os.path.join(input_dir, image_name)
        curr_frame = cv2.imread(curr_frame_path)
        if curr_frame is None: continue

        next_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 1. CALCULATE FLOW
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None, 
            PYR_SCALE, LEVELS, WIN_SIZE, ITERATIONS, POLY_N, POLY_SIGMA, FLAGS
        )

        flow /= WIN_SIZE
        flow *= 256

        # 2. SAVE DATA
        np.savez_compressed(
            os.path.join(DATA_DIR, f"{i:06d}.npz"),
            flow=flow.astype(np.int8) 
        )

        # # 3. VISUALIZE STANDARD (HSV)
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv_buffer[..., 0] = ang * 180 / np.pi / 2
        # hsv_buffer[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # vis_img = cv2.cvtColor(hsv_buffer, cv2.COLOR_HSV2BGR)
        # cv2.imwrite(os.path.join(VISUAL_DIR, f"flow_visual_{i:04d}.jpg"), vis_img)

        # # 4. VISUALIZE HEATMAP (With Bar) <--- NEW STEP
        # heatmap_path = os.path.join(HEATMAP_DIR, f"flow_heatmap_{i:04d}.jpg")
        # save_flow_heatmap_with_bar(flow, heatmap_path)

        # Update
        prev_gray = next_gray
        
        if i % 10 == 0:
            print(f"Processed frame {i}/{len(image_files)}")

    print("Completed. Heatmaps and Data saved.")

# ------------------------------------------------------------
# Usage
# ------------------------------------------------------------
INPUT_FOLDER = "data/real/panda-orb"
OUTPUT_FOLDER = "data_flow/real/panda-orb"

calculate_and_save_dense_flow(INPUT_FOLDER, OUTPUT_FOLDER)