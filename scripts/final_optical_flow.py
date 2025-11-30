import os
import cv2
import numpy as np

def calculate_and_save_flow_with_history(input_dir, output_dir):
    
    # --- CONFIGURATION ---
    # Farneback Params
    PYR_SCALE = 0.5   
    LEVELS = 3        
    WIN_SIZE = 15     
    ITERATIONS = 3    
    POLY_N = 5        
    POLY_SIGMA = 1.2  
    FLAGS = 0         
    
    # Threshold for scene change (Pixels per frame)
    SCENE_CHANGE_THRESHOLD = 20.0 
    
    # History length
    SEQ_LEN = 16
    
    os.makedirs(output_dir, exist_ok=True)

    # Sort files numerically
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
        print("Need at least two frames.")
        return

    # Buffer to keep track of valid previous filenames
    # This persists across the loop
    history_buffer = []

    print(f" Starting Processing...")
    print(f"  Scene Cut Threshold: {SCENE_CHANGE_THRESHOLD} px")

    # --- MAIN LOOP ---
    # Adjust range as needed. Here processing all available frames starting from index 1.
    for i in range(1, len(image_files)):
        
        # 1. Load Frames
        prev_image_name = image_files[i-1]
        curr_image_name = image_files[i]

        prev_path = os.path.join(input_dir, prev_image_name)
        curr_path = os.path.join(input_dir, curr_image_name)

        prev_frame = cv2.imread(prev_path)
        curr_frame = cv2.imread(curr_path)

        if prev_frame is None or curr_frame is None:
            continue

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 2. Calculate Flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None, 
            PYR_SCALE, LEVELS, WIN_SIZE, ITERATIONS, POLY_N, POLY_SIGMA, FLAGS
        )

        # 3. Check for Scene Change / High Flow
        # We check the Mean Magnitude. If the average movement is > 15, it's a jump.
        mag = np.linalg.norm(flow, axis=2)
        mag_f = mag.max()
        
        # Name of the file we are about to save
        current_filename = f"{i:06d}.npz"
        
        # List that will be saved inside the NPZ
        history_to_save = []

        if mag_f > SCENE_CHANGE_THRESHOLD:
            print(f"⚠ Frame {i}: Flow too high ({mag_f:.2f} > 20). Treating as SCENE CHANGE.")
            
            # ACTION A: Make Flow 0
            flow = np.zeros_like(flow)
            
            # ACTION B: Reset History for this save
            # User requirement: "save all files in the list before that frame as 0000.npz"
            history_to_save = ["000000.npz"] * SEQ_LEN
            
            # ACTION C: Clear global buffer
            # This ensures frame i+1 does not reference frames from before the cut
            history_buffer = []
            
        else:
            # Normal Operation
            
            # Get the last 16 items from the running buffer
            # Copy it to avoid modifying the global list
            valid_history = history_buffer[-SEQ_LEN:]
            
            # Pad with "0000.npz" if we don't have enough history yet
            padding_needed = SEQ_LEN - len(valid_history)
            history_to_save = (["0000.npz"] * padding_needed) + valid_history

        # 4. Save Data
        # We save the flow AND the list of strings
        # max_flow = max(max_flow, flow_mag.max())
        # flow /= 50
        # flow *= 256
        flow = (flow / 20) * 127

        save_path = os.path.join(output_dir, current_filename)
        np.savez_compressed(
            save_path,
            flow=flow.astype(np.int8),
            history=np.array(history_to_save) # Save as numpy array of strings
        )

        # 5. Update Global Buffer
        # Add the CURRENT file to the buffer so the NEXT frame can see it
        # If we had a scene change, we still add this (now zeroed) frame as the start of new history
        history_buffer.append(current_filename)
        
        if i % 10 == 0:
            print(f"Processed frame {i}")

    print(f"Completed. Data saved to {output_dir}")

# ------------------------------------------------------------
# Usage
# ------------------------------------------------------------
INPUT_FOLDER = "./data/real/panda-3cam_realsense"
OUTPUT_FOLDER = "./data_flow/real/panda-3cam_realsense"

calculate_and_save_flow_with_history(INPUT_FOLDER, OUTPUT_FOLDER)