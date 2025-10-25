import cv2
import os
import numpy as np
import asyncio

async def detect_blur_and_save(video_path, threshold=50, max_frames=None):
    """
    Async function to detect blur and extract frames from video
    Returns numpy array of frames for model processing
    """
   
    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    frames_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for blur detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance (sharpness score)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var >= threshold:
            saved_count += 1
            # filename = os.path.join(output_dir, f"frame_{frame_count}_score_{int(laplacian_var)}.jpg")
           # cv2.imwrite(filename, frame)
            
            # --- Preprocessing for the model ---
            # 1. Resize frame for model input
            resized_frame = cv2.resize(frame, (224, 224))

            # 2. Convert color from BGR (OpenCV default) to RGB (TensorFlow default)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # 3. Rescale pixel values to 0-1 and convert to float
            processed_frame = rgb_frame.astype('float32') / 255.0
            
            frames_list.append(processed_frame)

        frame_count += 1

        # Allow other tasks to run
        if frame_count % 10 == 0:
            await asyncio.sleep(0.001)
      
        if max_frames and saved_count >= max_frames:
            break

    cap.release()
    print(f"✅ Done! Saved {saved_count} sharp frames out of {frame_count} total frames.")
    
    # Convert list of processed frames to a single NumPy array
    if frames_list:
        # The shape will be (num_frames, 224, 224, 3), which is what model.predict expects
        frames_array = np.array(frames_list)
        return frames_array
    else:
        return np.array([])  # Return empty array if no frames found
