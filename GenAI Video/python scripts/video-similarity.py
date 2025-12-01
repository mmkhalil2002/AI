
#
# pip install opencv-python numpy scikit-image
#
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_videos(video1_path, video2_path, frame_sample_rate=30):
    """
    Compare two videos based on visual similarity using SSIM.

    Parameters
    ----------
    video1_path : str
        Path to the first video file.
    video2_path : str
        Path to the second video file.
    frame_sample_rate : int
        Compare every Nth frame for efficiency (default = 30).

    Returns
    -------
    float
        Average SSIM score between 0 and 1.
        - 1.0 = identical frames
        - 0.0 = completely different
    """

    # Open both video files
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Error: Unable to open one or both video files.")
        return None

    total_ssim = 0
    compared_frames = 0

    frame_idx = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Stop when either video ends
        if not ret1 or not ret2:
            break

        # Compare every Nth frame
        if frame_idx % frame_sample_rate == 0:
            # Resize both frames to same dimensions for fair comparison
            frame1 = cv2.resize(frame1, (256, 256))
            frame2 = cv2.resize(frame2, (256, 256))

            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Compute the Structural Similarity Index (SSIM)
            score, _ = ssim(gray1, gray2, full=True)
            total_ssim += score
            compared_frames += 1

        frame_idx += 1

    cap1.release()
    cap2.release()

    if compared_frames == 0:
        print("⚠️ No frames compared.")
        return None

    avg_ssim = total_ssim / compared_frames
    print(f"✅ Average visual similarity (SSIM): {avg_ssim:.3f}")
    return avg_ssim


# Example usage:
if __name__ == "__main__":
    # Replace these with your actual video file paths
    video1 = "path/to/video1.mp4"
    video2 = "path/to/video2.mp4"

    similarity = compare_videos(video1, video2)
    if similarity is not None:
        if similarity > 0.90:
            print("🎬 Videos are visually almost identical.")
        elif similarity > 0.70:
            print("🎞️ Videos are visually similar.")
        else:
            print("📉 Videos look quite different.")
