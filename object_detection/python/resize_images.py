import os
import cv2

def resize_images():
    input_dir = input("Enter the input directory path: ").strip()
    width = int(input("Enter the target width: "))
    height = int(input("Enter the target height: "))

    # Validate input directory
    if not os.path.isdir(input_dir):
        print("Invalid input directory.")
        return

    # Create output directory
    output_dir = os.path.join(input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)

            # Read image
            img = cv2.imread(input_path)
            if img is None:
                print(f"Skipping unreadable image: {filename}")
                continue

            # Resize image
            resized_img = cv2.resize(img, (width, height))

            # Create output file path
            output_filename = "resize_" + filename
            output_path = os.path.join(output_dir, output_filename)

            # Save resized image
            cv2.imwrite(output_path, resized_img)
            print(f"Saved: {output_filename}")

    print(f"\nAll images resized and saved in: {output_dir}")

# Run the routine
resize_images()
