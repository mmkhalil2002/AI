import urllib.request
import zipfile
import os
import shutil

url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
filename = 'tiny-imagenet-200.zip'
extract_dir = 'tiny-imagenet-200'

# Download the file
print("Downloading Tiny ImageNet dataset...")
urllib.request.urlretrieve(url, filename)
print("Download complete.")

# Unzip the file
print("Extracting files...")
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall()
print("Extraction complete.")

# Path to the 'train' directory
train_dir = os.path.join(extract_dir, 'train')

# Iterate over each class directory in 'train'
for class_dir in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_dir)
    images_path = os.path.join(class_path, 'images')

    # Check if 'images' directory exists
    if os.path.isdir(images_path):
        # Move all files from 'images' to the parent class directory
        for file_name in os.listdir(images_path):
            src_file = os.path.join(images_path, file_name)
            dest_file = os.path.join(class_path, file_name)
            shutil.move(src_file, dest_file)

        # Remove the now-empty 'images' directory
        os.rmdir(images_path)

    # Remove any files ending with '_boxes' in the class directory
    for file_name in os.listdir(class_path):
        if file_name.endswith('_boxes'):
            file_to_remove = os.path.join(class_path, file_name)
            os.remove(file_to_remove)

