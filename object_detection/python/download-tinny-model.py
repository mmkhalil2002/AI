import urllib.request
import zipfile
import os

url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
filename = 'tiny-imagenet-200.zip'

# Download the file
print("Downloading Tiny ImageNet dataset...")
urllib.request.urlretrieve(url, filename)
print("Download complete.")

# Unzip the file
print("Extracting files...")
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall()
print("Extraction complete.")
