#This script prepares the test set by renaming images with secure random filenames, resizing them, and compiling rich metadata into a CSV file.


import os
import uuid
import random
from PIL import Image
import pandas as pd

# CONFIG 
input_root = "data/art_subset_1000/images"  # Orignal test set with "real/" and "fake/" folders 
output_root = "data/art_testset"            # finalized and prepared test set with image renamed and resized with metadata
target_size = (512, 512)                   # width, height in pixels
valid_exts = {".jpg", ".jpeg", ".png", ".webp"}  # supported formats
csv_path = "data/art_testset_metadata.csv"
source_metadata = "data/art_subset_1000/metadata.csv"

#  SETUP 
os.makedirs(output_root, exist_ok=True)

# Load source metadata 
print(f"Loading source metadata from {source_metadata}")
if os.path.exists(source_metadata):
    source_meta_df = pd.read_csv(source_metadata)
    print(f"Loaded {len(source_meta_df)} metadata records")
else:
    print(f"Warning: Source metadata file not found at {source_metadata}")
    source_meta_df = pd.DataFrame()

# Collect all image information first
all_images = []

# SCAN FOLDERS AND COLLECT INFO 
for label in ["real", "fake"]:
    folder = os.path.join(input_root, label)
    if not os.path.exists(folder):
        print(f"Warning: Folder missing: {folder}")
        continue

    print(f"Scanning {folder}...")

    # Check subdirectories
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for fname in os.listdir(subdir_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_exts:
                continue

            src_path = os.path.join(subdir_path, fname)

            # Find metadata for this image
            meta_row = None
            if not source_meta_df.empty:
                # Try to match by filename
                matches = source_meta_df[source_meta_df['image_filename'] == fname]
                if not matches.empty:
                    meta_row = matches.iloc[0]
                else:
                    # Try to match by file path components
                    matches = source_meta_df[source_meta_df['file_path'].str.contains(fname.replace(ext, ''), na=False)]
                    if not matches.empty:
                        meta_row = matches.iloc[0]

            # Determine true label
            true_label = "human" if label == "real" else "ai"

            # Extract generator model and art style from metadata
            if meta_row is not None:
                generator_model = meta_row.get('generator_model', 'unknown')
                art_style = meta_row.get('art_style', 'unknown')
            else:
                generator_model = 'human' if true_label == "human" else 'unknown'
                art_style = 'unknown'

            all_images.append({
                'src_path': src_path,
                'original_filename': fname,
                'true_label': true_label,
                'generator_model': generator_model,
                'art_style': art_style,
                'subfolder': subdir
            })

print(f"Found {len(all_images)} images total")

# RANDOMIZE ORDER AND ASSIGN SECURE NAMES
print("Shuffling images and generating secure filenames...")
random.shuffle(all_images)  # Randomize order completely

records = []
processed_count = 0

for i, img_info in enumerate(all_images):
    try:
        # Generate secure random filename
        secure_filename = f"img_{i+1:06d}.jpg"  # Sequential but randomized order
        new_id = f"img_{i+1:06d}"

        # Load and process image
        img = Image.open(img_info['src_path']).convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Save with secure filename
        dst_path = os.path.join(output_root, secure_filename)
        img.save(dst_path, quality=95)

        # Create metadata record with rich information
        records.append({
            "image_id": new_id,
            "image_filename": secure_filename,
            "true_label": img_info['true_label'],
            "generator_model": img_info['generator_model'],
            "art_style": img_info['art_style'],
            "subfolder": img_info['subfolder'],
            "original_filename": img_info['original_filename']
        })

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(all_images)} images...")

    except Exception as e:
        print(f"Error processing {img_info['original_filename']}: {e}")

# SAVE METADATA
df = pd.DataFrame(records)

# Show distribution
print(f"\nDataset Distribution:")
print(f"Total images: {len(df)}")
print(f"Real images: {len(df[df['true_label'] == 'human'])}")
print(f"AI images: {len(df[df['true_label'] == 'ai'])}")

if 'generator_model' in df.columns:
    print(f"\nGenerator Distribution:")
    print(df['generator_model'].value_counts())

if 'art_style' in df.columns:
    print(f"\nArt Style Distribution:")
    print(df['art_style'].value_counts())

# Save metadata
df.to_csv(csv_path, index=False)
print(f"\nDone! {len(records)} images processed and saved to {output_root}")
print(f"Metadata written to {csv_path}")
print(f"Secure naming: Images named img_XXXXXX.jpg with randomized order")
print(f"Zero filename-based data leakage!")
