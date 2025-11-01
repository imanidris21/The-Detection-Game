import os
from PIL import Image
import pandas as pd

# --- CONFIG ---
input_root = "data/images"          # where your "real/" and "fake/" folders live
output_root = "data/images_resized" # where renamed, resized images will be saved
target_size = (512, 512)       # width, height
valid_exts = {".jpg", ".jpeg", ".png", ".webp"}  # supported formats
csv_path = "data/images_metadata.csv"

# --- SETUP ---
os.makedirs(output_root, exist_ok=True)

records = []
counter_human = 1
counter_ai = 1

# --- SCAN FOLDERS ---
for label in ["real", "fake"]:
    folder = os.path.join(input_root, label)
    if not os.path.exists(folder):
        print(f"⚠️ Folder missing: {folder}")
        continue

    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_exts:
            continue

        src_path = os.path.join(folder, fname)
        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize(target_size, Image.Resampling.LANCZOS)

            if label == "real":
                new_id = f"human_{counter_human:04d}"
                counter_human += 1
                true_label = "human"
            else:
                new_id = f"ai_{counter_ai:04d}"
                counter_ai += 1
                true_label = "ai"

            new_name = f"{new_id}.jpg"
            dst_path = os.path.join(output_root, new_name)
            img.save(dst_path, quality=95)

            # Add to metadata
            records.append({
                "image_id": new_id,
                "image_filename": new_name,
                "true_label": true_label,
                "style": ""  # optional, you can fill later
            })
        except Exception as e:
            print(f"Error processing {fname}: {e}")

# --- SAVE METADATA ---
df = pd.DataFrame(records)
df.to_csv(csv_path, index=False)
print(f"✅ Done! {len(records)} images processed and saved to {output_root}")
print(f"📄 Metadata written to {csv_path}")
