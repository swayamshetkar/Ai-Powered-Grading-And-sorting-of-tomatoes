import os
import shutil
from train import train_model

TEMP_VERIFIED = "data/temp_verified"
PROCESSED_DIR = "data/processed"

def fine_tune_from_verified(temp_verified_dir: str = TEMP_VERIFIED):
    """
    Fine-tune using the verified images in temp_verified_dir.
    This function expects temp_verified_dir to contain class subfolders, or be a class-root.
    After successful fine-tuning, moves images into data/processed/train/<class>/ and clears temp_verified_dir.
    """
    if not os.path.exists(temp_verified_dir):
        print("No verified images directory found:", temp_verified_dir)
        return

    # count total images
    total_images = 0
    for root, dirs, files in os.walk(temp_verified_dir):
        for f in files:
            total_images += 1

    if total_images == 0:
        print("No new verified images to fine-tune.")
        return

    print(f"🔹 Found {total_images} verified images. Starting fine-tuning...")

    # Call train_model with finetune=True (train_model will handle a single-folder split)
    try:
        best_val = train_model(data_dir=temp_verified_dir, finetune=True)
        print("Fine-tuning finished. Best val acc:", best_val)
    except Exception as e:
        print("Fine-tuning failed:", e)
        return

    # Move verified images into processed/train/<class>
    dest_root = os.path.join(PROCESSED_DIR, "train")
    os.makedirs(dest_root, exist_ok=True)

    for root, dirs, files in os.walk(temp_verified_dir):
        rel_root = os.path.relpath(root, temp_verified_dir)
        # rel_root '.' means root dir; in that case we don't create extra subfolder unless files are in class folders
        for f in files:
            src_path = os.path.join(root, f)
            # If verified images were stored in class subfolders, preserve that. If not, put them into 'unknown' folder.
            if rel_root == ".":
                # no class folder present; place under 'unknown' class to avoid mixing
                dst_dir = os.path.join(dest_root, "unknown")
            else:
                dst_dir = os.path.join(dest_root, rel_root)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, f)
            try:
                shutil.move(src_path, dst_path)
            except Exception:
                # fallback to copy+remove
                shutil.copy2(src_path, dst_path)
                try:
                    os.remove(src_path)
                except Exception:
                    pass

    # Remove temp_verified_dir root if empty
    try:
        shutil.rmtree(temp_verified_dir)
        print("✅ Fine-tuning complete and new images merged into processed/train/. TEMP_VERIFIED cleared.")
    except Exception as e:
        print("Fine-tuning complete, but failed to remove temp directory:", e)

if __name__ == "__main__":
    fine_tune_from_verified()
