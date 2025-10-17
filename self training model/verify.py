import os
import shutil
from PIL import Image, UnidentifiedImageError

NEW_ENTRIES = "data/new_entries"
TEMP_VERIFIED = "data/temp_verified"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

os.makedirs(TEMP_VERIFIED, exist_ok=True)
os.makedirs(NEW_ENTRIES, exist_ok=True)

def is_allowed(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def safe_move(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        shutil.move(src, dst)
    except Exception:
        # fallback to copy+remove if move across filesystems fails
        shutil.copy2(src, dst)
        os.remove(src)

def verify_and_move():
    """
    Walk NEW_ENTRIES (including class subfolders), verify images, and move valid ones to TEMP_VERIFIED
    preserving relative subfolder structure.
    Invalid or unreadable files are removed.
    """
    any_processed = False
    for root, dirs, files in os.walk(NEW_ENTRIES):
        # ignore directories themselves
        for f in files:
            if not is_allowed(f):
                # remove unsupported files
                try:
                    os.remove(os.path.join(root, f))
                    print(f"Removed unsupported file: {os.path.join(root, f)}")
                except Exception:
                    pass
                continue

            src_path = os.path.join(root, f)
            # compute relative path with respect to NEW_ENTRIES
            rel_path = os.path.relpath(src_path, NEW_ENTRIES)
            dst_path = os.path.join(TEMP_VERIFIED, rel_path)

            try:
                # Use PIL to verify file integrity
                with Image.open(src_path) as img:
                    img.verify()  # will raise if file is truncated/corrupt
                # Re-open to ensure readable and convert to RGB (some formats need this)
                with Image.open(src_path) as img:
                    img.convert("RGB")
                # Move to verified location preserving subfolder structure
                safe_move(src_path, dst_path)
                print(f"✅ Verified: {rel_path}")
                any_processed = True
            except (UnidentifiedImageError, OSError, ValueError) as e:
                # corrupted/unreadable -> remove
                try:
                    os.remove(src_path)
                    print(f"❌ Rejected (corrupt/unreadable): {rel_path}")
                except Exception:
                    print(f"❌ Rejected and removal failed: {rel_path}")
            except Exception as e:
                # unexpected exception: attempt to remove to avoid repeated failures
                try:
                    os.remove(src_path)
                except Exception:
                    pass
                print(f"❌ Rejected (error): {rel_path} -> {e}")

    if not any_processed:
        print("No valid new images were found in NEW_ENTRIES.")
    else:
        print("Verification pass complete. Verified images are in:", TEMP_VERIFIED)

if __name__ == "__main__":
    verify_and_move()
