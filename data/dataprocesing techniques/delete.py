import os

folder = "./new"  # Replace with your folder path
allowed_extensions = {".jpg", ".jpeg", ".png"}  # Keep only image files

for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    if not os.path.isfile(file_path):
        continue
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        print(f"Deleting: {filename}")
        os.remove(file_path)

print("Cleanup complete!")
