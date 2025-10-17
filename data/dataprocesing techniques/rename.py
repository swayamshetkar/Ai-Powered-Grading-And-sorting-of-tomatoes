import os

folder = "./texts"  # replace with your folder path
ext = ".txt"  # change if images are png or something else

files = sorted(os.listdir(folder))  # get all files sorted
count = 72  # start from 052

for f in files:
    old_path = os.path.join(folder, f)
    new_name = f"{count:03d}{ext}"  # 052.jpg, 053.jpg, ...
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)
    count += 1

print("Renaming complete!")
