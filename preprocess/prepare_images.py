import argparse
import os
import glob
import shutil


def main():
    parser = argparse.ArgumentParser(description="Copy and rename images.")
    parser.add_argument("--input_path", type=str, help="Path to the input directory")
    parser.add_argument("--output_path", type=str, help="Path to the output directory")
    args = parser.parse_args()

    # Define the input pattern to search for image files
    input_pattern = os.path.join(args.input_path, "images", "image_*", "*.png")

    for file_path in glob.glob(input_pattern):
        parts = file_path.split(os.sep)
        
        if len(parts) < 4:
            continue

        # Extract image number from the directory name
        image_num = int(parts[-2].split('_')[1])  # Extracts %02d format
        
        # Extract frame number from the file name
        frame_num = int(os.path.splitext(parts[-1])[0])  # Extracts %06d format

        # Convert frame number to %04d format
        new_frame_num = f"{frame_num:04d}"
        
        # Define new directory and file path
        new_dir = os.path.join(args.output_path, "images", f"image_{image_num:02d}")
        new_path = os.path.join(new_dir, f"{new_frame_num}.png")

        # Create the new directory if it doesn't exist
        os.makedirs(new_dir, exist_ok=True)
        
        # Copy the file to the new location with the updated name
        shutil.copy2(file_path, new_path)
        print(f"Copied: {file_path} -> {new_path}")


if __name__ == "__main__":
    main()
