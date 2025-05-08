import os
from PIL import Image


def collect_images_from_folder(root_dir):
    """
    Collect all jpg images from the folder and its subfolders.

    :param root_dir: The root folder containing subfolders with jpg images.
    :return: A list of image file paths.
    """
    image_paths = []
    num = 0
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".jpg"):  # Find all .jpg files (case-insensitive)
                image_paths.append(os.path.join(subdir, file))
                num += 1
                if num == 100000:
                    return image_paths




def convert_and_save_images(image_paths, output_dir, prefix):
    """
    Convert images from jpg to png and save them with the specified naming convention.

    :param image_paths: List of image file paths to be converted.
    :param output_dir: Directory where converted images will be saved.
    :param prefix: Prefix for naming the images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        # Convert to PNG format
        img.save(os.path.join(output_dir, f"{prefix}_{i:07d}.png"))


def main():
    root_dir = 'sample'
    output_dir = 'lsun_train_output_dir'
    prefix = 'bedroom'

    # Collect all jpg images from the root folder and its subfolders
    image_paths = collect_images_from_folder(root_dir)

    # Print the number of images found
    print(f"Found {len(image_paths)} jpg images.")

    # Convert images to PNG and save them
    convert_and_save_images(image_paths, output_dir, prefix)


if __name__ == "__main__":
    main()
