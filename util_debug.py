from scripts.demo.streamlit_helpers import *


def preprocess_samples(samples):
    processed_samples = {}
    for tome_ratio, image_arrays in samples.items():
        images_list = [[Image.fromarray(img_array.squeeze())] for img_array in image_arrays]
        processed_samples[tome_ratio] = images_list
    return processed_samples


def images_to_grid(images, grid_size=None, save_path="output_grid.png"):
    # Flatten the list of lists to get a simple list of images
    flat_images = [img[0] for img in images]

    if grid_size is None:
        grid_cols = int(math.ceil(math.sqrt(len(flat_images))))
        grid_rows = int(math.ceil(len(flat_images) / grid_cols))
    else:
        grid_rows, grid_cols = grid_size

    img_width, img_height = flat_images[0].size
    grid_img = Image.new('RGB', size=(img_width * grid_cols, img_height * grid_rows))

    for index, img in enumerate(flat_images):
        row = index // grid_cols
        col = index % grid_cols
        grid_img.paste(img, box=(col * img_width, row * img_height))

    grid_img.save(save_path)
    print(f"Grid image saved at {save_path}")


def save_samples_in_grids(samples, diffuser):
    """
    Save images in samples dictionary to grid images.
    """
    for tome_ratio, images_list in samples.items():
        save_path = f"output_images_grid_tome_{tome_ratio}_diffuser_{diffuser}.png"
        images_to_grid(images_list, save_path=save_path)
        print(f"Saved images grid for ToMe ratio {tome_ratio} at: {save_path}")