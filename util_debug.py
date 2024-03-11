from scripts.demo.streamlit_helpers import *
import random
import torch.nn.functional as F


def extract_info(events, keywords, column):
    result = {}
    if "self_cpu_time_total" in keywords:
        result['self_cpu_time_total'] = events.self_cpu_time_total
    for event in events:
        if event.key in keywords:
            value = getattr(event, column, None)
            if value is not None:
                result[event.key] = value
    return result


def merge_dictionary(events, keywords, column, merged_dict):
    """
    Create a merged dictionary with module name as key and a py list of profiling results as value
    """
    extracted_info = extract_info(events, keywords, column)
    for key, value in extracted_info.items():
        if key in merged_dict:
            merged_dict[key].append(value)
        else:
            merged_dict[key] = [value]

    return merged_dict


def clamp(x: int, min: int, max: int) -> int:
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x

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


def get_reward(args, logits, label):
    """ given logits, calculate reward score for interaction computation
    Input:
        args: args.softmax_type determines which type of score to compute the interaction
            - normal: use log p, p is the probability of the {label} class
            - modified: use log p/(1-p), p is the probability of the {label} class
            - yi: use logits the {label} class
        logits: (N,num_class) tensor, a batch of logits before the softmax layer
        label: (1,) tensor, ground truth label
    Return:
        v: (N,) tensor, reward score
    """
    if args.softmax_type == "normal": # log p
        v = F.log_softmax(logits, dim=1)[:, label[0]]
    elif args.softmax_type == "modified": # log p/(1-p)
        v = logits[:, label[0]] - torch.logsumexp(logits[:, np.arange(args.class_number) != label[0].item()],dim=1)
    elif args.softmax_type == "yi": # logits
        v = logits[:, label[0]]
    else:
        raise Exception(f"softmax type [{args.softmax_type}] not implemented")
    return v


def gen_pairs(grid_size: int, pair_num: int, stride: int = 1) -> np.ndarray:
    """
    Input:
        grid_size: int, the image is partitioned to grid_size * grid_size patches. Each patch is considered as a player.
        pair_num: int, how many (i,j) pairs to sample for one image
        stride: int, j should be sampled in a neighborhood of i. stride is the radius of the neighborhood.
            e.g. if stride=1, then j should be sampled from the 8 neighbors around i
                if stride=2, then j should be sampled from the 24 neighbors around i
    Return:
        total_pairs: (pair_num,2) array, sampled (i,j) pairs
    """

    neighbors = [(i, j) for i in range(-stride, stride + 1)
                 for j in range(-stride, stride + 1)
                 if
                 i != 0 or j != 0]

    total_pairs = []
    for _ in range(pair_num):
        while True:
            x1 = np.random.randint(0, grid_size)
            y1 = np.random.randint(0, grid_size)
            point1 = x1 * grid_size + y1

            neighbor = random.choice(neighbors)
            x2 = clamp(x1 + neighbor[0], 0, grid_size - 1)
            y2 = clamp(y1 + neighbor[1], 0, grid_size - 1)
            point2 = x2 * grid_size + y2

            if point1 == point2:
                continue

            if [point1, point2] in total_pairs or [point2, point1] in total_pairs:
                continue
            else:
                total_pairs.append(list([point1, point2]))
                break

    return np.array(total_pairs)
