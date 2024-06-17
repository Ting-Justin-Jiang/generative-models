from scripts.demo.streamlit_helpers import *
import os
import random
import torch.nn.functional as F
from torchvision import transforms
from functools import partial
from torchmetrics.functional.multimodal import clip_score

def load_model(model):
    model.cuda()


def unload_model(model):
    model.cpu()
    del model
    torch.cuda.empty_cache()


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


def preprocess_samples(samples):
    processed_samples = {}
    for tome_ratio, image_arrays in samples.items():
        images_list = [[Image.fromarray(img_array.squeeze())] for img_array in image_arrays]
        processed_samples[tome_ratio] = images_list
    return processed_samples


def images_to_grid(images, grid_size=None, save_path="output_grid.png"):
    # Flatten the list of lists to get a simple list of images
    flat_images = [img[0] for img in images]
    if len(flat_images) > 64:
        saved_images = flat_images[:64]
    else:
        saved_images = flat_images

    if grid_size is None:
        grid_cols = int(math.ceil(math.sqrt(len(saved_images))))
        grid_rows = int(math.ceil(len(saved_images) / grid_cols))
    else:
        grid_rows, grid_cols = grid_size

    img_width, img_height = saved_images[0].size
    grid_img = Image.new('RGB', size=(img_width * grid_cols, img_height * grid_rows))

    for index, img in enumerate(saved_images):
        row = index // grid_cols
        col = index % grid_cols
        grid_img.paste(img, box=(col * img_width, row * img_height))

    grid_img.save(save_path)
    print(f"Grid image saved at {save_path}")
    return flat_images


def save_and_evaluate(args, samples, prompts):
    transform = transforms.ToTensor()
    os.makedirs(args.experiment_folder, exist_ok=True)
    for tome_ratio, images_list in samples.items():
        save_path = (f"{args.experiment_folder}/{args.version}_merge_ratio_{tome_ratio}_"
                     f"unmerge_residual_{args.unmerge_residual}_push_unmerged_{args.push_unmerged}.png")

        images_list = images_to_grid(images_list, save_path=save_path)
        images_tensor = np.transpose(np.stack([transform(img) for img in images_list]), (0, 2, 3, 1))

        print(f"Saved images grid for ToMe ratio {tome_ratio} at: {save_path}")
        clip_score = calculate_clip_score(images_tensor, prompts)
        print(f"ToMe ratio: {tome_ratio} CLIP score: {clip_score}")


def calculate_clip_score(images, prompts):
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    images_int = (images * 255).astype("uint8")
    images_clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(images_clip_score), 4)


def get_discretization(discretization, options, key=1):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif discretization == "EDMDiscretization":
        sigma_min = options.get("sigma_min", 0.03)
        sigma_max = options.get("sigma_max", 14.61)
        rho = options.get("rho", 3.0)

        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
            },
        }
    return discretization_config


def get_guider(guider, options, key):
    additional_guider_kwargs = options.pop("additional_guider_kwargs", {})

    if guider == "IdentityGuider":
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif guider == "VanillaCFG":
        scale = options.get("cfg_scale", 7.0)
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {
                "scale": scale,
                **additional_guider_kwargs,
            },
        }
    elif guider == "LinearPredictionGuider":
        max_scale = options.get("max_cfg_scale", 1.5)
        min_scale = options.get("min_cfg_scale", 1.0)

        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.LinearPredictionGuider",
            "params": {
                "max_scale": max_scale,
                "min_scale": min_scale,
                "num_frames": options.get("num_frames", 10),
                **additional_guider_kwargs,
            },
        }
    else:
        raise NotImplementedError

    return guider_config


def get_sampler(sampler_name, steps, discretization_config, guider_config, options, key=1):
    # default values for sampler
    s_churn = options.get(f"s_churn_{key}", 0.0)
    s_tmin = options.get(f"s_tmin_{key}", 0.0)
    s_tmax = options.get(f"s_tmax_{key}", 999.0)
    s_noise = options.get(f"s_noise_{key}", 1.0)
    eta = options.get("eta", 1.0)
    order = options.get("order", 4)

    if sampler_name in ["EulerEDMSampler", "HeunEDMSampler"]:
        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "HeunEDMSampler":
            sampler = HeunEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
    elif sampler_name in ["EulerAncestralSampler", "DPMPP2SAncestralSampler"]:
        if sampler_name == "EulerAncestralSampler":
            sampler = EulerAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "DPMPP2SAncestralSampler":
            sampler = DPMPP2SAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
    elif sampler_name == "DPMPP2MSampler":
        sampler = DPMPP2MSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name == "LinearMultistepSampler":
        sampler = LinearMultistepSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            order=order,
            verbose=True,
        )
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler
