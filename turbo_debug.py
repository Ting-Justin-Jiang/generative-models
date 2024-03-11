import os
import argparse
import torch
import time
import turbo_tome as tomesd
from collections import defaultdict
from torch.cuda.amp import autocast
from torch.profiler import profile, record_function, ProfilerActivity
from util_debug import *
from diffusers import AutoPipelineForText2Image
from turbo_prompt import PROMPT

#####################################################################################
# helper functions from original turbo demo
#####################################################################################
VERSION2SPECS = {
    "SDXL-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_turbo_1.0.safetensors",
    },
    "SD-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/sd_turbo.safetensors",
    },
}


class SubstepSampler(EulerAncestralSampler):
    def __init__(self, n_sample_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        sigmas = sigmas[
            self.steps_subset[: self.n_sample_steps] + self.steps_subset[-1:]
            ]
        uc = cond
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc


def load_model(model):
    model.cuda()


def seeded_randn(shape, seed):
    randn = np.random.RandomState(seed).randn(*shape)
    randn = torch.from_numpy(randn).to(device="cuda", dtype=torch.float32)
    return randn


class SeededNoise:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, x):
        self.seed = self.seed + 1
        return seeded_randn(x.shape, self.seed)


def init_embedder_options(keys, init_dict, prompt=None, negative_prompt=None):
    value_dict = {}
    for key in keys:
        if key == "txt":
            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = ""

        if key == "original_size_as_tuple":
            orig_width = init_dict["orig_width"]
            orig_height = init_dict["orig_height"]

            value_dict["orig_width"] = orig_width
            value_dict["orig_height"] = orig_height

        if key == "crop_coords_top_left":
            crop_coord_top = 0
            crop_coord_left = 0

            value_dict["crop_coords_top"] = crop_coord_top
            value_dict["crop_coords_left"] = crop_coord_left

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = 6.0
            value_dict["negative_aesthetic_score"] = 2.5

        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]

    return value_dict

#####################################################################################
# remove streamlit for easy debugging
#####################################################################################
def sample(
        model,
        sampler,
        prompt="A lush garden with oversized flowers and vibrant colors, inhabited by miniature animals.",
        H=1024,
        W=1024,
        seed=0,
        filter=None,
        profile_visible=False
):
    """
    sampling function with torch profiling
    """
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    value_dict = init_embedder_options(
        keys=get_unique_embedder_keys_from_conditioner(model.conditioner),
        init_dict={
            "orig_width": W,
            "orig_height": H,
            "target_width": W,
            "target_height": H,
        },
        prompt=prompt,
    )

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1],
            )
            c = model.conditioner(batch)
            uc = None
            randn = seeded_randn(shape, seed)

            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model,
                    input,
                    sigma,
                    c,
                )

            # torch profile On/Off
            if profile_visible:
                with profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]) as prof:

                    with record_function("0Model: Model_Sampling"):
                        samples_z = sampler(denoiser, randn, cond=c, uc=uc)

                    with record_function("0Model: VAE_Decoder"):
                        samples_x = model.decode_first_stage(samples_z)
            else:
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                samples_x = model.decode_first_stage(samples_z)

            key_average = None
            if profile_visible:
                key_average = prof.key_averages()
                print(key_average.table(sort_by="cuda_time_total"))


            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
            if filter is not None:
                samples = filter(samples)
            samples = (
                (255 * samples)
                .to(dtype=torch.uint8)
                .permute(0, 2, 3, 1)
                .detach()
                .cpu()
                .numpy()
            )
    return samples, key_average


def multi_sampling(model, sampler, seed, prompts, iterations, filter=None, profile_visible=False):
    """
    Profile multiple sampling processes.
    - profile_visible: If True, perform profiling only on the last iteration.
    """
    total_samples = []
    start = time.time()

    # TODO probably need a more averaged profiling
    actual_iterations = 2 if profile_visible else iterations

    for i in range(actual_iterations):
        profile_this_iter = profile_visible if i == actual_iterations - 1 else False

        single_prompt = prompts[0] if profile_visible and len(prompts) == 1 else prompts[i % len(prompts)]
        samples, key_average = sample(model, sampler, H=512, W=512, seed=seed,
                         prompt=single_prompt, filter=filter, profile_visible=profile_this_iter)
        total_samples.append(samples)

    total_runtime = (time.time() - start)
    return total_samples, total_runtime, key_average


def load_model_from_config(config, ckpt=None, verbose=True):
    """
    an implementation without Streamlit
    """
    model = instantiate_from_config(config.model)
    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        if ckpt.endswith("ckpt"):
            pl_sd = torch.load(ckpt, map_location="cpu")
            if "global_step" in pl_sd:
                global_step = pl_sd["global_step"]
                st.info(f"loaded ckpt from global step {global_step}")
                print(f"Global Step: {pl_sd['global_step']}")
            sd = pl_sd["state_dict"]
        elif ckpt.endswith("safetensors"):
            sd = load_safetensors(ckpt)
        else:
            raise NotImplementedError("Checkpoint file format not supported.")

        msg = None

        m, u = model.load_state_dict(sd, strict=False)

        if len(m) > 0 and verbose:
            print("Missing keys in state dict:")
            print(m)
        if len(u) > 0 and verbose:
            print("Unexpected keys in state dict:")
            print(u)
    else:
        msg = None

    model = initial_model_load(model)
    model.eval()
    return model, msg


def init_without_st(version_dict, load_ckpt=True, load_filter=True, tome_ratio=0.5):
    state = dict()
    if not "model" in state:
        config = version_dict["config"]
        ckpt = version_dict["ckpt"]

        config = OmegaConf.load(config)
        model, msg = load_model_from_config(config, ckpt if load_ckpt else None)

        if tome_ratio > 0:
            model = tomesd.apply_patch(model, ratio=tome_ratio)

        state["msg"] = msg
        state["model"] = model
        state["ckpt"] = ckpt if load_ckpt else None
        state["config"] = config
        if load_filter:
            state["filter"] = DeepFloydDataFiltering(verbose=False)
    return state


def init_and_sampling(version_dict, sampler, seed, prompt, iterations, tome_ratio=0.0, profile_visible=True):
    print(f"Initializing and sampling with TOME ratio = {tome_ratio} ...")
    state = init_without_st(version_dict, load_ckpt=True, load_filter=True, tome_ratio=tome_ratio)
    model = state["model"]
    load_model(model)
    samples, total_runtime, key_average = multi_sampling(model, sampler, seed, prompt, iterations,
                                              filter=state.get("filter"), profile_visible=profile_visible)

    # note that key average is not None only for last sample when profiling
    return samples, total_runtime, key_average


def init_and_sampling_diffuser(n_steps, prompt, iterations, tome_ratio=0.0):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo",
                                                     height=512, width=512,
                                                     torch_dtype=torch.float32).to("cuda")
    if tome_ratio > 0:
        pipe = tomesd.apply_patch(pipe, ratio=tome_ratio)
    pipe.set_progress_bar_config(disable=True)

    total_samples = []
    start = time.time()
    for i in range(iterations):
        single_prompt = prompt[i]
        image = pipe(prompt=single_prompt, num_inference_steps=n_steps, guidance_scale=0.0).images
        total_samples.append(image)
    total_runtime = (time.time() - start)

    return total_samples, total_runtime


def main():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion Turbo.")
    parser.add_argument("--version", choices=VERSION2SPECS.keys(), default="SDXL-Turbo", help="Model version to use.")
    parser.add_argument("--n_steps", type=int, default=4, help="Number of sampling steps.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generation.")
    parser.add_argument("--output", default="output_image.png", help="Output image path.")
    parser.add_argument("--diffuser", action=argparse.BooleanOptionalAction, help="Use Huggingface diffuser")
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, help="Enable torch profiler")
    args = parser.parse_args()

    def ddict():
        return defaultdict(ddict)
    runtimes = ddict()
    samples = ddict()
    tome_ratios = [0.0]

    profile_visible = args.profile
    if profile_visible:
        assert args.diffuser is None, "Assertion: Unable to initialize profile with diffuser"
        print("Profiling with torch.profiler ...")
        model_sampling_cuda_time = {}
        total_cuda_time = {}
        total_cpu_time = {}


    # Use model source code
    if args.diffuser is None:
        version_dict = VERSION2SPECS[args.version]
        sampler = SubstepSampler(n_sample_steps=args.n_steps, num_steps=1000, eta=1.0, discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"))
        sampler.noise_sampler = SeededNoise(seed=args.seed)
        prompts = PROMPT

        for tome_ratio in tome_ratios:
            samples[tome_ratio], runtimes[tome_ratio], key_average = init_and_sampling(
                 version_dict, sampler, args.seed, prompts, len(prompts), tome_ratio=tome_ratio, profile_visible=profile_visible
            )
            if profile_visible:
                # TODO plotting function
                model_sampling_cuda_time = merge_dictionary(key_average,
                                                            ["0Model: Model_Sampling"],
                                                            "cuda_time_total",
                                                            model_sampling_cuda_time
                                                            )
                total_cuda_time = merge_dictionary(key_average,
                                                   ["0Model: Model_Sampling", "0Model: VAE_Decoder"],
                                                   "cuda_time_total",
                                                   total_cuda_time
                                                   )
                total_cpu_time = merge_dictionary(key_average,
                                                   ["self_cpu_time_total"],
                                                   "cuda_time_total",
                                                   total_cpu_time
                                                   )
                print(model_sampling_cuda_time)
                print(total_cuda_time)
                print(total_cpu_time)

            else:
                print(f"\t\tRuntime: {runtimes[tome_ratio]:.3f}")
        samples = preprocess_samples(samples)


    # Use Huggingface Diffuser
    elif args.diffuser:
        prompts = PROMPT
        # tome loop
        for tome_ratio in tome_ratios:
            samples[tome_ratio], runtimes[tome_ratio] = init_and_sampling_diffuser(
                args.n_steps, prompts, len(prompts), tome_ratio=tome_ratio
            )
            print(f"\t\tRuntime: {runtimes[tome_ratio]:.3f}")


    ## Print Results
    if runtimes[0] and not profile_visible:
        for tome_ratio, runtime in runtimes.items():
            no_tome_runtime = runtimes[0]
            time_perc = 100 * (no_tome_runtime - runtime) / no_tome_runtime
            print(
                f"ToMe ratio: {tome_ratio:.1f} -- runtime reduction: {time_perc:5.2f}%")

    save_samples_in_grids(samples, diffuser=args.diffuser)


if __name__ == "__main__":
    main()
