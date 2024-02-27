import os
import argparse
import torch
import tomesd
from torch.cuda.amp import autocast
from torch.profiler import profile, record_function, ProfilerActivity
from scripts.demo.streamlit_helpers import *

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
):
    """sampling function with torch profiling"""
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
            # start profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True,
                         profile_memory=True,
                         with_stack=True) as prof:
                with record_function("model_inference"):
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

                    samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                    samples_x = model.decode_first_stage(samples_z)
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
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return samples


def load_model_from_config(config, ckpt=None, verbose=True):
    """an implementation without Streamlit"""
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


def init_without_st(version_dict, load_ckpt=True, load_filter=True, token_merging=True):
    state = dict()
    if not "model" in state:
        config = version_dict["config"]
        ckpt = version_dict["ckpt"]

        config = OmegaConf.load(config)
        model, msg = load_model_from_config(config, ckpt if load_ckpt else None)

        #TODO Token merging doesn't really work here
        if token_merging:
            tomesd.apply_patch(model, ratio=0.5)

        state["msg"] = msg
        state["model"] = model
        state["ckpt"] = ckpt if load_ckpt else None
        state["config"] = config
        if load_filter:
            state["filter"] = DeepFloydDataFiltering(verbose=False)
    return state


def main():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion Turbo.")
    parser.add_argument("--version", choices=VERSION2SPECS.keys(), default="SDXL-Turbo", help="Model version to use.")
    parser.add_argument("--n_steps", type=int, default=4, help="Number of sampling steps.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation.")
    parser.add_argument("--prompt",
                        default="A cinematic shot of a baby racoon wearing an intricate Italian priest robe.",
                        help="Prompt for the model.")
    parser.add_argument("--output", default="output_image.png", help="Output image path.")
    parser.add_argument("--tome", action=argparse.BooleanOptionalAction, help="Token merging")
    args = parser.parse_args()

    version_dict = VERSION2SPECS[args.version]
    if args.tome:
        print("Initializing the state WITH Token Merging")
    else:
        print("Initializing the state WITHOUT Token Merging")
    state = init_without_st(version_dict, load_ckpt=True, load_filter=True, token_merging=args.tome)

    if state["msg"]:
        print(state["msg"])
    model = state["model"]
    load_model(model)

    sampler = SubstepSampler(
        n_sample_steps=args.n_steps,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        ),
    )

    sampler.noise_sampler = SeededNoise(seed=args.seed)
    out = sample(
        model, sampler, H=512, W=512, seed=args.seed, prompt=args.prompt, filter=state.get("filter")
    )

    Image.fromarray(out[0]).save(args.output)
    print(f"Image saved to {args.output}")


if __name__ == "__main__":
    main()
