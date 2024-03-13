import argparse
import time
from tome import turbo_tome as tomesd
from collections import defaultdict
from torch.cuda.amp import autocast
from torch.profiler import profile, record_function
from util_debug import *
from diffusers import DiffusionPipeline
from turbo_prompt import PROMPT
from pytorch_lightning import seed_everything

VERSION2SPECS = {
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
    }
}


def init_sampler(
    args,
    key=1,
    specify_num_samples: bool = True,
    options: Optional[Dict[str, int]] = None,
):
    options = {} if options is None else options

    # only support 1 sample currently
    num_rows, num_cols = 1, 1
    if specify_num_samples:
        num_cols = 1

    discretization_config = get_discretization(args.discretization, options=options, key=key)
    guider_config = get_guider(args.guider, options=options, key=key)
    sampler = get_sampler(args.sampler, args.n_steps, discretization_config, guider_config, options, key=key)

    return sampler, num_rows, num_cols


def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: Optional[List] = None,
    force_cond_zero_embeddings: Optional[List] = None,
    batch2model_input: List = None,
    return_latents=False,
    filter=None,
    T=None,
    additional_batch_uc_fields=None,
    decoding_t=None,
    profile_visible=False
):
    force_uc_zero_embeddings = default(force_uc_zero_embeddings, [])
    additional_batch_uc_fields = default(additional_batch_uc_fields, [])

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                if T is not None:
                    num_samples = [num_samples, T]
                else:
                    num_samples = [num_samples]

                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                    T=T,
                    additional_batch_uc_fields=additional_batch_uc_fields,
                )

                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                )

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                        )
                    if k in ["crossattn", "concat"] and T is not None:
                        uc[k] = repeat(uc[k], "b ... -> b t ...", t=T)
                        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=T)
                        c[k] = repeat(c[k], "b ... -> b t ...", t=T)
                        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=T)

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c
                    )

                # torch profile On/Off
                if profile_visible:
                    with profile(activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]) as prof:
                        with record_function("0Model: Model_Sampling"):
                            samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                        model.en_and_decode_n_samples_a_time = decoding_t
                        with record_function("0Model: VAE_Decoder"):
                            samples_x = model.decode_first_stage(samples_z)

                else:
                    samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                    model.en_and_decode_n_samples_a_time = decoding_t
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

                if return_latents:
                    return samples, samples_z
                return samples, key_average


def multi_sampling(args, model, sampler, prompts, filter=None,
                   profile_visible=False, is_legacy=False, return_latents=False):
    """
    Profile multiple sampling processes.
    - profile_visible: If True, perform profiling only on the last iteration.
    """
    version_dict = VERSION2SPECS[args.version]
    W = version_dict['W']
    H = version_dict['H']
    C = version_dict['C']
    F = version_dict['f']

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }

    total_samples = []
    start = time.time()

    actual_iterations = 4 if profile_visible else len(prompts)

    for i in range(actual_iterations):
        profile_this_iter = profile_visible if i == actual_iterations - 1 else False

        single_prompt = prompts[0] if profile_visible and len(prompts) == 1 else prompts[i % len(prompts)]

        # no negative_prompt required for XL
        value_dict = init_embedder_options(
            get_unique_embedder_keys_from_conditioner(model.conditioner),
            init_dict,
            prompt=single_prompt,
            negative_prompt=None
        )

        # set to 1 for lower GPU burden
        num_samples = 1
        samples, key_average = do_sample(model, sampler, value_dict,
                            num_samples,
                            1024, 1024, C, F,
                            force_uc_zero_embeddings=["txt"] if not is_legacy else [],
                            return_latents=return_latents,
                            filter=filter,
                            profile_visible=profile_this_iter
                            )
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

        if tome_ratio > 0.0:
            model = tomesd.apply_patch(model,
                                       ratio=tome_ratio,
                                       max_downsample=2)

        state["msg"] = msg
        state["model"] = model
        state["ckpt"] = ckpt if load_ckpt else None
        state["config"] = config
        if load_filter:
            state["filter"] = DeepFloydDataFiltering(verbose=False)
    return state


def init_and_sampling(args, sampler, prompts, tome_ratio=0.0, profile_visible=True):
    seed_everything(args.seed)
    print(f"Initializing and sampling with TOME ratio = {tome_ratio} ...")
    state = init_without_st(VERSION2SPECS[args.version], load_ckpt=True, load_filter=True, tome_ratio=tome_ratio)
    model = state["model"]
    load_model(model)
    samples, total_runtime, key_average = multi_sampling(args, model, sampler, prompts,
                                              filter=state.get("filter"), profile_visible=profile_visible)
    # note that key average is not None only for last sample when profiling
    unload_model(model)
    return samples, total_runtime, key_average


def init_and_sampling_diffuser(prompts, iterations, tome_ratio=0.0):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                             torch_dtype=torch.float16,
                                             use_safetensors=True,
                                             variant="fp16").to("cuda")
    if tome_ratio > 0:
        pipe = tomesd.apply_patch(pipe, ratio=tome_ratio, max_downsample=2)

    total_samples = []
    start = time.time()
    for i in range(iterations):
        single_prompt = prompts[i]
        image = pipe(prompt=single_prompt).images
        total_samples.append(image)
    total_runtime = (time.time() - start)

    return total_samples, total_runtime


def main():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion XL base.")
    parser.add_argument("--version", choices=VERSION2SPECS.keys(), default="SDXL-base-1.0", help="Model version to use.")
    parser.add_argument("--n_steps", type=int, default=50, help="Number of sampling steps.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation.")
    parser.add_argument("--sampler", default="EulerAncestralSampler", help="Sampler configuration.")
    parser.add_argument("--discretization", default="LegacyDDPMDiscretization", help="Discretization configuration.")
    parser.add_argument("--guider", default="VanillaCFG", help="Guider configuration.")
    parser.add_argument("--diffuser", action=argparse.BooleanOptionalAction, help="Use Huggingface diffuser.")
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, help="Enable torch profiler.")
    parser.add_argument("--return_latent", action=argparse.BooleanOptionalAction, help="Return last stage latent variable.")
    parser.add_argument("--resolution", type=tuple, default=(1024, 1024), help="Resolution of the image")
    parser.add_argument("--tome_ratios", type=list, default=[0.0, 0.0, 0.25, 0.5, 0.75], help="Token merging ratio")
    args = parser.parse_args()

    def ddict():
        return defaultdict(ddict)
    runtimes = ddict()
    samples = ddict()
    set_lowvram_mode(True)

    tome_ratios = args.tome_ratios
    profile_visible = args.profile
    if profile_visible:
        assert args.diffuser is None, "Assertion: Unable to initialize profile with diffuser"
        print("Profiling with torch.profiler ...")
        model_sampling_cuda_time = {}
        total_cuda_time = {}
        total_cpu_time = {}


    # Use model source code
    if args.diffuser is None:
        sampler, _, _ = init_sampler(args)
        prompts = PROMPT

        for tome_ratio in tome_ratios:
            samples[tome_ratio], runtimes[tome_ratio], key_average = init_and_sampling(
                 args, sampler, prompts, tome_ratio=tome_ratio, profile_visible=profile_visible
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


    # TODO add a new branch named interaction


    # Use Huggingface Diffuser
    elif args.diffuser:
        prompts = PROMPT
        # tome loop
        for tome_ratio in tome_ratios:
            samples[tome_ratio], runtimes[tome_ratio] = init_and_sampling_diffuser(
                prompts, len(prompts), tome_ratio=tome_ratio
            )
            print(f"\t\tRuntime: {runtimes[tome_ratio]:.3f}")


    ## Print Results
    if runtimes[0] and not profile_visible:
        for tome_ratio, runtime in runtimes.items():
            no_tome_runtime = runtimes[0]
            time_perc = 100 * (no_tome_runtime - runtime) / no_tome_runtime
            print(
                f"ToMe ratio: {tome_ratio:.1f} -- runtime reduction: {time_perc:5.2f}%")

    save_samples_in_grids(args, samples, diffuser=args.diffuser)


if __name__ == "__main__":
    main()