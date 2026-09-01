import os
import gc
import time
import ctypes
import warnings
import hashlib
import json
import math
import random
import re
from pathlib import Path

# Optional: make tokenizer/transformers quieter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import numpy as np

from PIL import Image, ImageDraw, ImageFont
try:
    from IPython.display import clear_output
except Exception:
    def clear_output(*args, **kwargs):
        return None
from tqdm.auto import tqdm
from transformers import pipeline
from diffusers import (
    AutoPipelineForText2Image,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
)

print("Imports successful.")


# -------------------------
# Startup cleanup
# -------------------------
def safe_startup_cleanup():
    try:
        plt.close("all")
    except Exception:
        pass

    gc.collect()
    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except Exception:
                pass
            torch.cuda.synchronize()

            print("CUDA cleanup done.")
            print(f"Device: {torch.cuda.get_device_name(0)}")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        else:
            print("CUDA is not available.")
    except Exception as e:
        print("Torch cleanup skipped:", e)

    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        print("malloc_trim(0) called.")
    except Exception:
        pass

    time.sleep(1)

    print("\n=== nvidia-smi ===")
    ret = os.system("nvidia-smi")
    if ret != 0:
        print("nvidia-smi not available in this environment.")



safe_startup_cleanup()

# -------------------------
# Determinism settings
# -------------------------
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

# -------------------------
# Config
# -------------------------
SEED = 42
BASE_MODEL_ID = "stabilityai/sdxl-turbo"
JUDGE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIFFUSION_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

WIDTH = 512
HEIGHT = 512

MODEL_TAG = BASE_MODEL_ID.split("/")[-1].replace(".", "_").replace("-", "_")
RUN_NAME = f"terrarium_butterfly_pref_mcmc_{MODEL_TAG}_joint"

# Master switch: when False, no files are written to disk at all
# (per-step images, baseline images, figures, JSON logs).
# Plots are still shown interactively when PLOT_EVERY_STEP=True.
SAVE_IMAGES = True

# Portable output directory
OUTPUT_DIR = Path("./files")
CASE_DIR = OUTPUT_DIR / RUN_NAME
if SAVE_IMAGES:
    CASE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Scene prompt
# -------------------------
GENERATOR_PROMPT = (
    "A macro photograph of a glass terrarium interior: "
    "a bright orange salamander climbing on a piece of grey driftwood, "
    "a small turquoise butterfly flying in the terrarium, "
    "and a red mushroom with white spots growing at the base near soil. "
)

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, extra limbs, "
    "watermark, text, cropped, out of frame, plastic toy, cartoon"
)

JUDGE_NAMES = ["SalamanderCurator", "ButterflyResearcher", "TerrariumEditor"]

# -------------------------
# Generator settings
# -------------------------
GEN_STEPS_TURBO = 2
GUIDANCE_SCALE_TURBO = 0.0

GEN_STEPS_STANDARD = 20
GUIDANCE_SCALE_STANDARD = 7.5
DDIM_ETA_STANDARD = 0.0

GEN_STEPS_LCM = 4
GUIDANCE_SCALE_LCM = 1.0

# -------------------------
# Chain settings
# -------------------------
CHAIN_LENGTH = 3000
NUM_VOTES = 9
SAVE_LAST_K = 200
NUM_RANDOM_BASELINE = 64
NUM_TRAJECTORY_SNAPSHOTS = 8

# Proposal q(z' | z): fixed mixture, all branches reversible wrt N(0, I)
LOCAL_BETA = 0.08
WIDE_BETA = 0.25

LOCAL_PROB = 0.6
WIDE_PROB = 0.3
REFRESH_PROB = 0.1

if not math.isclose(LOCAL_PROB + WIDE_PROB + REFRESH_PROB, 1.0, abs_tol=1e-12):
    raise ValueError("Proposal probabilities must sum to 1.")

# Judge decoding
JUDGE_MAX_NEW_TOKENS = 2
JUDGE_DO_SAMPLE = True
JUDGE_TEMPERATURE = 0.3
JUDGE_TOP_P = 0.8
JUDGE_REPETITION_PENALTY = 1.02

# Judge image size
JUDGE_IMAGE_SIZE = 512

# Step plotting
PLOT_EVERY_STEP = False
SAVE_STEP_PLOTS = False
CLEAR_PREVIOUS_STEP_OUTPUT = False

# -------------------------
# RNGs
# -------------------------
py_rng = random.Random(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

chain_torch_rng = torch.Generator(device="cpu").manual_seed(SEED + 1)
baseline_torch_rng = torch.Generator(device="cpu").manual_seed(SEED + 100)

# -------------------------
# Helpers
# -------------------------
def ensure_rgb(img, size=(WIDTH, HEIGHT)):
    return img.convert("RGB").resize(size)


def save_image(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def sample_standard_normal(shape, rng):
    return torch.randn(shape, generator=rng, device="cpu", dtype=torch.float32)


def pcn_proposal(x, beta):
    xi = sample_standard_normal(x.shape, chain_torch_rng)
    return math.sqrt(1.0 - beta**2) * x + beta * xi


def propose_latent(z):
    u = py_rng.random()
    if u < LOCAL_PROB:
        return pcn_proposal(z, LOCAL_BETA), "local"
    elif u < LOCAL_PROB + WIDE_PROB:
        return pcn_proposal(z, WIDE_BETA), "wide"
    else:
        z_new = sample_standard_normal(z.shape, chain_torch_rng)
        return z_new, "refresh"


def compute_r0(z, z_prop):
    return 1.0


def extract_ab_token(raw_output, rng=None, verbose=False):
    def unwrap(obj):
        if isinstance(obj, dict):
            if "generated_text" in obj:
                return unwrap(obj["generated_text"])
            if "content" in obj:
                return unwrap(obj["content"])
            return str(obj)
        if isinstance(obj, list):
            if len(obj) == 0:
                return ""
            return unwrap(obj[-1])
        return str(obj)

    text = unwrap(raw_output).strip()
    text_upper = text.upper()

    match = re.search(r"\b([AB])\b", text_upper)
    if match:
        return match.group(1)

    if text_upper.startswith("A"):
        return "A"
    if text_upper.startswith("B"):
        return "B"

    if rng is None:
        token = "A" if random.random() < 0.5 else "B"
    else:
        token = "A" if rng.random() < 0.5 else "B"

    if verbose:
        print(f"[judge parse fallback] raw output={text!r} -> sampled {token}")

    return token


def is_cuda_oom_error(exc):
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or "out of memory" in msg
        or "cublas_status_alloc_failed" in msg
    )


def latent_to_deterministic_seed(latent_noise):
    arr = latent_noise.detach().cpu().numpy()
    digest = hashlib.sha256(arr.tobytes()).digest()
    seed64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return seed64 % (2**63 - 1)


def judge_factor_from_k(k, num_votes):
    return 0.0 if k == 0 else (k / (num_votes - k + 1))


def pil_for_judge(img):
    return img.convert("RGB").resize((JUDGE_IMAGE_SIZE, JUDGE_IMAGE_SIZE))


def format_judge_status(judge_names, vote_counts=None, judge_factors=None):
    lines = []
    for i, name in enumerate(judge_names):
        votes_txt = "N/A"
        factor_txt = "N/A"
        if vote_counts is not None and i < len(vote_counts) and vote_counts[i] is not None:
            votes_txt = f"{vote_counts[i]}/{NUM_VOTES}"
        if judge_factors is not None and i < len(judge_factors) and judge_factors[i] is not None:
            factor_txt = f"{judge_factors[i]:.3f}"
        lines.append(f"{name}: votes={votes_txt} | factor={factor_txt}")
    return "\n".join(lines)


def plot_mh_step(
    step, current_img, proposal_img, accepted, proposal_type,
    judge_names, vote_counts=None, judge_factors=None,
    combined_judge_factor=None, r0=None, accept_prob=None, accept_u=None,
    save_dir=None,
):
    if accepted is None:
        decision_text = "INITIALIZATION"
    else:
        decision_text = "ACCEPTED" if accepted else "REJECTED"

    judge_text = format_judge_status(
        judge_names=judge_names,
        vote_counts=vote_counts,
        judge_factors=judge_factors,
    )

    summary_lines = []
    if r0 is not None:
        summary_lines.append(f"r0 = {r0:.6f}")
    if combined_judge_factor is not None:
        summary_lines.append(f"joint judge factor = {combined_judge_factor:.6f}")
    if accept_prob is not None:
        summary_lines.append(f"alpha = {accept_prob:.6f}")
    if accept_u is not None:
        summary_lines.append(f"u = {accept_u:.6f}")

    summary_text = "\n".join(summary_lines) if summary_lines else "N/A"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(current_img)
    axes[0].set_title(f"Step {step}: Current")
    axes[0].axis("off")
    axes[1].imshow(proposal_img)
    axes[1].set_title(f"Step {step}: Proposal ({proposal_type})")
    axes[1].axis("off")
    axes[2].axis("off")
    axes[2].text(
        0.03, 0.95,
        (
            f"Step: {step}\n\n"
            f"Proposal: {proposal_type}\n\n"
            f"Decision: {decision_text}\n\n"
            f"{judge_text}\n\n"
            f"{summary_text}"
        ),
        va="top", ha="left", fontsize=12, family="monospace",
    )
    fig.suptitle(f"Joint multi-judge exact N-vote MH step {step}", fontsize=16)
    fig.tight_layout()

    if save_dir is not None and SAVE_IMAGES:
        step_plot_path = save_dir / f"mh_step_{step:04d}.png"
        fig.savefig(step_plot_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


# -------------------------
# Generator loading / auto-config
# -------------------------
def load_text_to_image_pipeline(model_id):
    load_kwargs = {"torch_dtype": DIFFUSION_DTYPE}
    if DEVICE == "cuda":
        load_kwargs["use_safetensors"] = True
        load_kwargs["variant"] = "fp16"

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, **load_kwargs)
    except Exception:
        fallback = dict(load_kwargs)
        fallback.pop("variant", None)
        fallback.pop("use_safetensors", None)
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, **fallback)

    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "safety_checker"):
        try:
            pipe.safety_checker = None
        except Exception:
            pass

    return pipe


def configure_generator(pipe, model_id):
    model_id_lower = model_id.lower()
    pipeline_class_name = pipe.__class__.__name__.lower()
    is_sdxl = "xl" in pipeline_class_name
    is_turbo = "turbo" in model_id_lower
    is_lcm = "lcm" in model_id_lower

    if is_turbo:
        try:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing",
            )
        except Exception:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config
            )
        mode = "turbo"
        gen_steps = GEN_STEPS_TURBO
        guidance_scale = GUIDANCE_SCALE_TURBO
        ddim_eta = None
        use_negative_prompt = True

    elif is_lcm:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        mode = "lcm"
        gen_steps = GEN_STEPS_LCM
        guidance_scale = GUIDANCE_SCALE_LCM
        ddim_eta = None
        use_negative_prompt = False

    else:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        mode = "standard"
        gen_steps = GEN_STEPS_STANDARD
        guidance_scale = GUIDANCE_SCALE_STANDARD
        ddim_eta = DDIM_ETA_STANDARD
        use_negative_prompt = True

    cfg = {
        "pipeline_class_name": pipe.__class__.__name__,
        "is_sdxl": bool(is_sdxl),
        "is_turbo": bool(is_turbo),
        "is_lcm": bool(is_lcm),
        "mode": mode,
        "gen_steps": int(gen_steps),
        "guidance_scale": float(guidance_scale),
        "ddim_eta": None if ddim_eta is None else float(ddim_eta),
        "use_negative_prompt": bool(use_negative_prompt),
        "scheduler_name": pipe.scheduler.__class__.__name__,
    }
    return pipe, cfg


# -------------------------
# Load generator
# -------------------------
print("Loading text-to-image generator...")
gen_pipe = load_text_to_image_pipeline(BASE_MODEL_ID)
gen_pipe, GENERATOR_CFG = configure_generator(gen_pipe, BASE_MODEL_ID)
print("Generator loaded:")
print(json.dumps(GENERATOR_CFG, indent=2))

LATENT_CHANNELS = gen_pipe.unet.config.in_channels
VAE_SCALE_FACTOR = getattr(gen_pipe, "vae_scale_factor", 8)
LATENT_H = HEIGHT // VAE_SCALE_FACTOR
LATENT_W = WIDTH // VAE_SCALE_FACTOR


@torch.inference_mode()
def render_candidate(latent_noise, prompt_text):
    latents = latent_noise.unsqueeze(0).to(device=DEVICE, dtype=DIFFUSION_DTYPE)
    render_seed = latent_to_deterministic_seed(latent_noise)
    render_generator = torch.Generator(device=DEVICE).manual_seed(render_seed)

    pipe_kwargs = dict(
        prompt=prompt_text,
        width=WIDTH,
        height=HEIGHT,
        latents=latents,
        generator=render_generator,
        num_inference_steps=GENERATOR_CFG["gen_steps"],
        guidance_scale=GENERATOR_CFG["guidance_scale"],
        output_type="pil",
    )
    if GENERATOR_CFG["use_negative_prompt"]:
        pipe_kwargs["negative_prompt"] = NEGATIVE_PROMPT
    if GENERATOR_CFG["ddim_eta"] is not None:
        pipe_kwargs["eta"] = GENERATOR_CFG["ddim_eta"]

    out = gen_pipe(**pipe_kwargs)
    return ensure_rgb(out.images[0])


def init_chain_from_prior():
    z0 = sample_standard_normal(
        (LATENT_CHANNELS, LATENT_H, LATENT_W), chain_torch_rng
    )
    img0 = render_candidate(z0, GENERATOR_PROMPT)
    return z0, img0


# -------------------------
# Generate random baseline samples from prior
# -------------------------
print(f"Generating {NUM_RANDOM_BASELINE} random baseline samples from prior...")
random_baseline_images = []
for i in tqdm(range(NUM_RANDOM_BASELINE), desc="Baseline"):
    z_rand = sample_standard_normal(
        (LATENT_CHANNELS, LATENT_H, LATENT_W), baseline_torch_rng
    )
    img_rand = render_candidate(z_rand, GENERATOR_PROMPT)
    random_baseline_images.append(img_rand)
    if SAVE_IMAGES:
        save_image(img_rand, CASE_DIR / f"baseline_{i:03d}.png")

print("Baseline samples generated.")

# -------------------------
# Load VLM judge
# -------------------------
print("Loading VLM judge pipeline...")
judge_pipe = pipeline(
    task="image-text-to-text",
    model=JUDGE_MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)

if hasattr(judge_pipe, "tokenizer") and judge_pipe.tokenizer is not None:
    judge_pipe.tokenizer.padding_side = "left"

JUDGE_SYSTEM_PROMPT = (
    "You are comparing two candidate images on behalf of a specific person. "
    "Choose the image that person would prefer for their own purpose. "
    "Focus only on the subject described in the role. Ignore everything else in the image. "
    "Do not reward or penalize an image because of other animals, objects, composition, lighting, realism, or background details, "
    "except insofar as they affect how well the role's subject is shown. "
    "If one image fits the role's preference better, choose it. "
    "If both images fit the role's preferred description equally well, choose randomly. "
    "Output exactly one character: A or B."
)



def build_salamander_prompt():
    return (
        "You take care of the salamander in this terrarium exhibit. "
        "The zoo needs one photo to represent the exhibit, and you want the salamander to be present in the image. "
        "Focus only on whether the image shows a bright orange salamander on or around driftwood. "
        "Ignore everything else in the image. In particular, do not penalize or reward an image because of the frog, the mushroom, "
        "plants, background, composition, or overall beauty. "
        "If both images already show your preferred salamander image equally well, choose randomly. "
        "Which image would you prefer to use? "
        "Output exactly one character: A or B."
    )

def build_butterfly_prompt():
    return (
        "You are an insect researcher responsible for the butterfly featured in this terrarium exhibit. "
        "The zoo needs one photo to represent the exhibit, and you want the butterfly to be present in the image. "
        "Focus only on whether the image shows a small turquoise or blue butterfly flying in the scene. "
        "Ignore everything else in the image. In particular, do not penalize or reward an image because of the salamander, the mushroom, "
        "driftwood, plants, background, composition, or overall beauty. "
        "If both images already show your preferred butterfly image equally well, choose randomly. "
        "Which image would you prefer to use? "
        "Output exactly one character: A or B."
    )


def build_mushroom_prompt():
    return (
        "You designed the forest-floor details of this terrarium exhibit. "
        "The zoo needs one photo to represent the exhibit, and you want that detail to come across well. "
        "Focus only on whether the image shows a red mushroom with white spots near the soil or lower part of the scene. "
        "Ignore everything else in the image. In particular, do not penalize or reward an image because of the salamander, the frog, "
        "plants, background, composition, or overall beauty. "
        "If both images already show your preferred mushroom image equally well, choose randomly. "
        "Which image would you prefer to use? "
        "Output exactly one character: A or B."
    )


PROMPT_BUILDERS = [
    ("SalamanderCurator", build_salamander_prompt),
    ("ButterflyResearcher", build_butterfly_prompt),
    ("TerrariumEditor", build_mushroom_prompt),
]


# -------------------------
# Shared judge bundle
# -------------------------
class QwenJudgeBundle:
    def __init__(self, shared_pipe, system_prompt):
        self.pipe = shared_pipe
        self.system_prompt = system_prompt
        self.prompt_builders = PROMPT_BUILDERS
        self.cached_batch_size = None
        self.last_used_batch_size = None

    def _single_message(self, img_a, img_b, criterion_text):
        img_a_small = pil_for_judge(img_a)
        img_b_small = pil_for_judge(img_b)
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Candidate A:"},
                    {"type": "image", "image": img_a_small},
                    {"type": "text", "text": "Candidate B:"},
                    {"type": "image", "image": img_b_small},
                    {"type": "text", "text": criterion_text},
                ],
            },
        ]

    def _run_messages_with_batch_size(self, message_batch, batch_size):
        outputs_all = []
        for start in range(0, len(message_batch), batch_size):
            sub_batch = message_batch[start : start + batch_size]
            outputs = self.pipe(
                text=sub_batch,
                batch_size=len(sub_batch),
                do_sample=JUDGE_DO_SAMPLE,
                temperature=JUDGE_TEMPERATURE,
                top_p=JUDGE_TOP_P,
                repetition_penalty=JUDGE_REPETITION_PENALTY,
                return_full_text=False,
                max_new_tokens=JUDGE_MAX_NEW_TOKENS,
            )
            if not outputs or len(outputs) != len(sub_batch):
                raise RuntimeError(
                    f"Judge pipeline returned unexpected output: {outputs!r}"
                )
            outputs_all.extend(outputs)
        return outputs_all

    def _run_with_adaptive_batching(self, message_batch):
        full_batch_size = len(message_batch)
        attempt_batch_size = (
            full_batch_size
            if self.cached_batch_size is None
            else min(self.cached_batch_size, full_batch_size)
        )

        while True:
            try:
                outputs = self._run_messages_with_batch_size(
                    message_batch, attempt_batch_size
                )
                self.cached_batch_size = attempt_batch_size
                self.last_used_batch_size = attempt_batch_size
                return outputs
            except Exception as exc:
                if not is_cuda_oom_error(exc):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if attempt_batch_size <= 1:
                    print("Judge batch size reached 1 and still OOM.")
                    raise
                new_batch_size = max(1, attempt_batch_size // 2)
                print(
                    f"Judge OOM batch_size={attempt_batch_size}, "
                    f"retrying with {new_batch_size}."
                )
                attempt_batch_size = new_batch_size

    def batch_vote(self, current_img, proposal_img, num_votes):
        message_batch = []
        meta = []
        raw_tokens = []

        for _ in range(num_votes):
            for judge_idx, (_, prompt_builder) in enumerate(self.prompt_builders):
                swap = py_rng.random() < 0.5
                if not swap:
                    img_a, img_b = current_img, proposal_img
                else:
                    img_a, img_b = proposal_img, current_img

                criterion = prompt_builder()
                message_batch.append(
                    self._single_message(img_a, img_b, criterion)
                )
                meta.append((judge_idx, swap))

        outputs = self._run_with_adaptive_batching(message_batch)
        win_counts = [0] * len(self.prompt_builders)

        for out, (judge_idx, swap) in zip(outputs, meta):
            token = extract_ab_token(out, rng=py_rng, verbose=False)
            raw_tokens.append(token)
            vote_for_proposal = 1 if token == "B" else 0
            if swap:
                vote_for_proposal = 1 - vote_for_proposal
            win_counts[judge_idx] += vote_for_proposal

        return win_counts, raw_tokens


judge_bundle = QwenJudgeBundle(judge_pipe, JUDGE_SYSTEM_PROMPT)


# -------------------------
# Joint exact N-vote acceptance rule
# -------------------------
def exact_n_vote_accept_joint(current_img, proposal_img, r0=1.0, num_votes=1):
    ks, raw_tokens = judge_bundle.batch_vote(
        current_img=current_img,
        proposal_img=proposal_img,
        num_votes=num_votes,
    )

    judge_factors = [judge_factor_from_k(k, num_votes) for k in ks]

    combined_judge_factor = 1.0
    for fac in judge_factors:
        combined_judge_factor *= fac

    accept_prob = min(1.0, r0 * combined_judge_factor)
    accept_u = py_rng.random()
    accept = accept_u < accept_prob

    return (
        accept,
        ks,
        judge_factors,
        combined_judge_factor,
        accept_prob,
        accept_u,
        raw_tokens,
    )


# -------------------------
# Tracking arrays
# -------------------------
saved_states = []
distinct_chain_states = []
step_records = []
accepted_steps_after_init = 0
proposal_type_counts = {
    "local": 0, "wide": 0, "refresh": 0, "initial_from_prior": 0,
}

accept_history = []
per_judge_k_history = {name: [] for name in JUDGE_NAMES}
cumulative_accept_rate = []

trajectory_indices = set()
if CHAIN_LENGTH > 1:
    step_gap = max(1, (CHAIN_LENGTH - 1) / (NUM_TRAJECTORY_SNAPSHOTS - 1))
    for i in range(NUM_TRAJECTORY_SNAPSHOTS):
        idx = min(int(round(i * step_gap)), CHAIN_LENGTH - 1)
        trajectory_indices.add(idx)
trajectory_indices.add(0)
trajectory_indices.add(CHAIN_LENGTH - 1)
trajectory_images = {}

# -------------------------
# Run chain
# -------------------------
print("Initializing chain from prior...")
x = None
current_img = None

print(f"Running MCMC chain for {CHAIN_LENGTH} steps...")
for step in tqdm(range(CHAIN_LENGTH), desc="MCMC"):
    if step == 0:
        x, current_img = init_chain_from_prior()
        proposal_type_counts["initial_from_prior"] += 1

        saved_states.append(current_img.copy())
        distinct_chain_states.append((step, current_img.copy()))
        if SAVE_IMAGES:
            save_image(current_img, CASE_DIR / f"state_step_{step:04d}.png")

        if step in trajectory_indices:
            trajectory_images[step] = current_img.copy()

        if PLOT_EVERY_STEP:
            if CLEAR_PREVIOUS_STEP_OUTPUT:
                clear_output(wait=True)
            plot_mh_step(
                step=step, current_img=current_img, proposal_img=current_img,
                accepted=None, proposal_type="initial_from_prior",
                judge_names=JUDGE_NAMES,
                save_dir=CASE_DIR if SAVE_STEP_PLOTS else None,
            )

        step_records.append({
            "step": step,
            "proposal_type": "initial_from_prior",
            "accepted": None,
            "vote_counts": None,
            "judge_factors": None,
            "combined_judge_factor": None,
            "accept_prob": None,
            "accept_u": None,
            "raw_tokens": None,
            "judge_batch_size_used": None,
            "initialization_only": True,
            "r0": None,
        })
        continue

    current_before = current_img.copy()

    x_prop, proposal_type = propose_latent(x)
    proposal_type_counts[proposal_type] += 1

    proposal_img = render_candidate(x_prop, GENERATOR_PROMPT)

    r0 = compute_r0(x, x_prop)

    (
        accept,
        vote_counts,
        judge_factors,
        combined_judge_factor,
        accept_prob,
        accept_u,
        raw_tokens,
    ) = exact_n_vote_accept_joint(
        current_img=current_before,
        proposal_img=proposal_img,
        r0=r0,
        num_votes=NUM_VOTES,
    )

    if accept:
        x = x_prop
        current_img = proposal_img
        accepted_steps_after_init += 1
        distinct_chain_states.append((step, current_img.copy()))
    else:
        current_img = current_before

    saved_states.append(current_img.copy())
    if SAVE_IMAGES:
        save_image(current_img, CASE_DIR / f"state_step_{step:04d}.png")

    if step in trajectory_indices:
        trajectory_images[step] = current_img.copy()

    if PLOT_EVERY_STEP:
        if CLEAR_PREVIOUS_STEP_OUTPUT:
            clear_output(wait=True)
        plot_mh_step(
            step=step, current_img=current_before, proposal_img=proposal_img,
            accepted=accept, proposal_type=proposal_type,
            judge_names=JUDGE_NAMES,
            vote_counts=vote_counts, judge_factors=judge_factors,
            combined_judge_factor=combined_judge_factor,
            r0=r0, accept_prob=accept_prob, accept_u=accept_u,
            save_dir=CASE_DIR if SAVE_STEP_PLOTS else None,
        )

    accept_history.append(accept)
    for i, name in enumerate(JUDGE_NAMES):
        per_judge_k_history[name].append(vote_counts[i])
    cumulative_accept_rate.append(
        accepted_steps_after_init / len(accept_history)
    )

    step_records.append({
        "step": step,
        "proposal_type": proposal_type,
        "accepted": bool(accept),
        "vote_counts": [int(v) for v in vote_counts],
        "judge_factors": [float(v) for v in judge_factors],
        "combined_judge_factor": float(combined_judge_factor),
        "accept_prob": float(accept_prob),
        "accept_u": float(accept_u),
        "raw_tokens": raw_tokens,
        "judge_batch_size_used": judge_bundle.last_used_batch_size,
        "initialization_only": False,
        "r0": float(r0),
    })

print(f"\nChain complete. Accepted {accepted_steps_after_init}/{CHAIN_LENGTH-1} "
      f"proposals ({100*accepted_steps_after_init/(CHAIN_LENGTH-1):.1f}%).")


# =================================================================
# FIGURE 1: Random Baseline Grid
# =================================================================
def make_grid_figure(images, title, ncols=4, labels=None):
    nrows = math.ceil(len(images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for idx, img in enumerate(images):
        axes[idx].imshow(img)
        if labels is None:
            axes[idx].set_title(f"Sample {idx+1}", fontsize=11)
        else:
            axes[idx].set_title(labels[idx], fontsize=11)
        axes[idx].axis("off")
    for idx in range(len(images), len(axes)):
        axes[idx].axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


fig_baseline = make_grid_figure(
    random_baseline_images,
    f"Random Draws from Prior $p_0$ (no MCMC)",
    ncols=4,
    labels=[f"Sample {i+1}" for i in range(len(random_baseline_images))],
)
baseline_path = CASE_DIR / "fig_baseline_grid.png"
if SAVE_IMAGES:
    fig_baseline.savefig(baseline_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_baseline)
if SAVE_IMAGES:
    print(f"Saved baseline grid: {baseline_path}")


# =================================================================
# FIGURE 2: MCMC Trajectory
# =================================================================
sorted_traj_steps = sorted(trajectory_images.keys())
traj_imgs = [trajectory_images[s] for s in sorted_traj_steps]

ncols_traj = len(traj_imgs)
fig_traj, axes_traj = plt.subplots(1, ncols_traj, figsize=(3.5 * ncols_traj, 4))
if ncols_traj == 1:
    axes_traj = [axes_traj]

for idx, (s, img) in enumerate(zip(sorted_traj_steps, traj_imgs)):
    axes_traj[idx].imshow(img)
    axes_traj[idx].set_title(f"Step {s}", fontsize=11)
    axes_traj[idx].axis("off")

fig_traj.suptitle("MCMC Chain Trajectory (early → late)", fontsize=16, fontweight="bold")
fig_traj.tight_layout()
traj_path = CASE_DIR / "fig_trajectory.png"
if SAVE_IMAGES:
    fig_traj.savefig(traj_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_traj)
if SAVE_IMAGES:
    print(f"Saved trajectory: {traj_path}")


# =================================================================
# Select last distinct MCMC states
# =================================================================
last_distinct_states = distinct_chain_states[-SAVE_LAST_K:]
last_distinct_steps = [step for step, _ in last_distinct_states]
last_distinct_images = [img for _, img in last_distinct_states]


# =================================================================
# FIGURE 3: Last distinct MCMC states
# =================================================================
fig_mcmc_grid = make_grid_figure(
    last_distinct_images,
    f"Last {len(last_distinct_images)} Distinct MCMC States",
    ncols=4,
    labels=[f"Step {s}" for s in last_distinct_steps],
)
mcmc_grid_path = CASE_DIR / "fig_mcmc_last_distinct_grid.png"
if SAVE_IMAGES:
    fig_mcmc_grid.savefig(mcmc_grid_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_mcmc_grid)
if SAVE_IMAGES:
    print(f"Saved MCMC distinct-state grid: {mcmc_grid_path}")


# =================================================================
# FIGURE 4: Side-by-side comparison
# =================================================================
n_compare = min(len(last_distinct_images), len(random_baseline_images), SAVE_LAST_K)

fig_compare = plt.figure(figsize=(4 * n_compare, 9))
gs = gridspec.GridSpec(2, n_compare, hspace=0.15, wspace=0.05)

for col in range(n_compare):
    ax_top = fig_compare.add_subplot(gs[0, col])
    ax_top.imshow(random_baseline_images[col])
    ax_top.set_title(f"Random {col+1}", fontsize=10)
    ax_top.axis("off")

    ax_bot = fig_compare.add_subplot(gs[1, col])
    ax_bot.imshow(last_distinct_images[col])
    ax_bot.set_title(f"MCMC step {last_distinct_steps[col]}", fontsize=10)
    ax_bot.axis("off")

fig_compare.text(
    0.02, 0.75, "Random\n(prior $p_0$)",
    fontsize=14, fontweight="bold", va="center", ha="center", rotation=90,
)
fig_compare.text(
    0.02, 0.28, "MCMC\n(distinct late states)",
    fontsize=14, fontweight="bold", va="center", ha="center", rotation=90,
)

fig_compare.suptitle(
    "Random Prior Samples  vs  Distinct Late-Chain MCMC States",
    fontsize=16, fontweight="bold",
)
compare_path = CASE_DIR / "fig_comparison_panel.png"
if SAVE_IMAGES:
    fig_compare.savefig(compare_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_compare)
if SAVE_IMAGES:
    print(f"Saved comparison panel: {compare_path}")


# =================================================================
# FIGURE 5: Diagnostics
# =================================================================
fig_diag, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

steps_mh = list(range(1, CHAIN_LENGTH))

ax1.plot(steps_mh, cumulative_accept_rate, color="black", linewidth=1.5, label="Cumulative")

window = min(20, len(accept_history))
if window > 0:
    rolling_accept = []
    for i in range(len(accept_history)):
        lo = max(0, i - window + 1)
        rolling_accept.append(
            sum(accept_history[lo : i + 1]) / (i - lo + 1)
        )
    ax1.plot(steps_mh, rolling_accept, color="tab:blue", linewidth=1.2,
             alpha=0.7, label=f"Rolling (w={window})")

ax1.set_ylabel("Acceptance Rate", fontsize=12)
ax1.set_title("MH Acceptance Rate Over Chain", fontsize=14)
ax1.legend(fontsize=10)
ax1.set_ylim(-0.02, 1.02)
ax1.grid(True, alpha=0.3)

colors = {
    "SalamanderCurator": "darkorange",
    "ButterflyResearcher": "teal",
    "TerrariumEditor": "crimson",
}
for name in JUDGE_NAMES:
    ks = per_judge_k_history[name]
    win_rates = [k / NUM_VOTES for k in ks]

    rolling_wr = []
    for i in range(len(win_rates)):
        lo = max(0, i - window + 1)
        rolling_wr.append(np.mean(win_rates[lo : i + 1]))

    ax2.plot(
        steps_mh, rolling_wr,
        color=colors.get(name, "gray"),
        linewidth=1.5,
        label=f"{name} (rolling avg)",
    )

ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.set_xlabel("Chain Step", fontsize=12)
ax2.set_ylabel(f"Proposal Win Rate ($K_i / {NUM_VOTES}$)", fontsize=12)
ax2.set_title("Per-Judge Rolling Proposal Win Rate", fontsize=14)
ax2.legend(fontsize=10)
ax2.set_ylim(-0.02, 1.02)
ax2.grid(True, alpha=0.3)

fig_diag.tight_layout()
diag_path = CASE_DIR / "fig_diagnostics.png"
if SAVE_IMAGES:
    fig_diag.savefig(diag_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig_diag)
if SAVE_IMAGES:
    print(f"Saved diagnostics: {diag_path}")


# =================================================================
# Save summary JSON
# =================================================================
summary = {
    "run_name": RUN_NAME,
    "seed": SEED,
    "device": DEVICE,
    "dtype": str(DIFFUSION_DTYPE),
    "scene_name": "terrarium_macro_scene_butterfly",
    "chain_length_total_steps": CHAIN_LENGTH,
    "kernel_steps": max(0, CHAIN_LENGTH - 1),
    "num_votes": NUM_VOTES,
    "generator_prompt": GENERATOR_PROMPT,
    "targets": {
        "M1_salamander_curator_prefers_orange_salamander_on_grey_driftwood": "image features a bright orange salamander climbing on a piece of grey driftwood",
        "M2_butterfly_researcher_prefers_turquoise_butterfly_on_green_leaf": "image features a small turquoise butterfly resting on a broad green leaf",
        "M3_terrarium_editor_prefers_red_mushroom_with_white_spots_near_soil": "image features a red mushroom with white spots growing at the base near soil",
    },
    "models": {
        "base_model_id": BASE_MODEL_ID,
        "judge_model_id": JUDGE_MODEL_ID,
    },
    "generator_config": GENERATOR_CFG,
    "determinism": {
        "generator_uses_explicit_latents": True,
        "generator_uses_latent_derived_seed": True,
        "generator_deterministic_given_latent_in_fixed_env": True,
        "judge_is_stochastic": True,
    },
    "conditioning": {
        "type": "text_only",
        "uses_source_image_in_generator": False,
        "uses_control_image": False,
        "guidance_scale": GENERATOR_CFG["guidance_scale"],
    },
    "proposal_mixture": {
        "r0_identically_one": True,
        "local": {"type": "pcn", "prob": LOCAL_PROB, "beta": LOCAL_BETA},
        "wide": {"type": "pcn", "prob": WIDE_PROB, "beta": WIDE_BETA},
        "refresh": {"type": "independent_prior_draw", "prob": REFRESH_PROB},
    },
    "acceptance_rule": {
        "type": "joint_multi_judge_single_coin",
        "formula": "min(1, r0 * prod_i(K_i / (N-K_i+1)))",
        "single_final_coin_flip": True,
        "per_judge_sequential_coin_flips": False,
    },
    "proposal_type_counts": proposal_type_counts,
    "accept_rate_after_init": accepted_steps_after_init / max(1, CHAIN_LENGTH - 1),
    "num_accepted_after_init": accepted_steps_after_init,
    "num_saved_states": len(saved_states),
    "num_distinct_chain_states": len(distinct_chain_states),
    "num_random_baseline_samples": NUM_RANDOM_BASELINE,
    "judge_cached_batch_size_final": judge_bundle.cached_batch_size,
}

if SAVE_IMAGES:
    with open(CASE_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(CASE_DIR / "step_records.json", "w") as f:
        json.dump(step_records, f, indent=2)


# =================================================================
# Final display
# =================================================================
print("\n" + "=" * 60)
print("RUN COMPLETE")
print("=" * 60)
print(json.dumps(summary, indent=2))

if SAVE_IMAGES:
    print(f"\nOutputs saved to: {CASE_DIR}")
    print(f"  - Baseline grid:      {baseline_path.name}")
    print(f"  - Trajectory:         {traj_path.name}")
    print(f"  - Distinct MCMC grid: {mcmc_grid_path.name}")
    print(f"  - Comparison panel:   {compare_path.name}")
    print(f"  - Diagnostics:        {diag_path.name}")
    print(f"  - Per-step images:    state_step_*.png")
    print(f"  - Summary:            summary.json")
    print(f"  - Step records:       step_records.json")
else:
    print("\nSAVE_IMAGES=False — no files were written to disk.")