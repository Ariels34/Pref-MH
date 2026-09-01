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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import numpy as np

from PIL import Image
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
RUN_NAME = f"terrarium_butterfly_pointwise_mh_{MODEL_TAG}_joint"

# Master switch: when False, no files are written to disk at all.
SAVE_IMAGES = True

OUTPUT_DIR = Path("./files_pointwise")
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

# Pointwise-score settings
SCORE_REPEATS = 1

POINTWISE_BETAS = {
    "SalamanderCurator": 1.0,
    "ButterflyResearcher": 1.0,
    "TerrariumEditor": 1.0,
}

# Judge decoding
# Deterministic scoring is recommended for a cleaner MH target
JUDGE_MAX_NEW_TOKENS = 64
JUDGE_DO_SAMPLE = False
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0
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
    # For this proposal mixture, the p0/proposal term cancels exactly.
    return 1.0


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


def pil_for_judge(img):
    return img.convert("RGB").resize((JUDGE_IMAGE_SIZE, JUDGE_IMAGE_SIZE))


def safe_exp(x):
    # Avoid overflow/underflow in extreme cases
    x = float(x)
    if x > 709:
        x = 709
    elif x < -745:
        x = -745
    return math.exp(x)


def unwrap_generated_text(raw_output):
    if isinstance(raw_output, dict):
        if "generated_text" in raw_output:
            return unwrap_generated_text(raw_output["generated_text"])
        if "content" in raw_output:
            return unwrap_generated_text(raw_output["content"])
        if "text" in raw_output:
            return unwrap_generated_text(raw_output["text"])
        return str(raw_output)
    if isinstance(raw_output, list):
        if len(raw_output) == 0:
            return ""
        return unwrap_generated_text(raw_output[-1])
    return str(raw_output)


def extract_score_0_to_10(raw_output, verbose=False):
    """
    Parse a 0..10 score from judge output.
    Expects JSON like {"score": 7.5}, but also handles looser outputs.
    """
    text = unwrap_generated_text(raw_output).strip()

    # First try strict JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "score" in obj:
            val = float(obj["score"])
            return float(np.clip(val, 0.0, 10.0)), text
        if isinstance(obj, (int, float)):
            val = float(obj)
            return float(np.clip(val, 0.0, 10.0)), text
    except Exception:
        pass

    # Then try regex fallback
    patterns = [
        r'"score"\s*:\s*(-?\d+(?:\.\d+)?)',
        r"score\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"\b(-?\d+(?:\.\d+)?)\s*/\s*10\b",
        r"\b(-?\d+(?:\.\d+)?)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = float(m.group(1))
            return float(np.clip(val, 0.0, 10.0)), text

    if verbose:
        print(f"[score parse fallback] raw output={text!r} -> score 5.0")

    # Neutral fallback
    return 5.0, text


def get_beta_vector():
    return [float(POINTWISE_BETAS[name]) for name in JUDGE_NAMES]


def format_judge_status(judge_names, current_scores=None, proposal_scores=None, log_factors=None):
    lines = []
    for i, name in enumerate(judge_names):
        cur_txt = "N/A"
        prop_txt = "N/A"
        delta_txt = "N/A"
        logfac_txt = "N/A"

        if current_scores is not None and i < len(current_scores):
            cur_txt = f"{current_scores[i]:.2f}"
        if proposal_scores is not None and i < len(proposal_scores):
            prop_txt = f"{proposal_scores[i]:.2f}"
        if (
            current_scores is not None
            and proposal_scores is not None
            and i < len(current_scores)
            and i < len(proposal_scores)
        ):
            delta_txt = f"{proposal_scores[i] - current_scores[i]:+.2f}"
        if log_factors is not None and i < len(log_factors):
            logfac_txt = f"{log_factors[i]:+.3f}"

        lines.append(
            f"{name}: score cur={cur_txt} | prop={prop_txt} | delta={delta_txt} | logfac={logfac_txt}"
        )
    return "\n".join(lines)


def plot_mh_step(
    step, current_img, proposal_img, accepted, proposal_type,
    judge_names, current_scores=None, proposal_scores=None, per_judge_log_factors=None,
    combined_log_factor=None, r0=None, accept_prob=None, accept_u=None,
    save_dir=None,
):
    if accepted is None:
        decision_text = "INITIALIZATION"
    else:
        decision_text = "ACCEPTED" if accepted else "REJECTED"

    judge_text = format_judge_status(
        judge_names=judge_names,
        current_scores=current_scores,
        proposal_scores=proposal_scores,
        log_factors=per_judge_log_factors,
    )

    summary_lines = []
    if r0 is not None:
        summary_lines.append(f"r0 = {r0:.6f}")
    if combined_log_factor is not None:
        summary_lines.append(f"joint log judge factor = {combined_log_factor:+.6f}")
        summary_lines.append(f"joint judge factor = {safe_exp(combined_log_factor):.6f}")
    if accept_prob is not None:
        summary_lines.append(f"alpha = {accept_prob:.6f}")
    if accept_u is not None:
        summary_lines.append(f"u = {accept_u:.6f}")

    summary_text = "\n".join(summary_lines) if summary_lines else "N/A"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
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
        va="top", ha="left", fontsize=11, family="monospace",
    )
    fig.suptitle(f"Joint multi-judge pointwise-score MH step {step}", fontsize=16)
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
# Stop Transformers warning:
# "Both max_new_tokens and max_length seem to have been set..."
try:
    judge_pipe.model.generation_config.max_length = None
except Exception:
    pass

try:
    judge_pipe.generation_config.max_length = None
except Exception:
    pass

if hasattr(judge_pipe, "tokenizer") and judge_pipe.tokenizer is not None:
    judge_pipe.tokenizer.padding_side = "left"


# -------------------------
# Judge prompts
# -------------------------
JUDGE_SYSTEM_PROMPT = (
    "You are scoring one candidate image for one specific criterion. "
    "Focus only on the subject described by the criterion. "
    "Ignore all other animals, objects, composition, lighting, realism, and background details, "
    "except when they directly affect the requested subject. "
    "Return only valid JSON in exactly this format: {\"score\": number}. "
    "The score must be a number from 0 to 10. Do not include explanations."
)


def build_salamander_score_prompt():
    return (
        "Score this image from 0 to 10 for the salamander criterion. "
        "10 means the image clearly shows a bright orange salamander on or around grey driftwood. "
        "5 means a possible salamander or driftwood is partially visible but unclear, incomplete, "
        "or the salamander is not strongly orange or not clearly on/around driftwood. "
        "0 means there is no recognizable salamander relevant to the criterion. "
        "Any other value sits in this range."
        "Ignore the butterfly, mushroom, plants, background, composition, and overall beauty. "
        "Output only JSON: {\"score\": number}."
    )


def build_butterfly_score_prompt():
    return (
        "Score this image from 0 to 10 for the butterfly criterion. "
        "10 means the image clearly shows a small turquoise or blue butterfly visibly flying in the scene. "
        "5 means a possible butterfly is present but unclear, the color is wrong. "
        "0 means there is no recognizable butterfly relevant to the criterion. "
        "Any other value sits in this range."
        "Ignore the salamander, mushroom, driftwood, plants, background, composition, and overall beauty. "
        "Output only JSON: {\"score\": number}."
    )


def build_mushroom_score_prompt():
    return (
        "Score this image from 0 to 10 for the mushroom criterion. "
        "10 means the image clearly shows a red mushroom with white spots near the soil or lower part of the scene. "
        "5 means a possible mushroom is present but unclear, not strongly red, missing white spots. "
        "0 means there is no recognizable mushroom relevant to the criterion. "
        "Any other value sits in this range."
        "Ignore the salamander, butterfly, plants, background, composition, and overall beauty. "
        "Output only JSON: {\"score\": number}."
    )


PROMPT_BUILDERS = [
    ("SalamanderCurator", build_salamander_score_prompt),
    ("ButterflyResearcher", build_butterfly_score_prompt),
    ("TerrariumEditor", build_mushroom_score_prompt),
]


# -------------------------
# Shared pointwise judge bundle
# -------------------------
class QwenPointwiseJudgeBundle:
    def __init__(self, shared_pipe, system_prompt):
        self.pipe = shared_pipe
        self.system_prompt = system_prompt
        self.prompt_builders = PROMPT_BUILDERS
        self.cached_batch_size = None
        self.last_used_batch_size = None

    def _single_message(self, img, criterion_text):
        img_small = pil_for_judge(img)
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Candidate image:"},
                    {"type": "image", "image": img_small},
                    {"type": "text", "text": criterion_text},
                ],
            },
        ]

    def _run_messages_with_batch_size(self, message_batch, batch_size):
        outputs_all = []
        for start in range(0, len(message_batch), batch_size):
            sub_batch = message_batch[start : start + batch_size]

            gen_kwargs = dict(
                text=sub_batch,
                batch_size=len(sub_batch),
                return_full_text=False,
                max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                repetition_penalty=JUDGE_REPETITION_PENALTY,
            )

            if JUDGE_DO_SAMPLE:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=JUDGE_TEMPERATURE,
                    top_p=JUDGE_TOP_P,
                )
            else:
                gen_kwargs.update(do_sample=False)

            outputs = self.pipe(**gen_kwargs)

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
                    f"Judge OOM batch_size={attempt_batch_size}, retrying with {new_batch_size}."
                )
                attempt_batch_size = new_batch_size

    def batch_score_images(self, images, score_repeats=1):
        """
        Score a list of images for all judges.

        Returns:
            mean_scores: list[list[float]], shape [num_images][num_judges]
            raw_texts:   list[list[list[str]]], shape [num_images][num_judges][score_repeats]
        """
        message_batch = []
        meta = []

        for image_idx, img in enumerate(images):
            for repeat_idx in range(score_repeats):
                for judge_idx, (_, prompt_builder) in enumerate(self.prompt_builders):
                    criterion = prompt_builder()
                    message_batch.append(self._single_message(img, criterion))
                    meta.append((image_idx, judge_idx, repeat_idx))

        outputs = self._run_with_adaptive_batching(message_batch)

        num_images = len(images)
        num_judges = len(self.prompt_builders)

        score_lists = [[[] for _ in range(num_judges)] for _ in range(num_images)]
        raw_texts = [[[] for _ in range(num_judges)] for _ in range(num_images)]

        for out, (image_idx, judge_idx, repeat_idx) in zip(outputs, meta):
            score, raw_text = extract_score_0_to_10(out, verbose=False)
            normalized_score = score/10.0
            score_lists[image_idx][judge_idx].append(float(normalized_score))
            raw_texts[image_idx][judge_idx].append(raw_text)

        mean_scores = []
        for image_idx in range(num_images):
            per_image = []
            for judge_idx in range(num_judges):
                vals = score_lists[image_idx][judge_idx]
                per_image.append(float(np.mean(vals)) if vals else 5.0)
            mean_scores.append(per_image)

        return mean_scores, raw_texts


judge_bundle = QwenPointwiseJudgeBundle(judge_pipe, JUDGE_SYSTEM_PROMPT)


# -------------------------
# Pointwise acceptance rule
# -------------------------
def pointwise_score_accept_joint(current_scores, proposal_scores, r0=1.0):
    betas = get_beta_vector()

    per_judge_score_deltas = []
    per_judge_log_factors = []
    per_judge_factors = []

    for beta, s_cur, s_prop in zip(betas, current_scores, proposal_scores):
        delta = float(s_prop) - float(s_cur)
        log_fac = float(beta) * delta
        per_judge_score_deltas.append(delta)
        per_judge_log_factors.append(log_fac)
        per_judge_factors.append(safe_exp(log_fac))

    if r0 <= 0.0:
        log_alpha_ratio = -float("inf")
    else:
        log_alpha_ratio = math.log(float(r0)) + float(np.sum(per_judge_log_factors))

    accept_prob = 1.0 if log_alpha_ratio >= 0.0 else safe_exp(log_alpha_ratio)
    accept_u = py_rng.random()
    accept = accept_u < accept_prob

    return (
        accept,
        per_judge_score_deltas,
        per_judge_log_factors,
        per_judge_factors,
        float(np.sum(per_judge_log_factors)),
        float(log_alpha_ratio),
        float(accept_prob),
        float(accept_u),
    )


# -------------------------
# Tracking arrays
# -------------------------
saved_states = []
distinct_chain_states = []
step_records = []
accepted_steps_after_init = 0

proposal_type_counts = {
    "local": 0,
    "wide": 0,
    "refresh": 0,
    "initial_from_prior": 0,
}

accept_history = []
cumulative_accept_rate = []
per_judge_current_score_history = {name: [] for name in JUDGE_NAMES}
per_judge_proposal_score_history = {name: [] for name in JUDGE_NAMES}
per_judge_delta_history = {name: [] for name in JUDGE_NAMES}
per_judge_log_factor_history = {name: [] for name in JUDGE_NAMES}

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
current_scores = None
current_raw_score_texts = None

print(f"Running POINTWISE-SCORE MCMC chain for {CHAIN_LENGTH} steps...")
for step in tqdm(range(CHAIN_LENGTH), desc="Pointwise-MH"):
    if step == 0:
        x, current_img = init_chain_from_prior()
        proposal_type_counts["initial_from_prior"] += 1

        current_scores_list, current_raw_list = judge_bundle.batch_score_images(
            [current_img], score_repeats=SCORE_REPEATS
        )
        current_scores = current_scores_list[0]
        current_raw_score_texts = current_raw_list[0]

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
                step=step,
                current_img=current_img,
                proposal_img=current_img,
                accepted=None,
                proposal_type="initial_from_prior",
                judge_names=JUDGE_NAMES,
                current_scores=current_scores,
                proposal_scores=current_scores,
                save_dir=CASE_DIR if SAVE_STEP_PLOTS else None,
            )

        step_records.append({
            "step": step,
            "proposal_type": "initial_from_prior",
            "accepted": None,
            "current_scores": [float(v) for v in current_scores],
            "proposal_scores": None,
            "score_deltas": None,
            "per_judge_log_factors": None,
            "per_judge_factors": None,
            "combined_log_judge_factor": None,
            "log_alpha_ratio": None,
            "accept_prob": None,
            "accept_u": None,
            "current_raw_score_texts": current_raw_score_texts,
            "proposal_raw_score_texts": None,
            "judge_batch_size_used": judge_bundle.last_used_batch_size,
            "initialization_only": True,
            "r0": None,
        })
        continue

    current_before = current_img.copy()
    current_scores_before = list(current_scores)
    current_raw_before = current_raw_score_texts

    x_prop, proposal_type = propose_latent(x)
    proposal_type_counts[proposal_type] += 1

    proposal_img = render_candidate(x_prop, GENERATOR_PROMPT)
    proposal_scores_list, proposal_raw_list = judge_bundle.batch_score_images(
        [proposal_img], score_repeats=SCORE_REPEATS
    )
    proposal_scores = proposal_scores_list[0]
    proposal_raw_score_texts = proposal_raw_list[0]

    r0 = compute_r0(x, x_prop)

    (
        accept,
        score_deltas,
        per_judge_log_factors,
        per_judge_factors,
        combined_log_judge_factor,
        log_alpha_ratio,
        accept_prob,
        accept_u,
    ) = pointwise_score_accept_joint(
        current_scores=current_scores_before,
        proposal_scores=proposal_scores,
        r0=r0,
    )

    if accept:
        x = x_prop
        current_img = proposal_img
        current_scores = list(proposal_scores)
        current_raw_score_texts = proposal_raw_score_texts
        accepted_steps_after_init += 1
        distinct_chain_states.append((step, current_img.copy()))
    else:
        current_img = current_before
        current_scores = current_scores_before
        current_raw_score_texts = current_raw_before

    saved_states.append(current_img.copy())

    if SAVE_IMAGES:
        save_image(current_img, CASE_DIR / f"state_step_{step:04d}.png")

    if step in trajectory_indices:
        trajectory_images[step] = current_img.copy()

    if PLOT_EVERY_STEP:
        if CLEAR_PREVIOUS_STEP_OUTPUT:
            clear_output(wait=True)
        plot_mh_step(
            step=step,
            current_img=current_before,
            proposal_img=proposal_img,
            accepted=accept,
            proposal_type=proposal_type,
            judge_names=JUDGE_NAMES,
            current_scores=current_scores_before,
            proposal_scores=proposal_scores,
            per_judge_log_factors=per_judge_log_factors,
            combined_log_factor=combined_log_judge_factor,
            r0=r0,
            accept_prob=accept_prob,
            accept_u=accept_u,
            save_dir=CASE_DIR if SAVE_STEP_PLOTS else None,
        )

    accept_history.append(bool(accept))
    cumulative_accept_rate.append(accepted_steps_after_init / len(accept_history))

    for i, name in enumerate(JUDGE_NAMES):
        per_judge_current_score_history[name].append(float(current_scores[i]))
        per_judge_proposal_score_history[name].append(float(proposal_scores[i]))
        per_judge_delta_history[name].append(float(score_deltas[i]))
        per_judge_log_factor_history[name].append(float(per_judge_log_factors[i]))

    step_records.append({
        "step": step,
        "proposal_type": proposal_type,
        "accepted": bool(accept),
        "current_scores_before": [float(v) for v in current_scores_before],
        "proposal_scores": [float(v) for v in proposal_scores],
        "current_scores_after": [float(v) for v in current_scores],
        "score_deltas": [float(v) for v in score_deltas],
        "per_judge_log_factors": [float(v) for v in per_judge_log_factors],
        "per_judge_factors": [float(v) for v in per_judge_factors],
        "combined_log_judge_factor": float(combined_log_judge_factor),
        "combined_judge_factor": float(safe_exp(combined_log_judge_factor)),
        "log_alpha_ratio": float(log_alpha_ratio),
        "accept_prob": float(accept_prob),
        "accept_u": float(accept_u),
        "current_raw_score_texts_before": current_raw_before,
        "proposal_raw_score_texts": proposal_raw_score_texts,
        "judge_batch_size_used": judge_bundle.last_used_batch_size,
        "initialization_only": False,
        "r0": float(r0),
    })

print(
    f"\nChain complete. Accepted {accepted_steps_after_init}/{CHAIN_LENGTH-1} "
    f"proposals ({100*accepted_steps_after_init/(CHAIN_LENGTH-1):.1f}%)."
)


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
    "Random Draws from Prior $p_0$ (no MCMC)",
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

fig_traj.suptitle("Pointwise-score MH Chain Trajectory (early → late)", fontsize=16, fontweight="bold")
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
    f"Last {len(last_distinct_images)} Distinct Pointwise-MH States",
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
    ax_bot.set_title(f"MH step {last_distinct_steps[col]}", fontsize=10)
    ax_bot.axis("off")

fig_compare.text(
    0.02, 0.75, "Random\n(prior $p_0$)",
    fontsize=14, fontweight="bold", va="center", ha="center", rotation=90,
)
fig_compare.text(
    0.02, 0.28, "Pointwise MH\n(distinct late states)",
    fontsize=14, fontweight="bold", va="center", ha="center", rotation=90,
)

fig_compare.suptitle(
    "Random Prior Samples  vs  Distinct Late-Chain Pointwise-MH States",
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
fig_diag, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
steps_mh = list(range(1, CHAIN_LENGTH))

# Acceptance rate
ax1.plot(steps_mh, cumulative_accept_rate, color="black", linewidth=1.5, label="Cumulative")

window = min(20, len(accept_history))
if window > 0:
    rolling_accept = []
    for i in range(len(accept_history)):
        lo = max(0, i - window + 1)
        rolling_accept.append(sum(accept_history[lo : i + 1]) / (i - lo + 1))
    ax1.plot(steps_mh, rolling_accept, linewidth=1.2, alpha=0.7, label=f"Rolling (w={window})")

ax1.set_ylabel("Acceptance Rate", fontsize=12)
ax1.set_title("MH Acceptance Rate Over Chain", fontsize=14)
ax1.legend(fontsize=10)
ax1.set_ylim(-0.02, 1.02)
ax1.grid(True, alpha=0.3)

# Accepted-state scores
colors = {
    "SalamanderCurator": "darkorange",
    "ButterflyResearcher": "teal",
    "TerrariumEditor": "crimson",
}

for name in JUDGE_NAMES:
    scores = per_judge_current_score_history[name]
    rolling_scores = []
    for i in range(len(scores)):
        lo = max(0, i - window + 1)
        rolling_scores.append(float(np.mean(scores[lo : i + 1])))
    ax2.plot(
        steps_mh,
        rolling_scores,
        color=colors.get(name, "gray"),
        linewidth=1.5,
        label=f"{name} accepted-state score",
    )

ax2.set_ylabel("Score 0-10", fontsize=12)
ax2.set_title("Per-Judge Rolling Score of Current Chain State", fontsize=14)
ax2.legend(fontsize=10)
ax2.set_ylim(-0.2, 10.2)
ax2.grid(True, alpha=0.3)

# Proposal score deltas
for name in JUDGE_NAMES:
    deltas = per_judge_delta_history[name]
    rolling_delta = []
    for i in range(len(deltas)):
        lo = max(0, i - window + 1)
        rolling_delta.append(float(np.mean(deltas[lo : i + 1])))
    ax3.plot(
        steps_mh,
        rolling_delta,
        color=colors.get(name, "gray"),
        linewidth=1.5,
        label=f"{name} proposal-current delta",
    )

ax3.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax3.set_xlabel("Chain Step", fontsize=12)
ax3.set_ylabel("Proposal score delta", fontsize=12)
ax3.set_title("Per-Judge Rolling Proposal Score Advantage", fontsize=14)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

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
    "generator_prompt": GENERATOR_PROMPT,
    "targets": {
        "M1_salamander_curator_prefers_orange_salamander_on_grey_driftwood": (
            "image features a bright orange salamander climbing on a piece of grey driftwood"
        ),
        "M2_butterfly_researcher_prefers_turquoise_butterfly_flying": (
            "image features a small turquoise butterfly flying in the terrarium"
        ),
        "M3_terrarium_editor_prefers_red_mushroom_with_white_spots_near_soil": (
            "image features a red mushroom with white spots growing at the base near soil"
        ),
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
        "judge_do_sample": JUDGE_DO_SAMPLE,
        "judge_is_deterministic_if_backend_deterministic": not JUDGE_DO_SAMPLE,
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
    "score_repeats": SCORE_REPEATS,
    "pointwise_betas": POINTWISE_BETAS,
    "target_distribution": {
        "type": "pointwise_score_joint_multi_judge",
        "formula": "pi(z) proportional to p0(z) * prod_i exp(beta_i * score_i(image(z)))",
    },
    "acceptance_rule": {
        "type": "joint_multi_judge_pointwise_score_single_coin",
        "formula": "min(1, r0 * exp(sum_i beta_i*(score_i(prop)-score_i(cur))))",
        "single_final_coin_flip": True,
        "per_judge_sequential_coin_flips": False,
        "current_scores_cached_until_state_changes": True,
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



