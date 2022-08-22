import io
import logging
import typing

import torch
from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)


def init_model(hf_hub_token: str, cuda: bool):
    """
    Return a HuggingFace pipeline for image generation based on the Stable
    Diffusion model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device} ({n_gpu} GPUs)")

    return StableDiffusionPipeline.from_pretrained(  # type: ignore
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=hf_hub_token,
    ).to(device)


def generate_images(
    *,
    prompt: str,
    pipe,
    n_images_per_prompt: int,
) -> typing.Iterable[io.BytesIO]:
    logger.info(f"Generating {n_images_per_prompt} images for prompt: {prompt}")

    for i in range(n_images_per_prompt):
        logger.info(f"Generating image {i}")

        with torch.autocast("cuda"):  # type: ignore
            image = pipe(prompt)["sample"][0]

        img_buf = io.BytesIO()
        image.save(img_buf, format="png")
        # Rewind to the beginning of the buf for readers
        img_buf.seek(0)
        yield img_buf
