"""
Inference code for DALL-E Mini, adapted from:
https://github.com/borisdayma/dalle-mini/blob/main/tools/inference/inference_pipeline.ipynb
"""
import io
import random
import typing
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from dalle_mini import DalleBart, DalleBartProcessor
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from loguru import logger
from PIL import Image
from vqgan_jax.modeling_flax_vqgan import VQModel

# create a random key
SEED = random.randint(0, 2**32 - 1)
KEY = jax.random.PRNGKey(SEED)

# DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
DALLE_COMMIT_ID = None

VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

# Generation params
TOP_K = None
TOP_P = None
TEMPERATURE = None
CONDITION_SCALE = 10.0


def init_models():
    """
    Initialize the needed models and parameters for image from text generation.
    """

    dalle_model, dalle_params = DalleBart.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )

    dalle_processor = DalleBartProcessor.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID
    )

    vqgan, vqgan_params = VQModel.from_pretrained(  # type: ignore
        VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
    )

    # replicate model params on device for faster inference
    dalle_params = replicate(dalle_params)
    vqgan_params = replicate(vqgan_params)

    return dalle_model, dalle_params, vqgan, vqgan_params, dalle_processor


# Generate encoded image
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6, 7))
def _p_generate(
    tokenized_prompt,
    key,
    dalle_params,
    dalle_model,
    top_k,
    top_p,
    temperature,
    condition_scale,
):
    return dalle_model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=dalle_params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# Decode image
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(2,))
def _p_decode(indices, vqgan_params, vqgan):
    return vqgan.decode_code(indices, params=vqgan_params)


def generate_images(
    *,
    prompt: str,
    dalle_model,
    dalle_params,
    vqgan,
    vqgan_params,
    dalle_processor,
    n_images_per_prompt: int,
) -> typing.Iterable[io.BytesIO]:
    """
    Generate a set of images from a prompt, write each to a BytesIO buffer,
    and yield each one.
    """
    logger.info("Generating %d images for prompt: %s", n_images_per_prompt, prompt)
    # Tokenize the prompt
    tokenized_prompts = dalle_processor([prompt])
    tokenized_prompts = replicate(tokenized_prompts)

    # Get a new key
    for i in range(n_images_per_prompt):
        logger.info("Generating image %d", i)
        global KEY
        KEY, sub_key = jax.random.split(KEY)

        # Generate images
        encoded_images = _p_generate(
            tokenized_prompts,
            shard_prng_key(sub_key),
            dalle_params,
            dalle_model,
            TOP_K,
            TOP_P,
            TEMPERATURE,
            CONDITION_SCALE,
        )

        # Remove BOS
        encoded_images = encoded_images.sequences[..., 1:]

        # Decode images
        decoded_images = _p_decode(encoded_images, vqgan_params, vqgan)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            img_buf = io.BytesIO()
            img.save(img_buf, format="png")
            # Rewind to the beginning of the buf for readers
            img_buf.seek(0)
            yield img_buf
