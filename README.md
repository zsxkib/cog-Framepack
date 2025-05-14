# FramePack: Cog implementation for advanced video generation

[![Replicate](https://replicate.com/zsxkib/framepack/badge)](https://replicate.com/zsxkib/framepack)

This repository has a Cog container for **FramePack**, a video generation model that creates videos bit by bit from an image and a text prompt. This setup is made for Cog, using an API-first approach, and is based on ideas from the original [FramePack project](https://lllyasviel.github.io/frame_pack_gitpage/).

It works efficiently on different kinds of hardware, from consumer GPUs (like RTX 4090) to high-end datacenter GPUs (A100s/H100s), because it adjusts how it uses memory (you can see the details in `predict.py`).

**Model links and information:**
*   Original Project Page: [lllyasviel.github.io/frame_pack_gitpage/](https://lllyasviel.github.io/frame_pack_gitpage/)
*   Original Paper (Arxiv): [arxiv.org/abs/2504.12626](https://arxiv.org/abs/2504.12626)
*   The main transformer model used here is `lllyasviel/FramePackI2V_HY`.
*   This Cog packaging by: [zsxkib on GitHub](https://github.com/zsxkib) / [@zsakib\_ on Twitter](https://twitter.com/zsakib_)

## Prerequisites

*   **Docker**: You need Docker to build and run the container. [Install Docker](https://docs.docker.com/get-docker/).
*   **Cog**: You need Cog to build and run this model locally. [Install Cog](https://github.com/replicate/cog#install).
*   **NVIDIA GPU**: You need an NVIDIA GPU. This setup adjusts memory use for different GPUs:
    *   **Consumer GPUs (like RTX 4090 with about 24GB video memory)**: Runs in a low video memory mode, moving parts of the model around to save space.
    *   **Datacenter GPUs (like A100/H100 with 40GB or more video memory)**: Runs in a high video memory mode with models loaded on the GPU for more speed.
    *   This usually changes if you have less than 30GB of video memory (see `predict.py`).

## Run locally

Cog makes it easier to run FramePack locally. It handles building the container and downloading model files for you.

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_COG_FRAMEPACK_REPO.git # <-- TODO: Update this URL
    cd YOUR_COG_FRAMEPACK_REPO 
    ```

2.  **Run the model:**
    The first time you run `cog predict`, it builds the container and downloads all the model files (which can be large, over 30GB), so it might take a while. Runs after that will be faster.

    You can give the input image as a local file path (add `@` before the path) or as a public URL.

    ```bash
    # Example 1: Default settings (3 seconds video, random seed)
    cog predict \\
      -i input_image=@path/to/another_image.jpg \\
      -i prompt="A man walking through a bustling cyberpunk market at night, rain falling."

    # Example 2: Longer video, more steps, custom video quality
    cog predict \\
      -i input_image=@path/to/another_image.jpg \\
      -i prompt="The man transforms into a liquid metal statue, then reforms. Epic cinematic style." \\
      -i total_video_length_seconds=7.0 \\
      -i steps=40 \\
      -i mp4_crf=20

    # Example 3: Shorter video, different guidance scales, no TeaCache
    cog predict \\
      -i input_image=@path/to/another_image.jpg \\
      -i prompt="A close-up of the man\'s face, expressing surprise, then slowly smiling. Soft lighting." \\
      -i total_video_length_seconds=2.0 \\
      -i steps=20 \\
      -i cfg_scale=1.5 \\
      -i distilled_cfg_scale=8.0 \\
      -i use_teacache=false
    ```
    Cog saves the video (for example, `/tmp/cog_output_xxxx.mp4`) and prints the path.

    See the "Model parameters" section below or look at `predict.py` for all the options you can use.

## How it works

Cog uses `cog.yaml` to set up the Python environment, system packages, and other dependencies. The main logic for making videos is in `predict.py`.

*   **`setup()` method**: This is called when the Cog worker starts.
    1.  It sets up cache directories (for example, for Hugging Face models).
    2.  It checks if a GPU is available and how much video memory (VRAM) it has. This helps decide if it should run in `low_vram_mode` (usually if VRAM is less than 30GB).
    3.  It loads these models:
        *   Text Encoders: `LlamaModel` and `CLIPTextModel` from `hunyuanvideo-community/HunyuanVideo` (to understand your text prompt).
        *   VAE: `AutoencoderKLHunyuanVideo` from `hunyuanvideo-community/HunyuanVideo` (to handle the image data).
        *   Image Encoder: `SiglipVisionModel` from `lllyasviel/flux_redux_bfl` (to understand the input image).
        *   Transformer: `HunyuanVideoTransformer3DModelPacked` from `lllyasviel/FramePackI2V_HY` (the main model that creates the video frames).
    4.  If your computer is in low video memory mode, models are first loaded to the CPU and moved to/from the GPU when needed. In high video memory mode, they stay on the GPU.
    5.  It uses optimizations like VAE slicing/tiling and `DynamicSwapInstaller` (if they apply and are turned on in `predict.py`) in low video memory mode to help manage memory.

*   **`predict()` method**: This handles how the video is made, based on what you provide.
    1.  It takes inputs like `input_image`, `prompt`, `total_video_length_seconds`, and others (see "Model parameters").
    2.  It encodes your text prompts and gets the input image ready (resizing it, and using the VAE and CLIP vision models to process it). It manages moving models between the CPU and GPU if your computer is in low video memory mode.
    3.  It makes the video in sections, one after the other:
        *   The `transformer` model creates the video frames. In low video memory mode, other large models are moved to the CPU to make space on the GPU for the `transformer`. The `transformer` is then moved to the GPU to do its work and back to the CPU afterwards.
    4.  It then turns these intermediate frames (called latents) into the actual video pixels using the VAE, again managing GPU memory in low video memory mode.
    5.  It saves the final video as an `mp4` file.
    6.  After the video is made, it makes sure models are moved back to the CPU in low video memory mode to free up GPU memory.

## Model parameters

You can give these parameters to `cog predict` with the `-i` flag (for example, `-i parameter_name=value`):

| Parameter                    | Description                                                                                                | Default Value | Type          | Constraints                 |
| :--------------------------- | :--------------------------------------------------------------------------------------------------------- | :------------ | :------------ | :-------------------------- |
| `input_image`                | Input image for making the video. (Required)                                                              | _N/A_         | `Path`        |                             |
| `prompt`                     | Text prompt describing what you want in the video. (Required)                                              | _N/A_         | `str`         |                             |
| `negative_prompt`            | Text prompt to say what you want to avoid in the video.                                                    | `""`          | `str`         |                             |
| `seed`                       | A number to make sure you get the same output if you run it again. Leave it blank for a random one.        | `None`        | `Optional[int]` |                             |
| `total_video_length_seconds` | How long the video should be in seconds. Max 10 seconds for the API to work reliably.                      | `3.0`         | `float`       | `1.0` to `10.0`             |
| `latent_window_size`         | Size of the video sections being processed. This is an advanced setting; the default usually works best.     | `9`           | `int`         | `1` to `16`                 |
| `steps`                      | Number of steps the model takes to create each part of the video. This is an advanced setting; the default usually works best. | `25`          | `int`         | `1` to `50`                 |
| `cfg_scale`                  | How much the video should follow your prompt. This is an advanced setting; the default usually works best.   | `1.0`         | `float`       | `1.0` to `32.0`             |
| `distilled_cfg_scale`        | Another setting for how much the video should follow your prompt. This is an advanced setting; the default usually works best. | `10.0`        | `float`       | `1.0` to `32.0`             |
| `cfg_rescale`                | A way to adjust the guidance. This is an advanced setting; the default usually works best.                 | `0.0`         | `float`       | `0.0` to `1.0`              |
| `use_teacache`               | Use TeaCache to make it run faster. This might change the results a little.                                | `True`        | `bool`        |                             |
| `mp4_crf`                    | Controls the MP4 video quality and file size (0-51). Lower numbers mean better quality. Around 23 is a good balance. | `23`          | `int`         | `0` to `51`                 |

## License

The code in this repository for packaging the FramePack model with Cog uses the MIT License. See the [LICENSE](LICENSE) file for details.

Please see the original FramePack project for license information about the model files and original code. The model code and weights are under the Apache License 2.0 (see [LICENSE](LICENSE)), while the implementation code in [predict.py](predict.py) is under the MIT License. The other libraries and models this project uses (like Hugging Face Transformers, Diffusers, Hunyuan-DiT) each have their own licenses. These are often open-source licenses like Apache 2.0.

---

⭐ Star this on [GitHub](https://github.com/zsxkib/cog-Framepack)!

👋 Follow `zsxkib` on [Twitter/X](https://twitter.com/zsakib_)