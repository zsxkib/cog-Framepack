# Prediction interface for Cog ⚙️
# https://cog.run/python

import os

MODEL_CACHE = "model_cache"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import torch
import numpy as np
import tempfile
from PIL import Image
from typing import Optional

from cog import BasePredictor, Input, Path

# Assuming these helpers are available in the Cog environment
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu, gpu, get_cuda_free_memory_gb, # Ensure these resolve to 'cpu' and 'cuda'
    load_model_as_complete, unload_complete_models,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device,
    DynamicSwapInstaller  # If available and used
)
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# VRAM threshold for low_vram_mode (e.g., less than 30GB)
LOW_VRAM_THRESHOLD_GB = 65.0 
# Default preserved memory for transformer operations in low VRAM mode (from demo)
DEFAULT_GPU_PRESERVED_MEMORY_TRANSFORMER_LOAD_GB = 6
DEFAULT_GPU_PRESERVED_MEMORY_TRANSFORMER_OFFLOAD_GB = 8


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load models into memory once when the new worker is starting."""
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.low_vram_mode = False
        self.cpu_device = torch.device("cpu") # Explicit CPU device

        if self.device.type == "cuda":
            try:
                # Attempt to get free memory; if it fails, assume high VRAM or proceed cautiously.
                # get_cuda_free_memory_gb might not be universally available or fail.
                # A more robust way is to check total memory.
                device_props = torch.cuda.get_device_properties(self.device)
                total_vram_gb = device_props.total_memory / (1024 ** 3)
                print(f"Total VRAM: {total_vram_gb:.2f} GB on {device_props.name}")
                if total_vram_gb < LOW_VRAM_THRESHOLD_GB:
                    self.low_vram_mode = True
            except Exception as e:
                print(f"Could not determine VRAM size (error: {e}), assuming high VRAM mode.")
                # If unsure, default to high VRAM mode for Replicate-like environments.
                # For local Cog runs on low VRAM, user might need to force this or it might fail.
        else: # CPU mode
            self.low_vram_mode = True # Effectively, CPU mode is a low_vram_mode in terms of model handling
            print("Running on CPU, low_vram_mode equivalent enabled.")


        print(f"Using device: {self.device}")
        print(f"Low VRAM Mode: {self.low_vram_mode}")

        # Determine initial loading device based on VRAM mode
        initial_load_device = self.cpu_device if self.low_vram_mode else self.device

        print(f"Loading models to initial device: {initial_load_device}...")
        # Text Encoders and Tokenizers (Tokenizers are small, always load to CPU is fine)
        self.text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16, cache_dir=MODEL_CACHE).to(initial_load_device)
        self.text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16, cache_dir=MODEL_CACHE).to(initial_load_device)
        self.tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer', cache_dir=MODEL_CACHE)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2', cache_dir=MODEL_CACHE)
        
        # VAE
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16, cache_dir=MODEL_CACHE).to(initial_load_device)
        
        # Image Encoder and Feature Extractor
        self.feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor', cache_dir=MODEL_CACHE)
        self.image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16, cache_dir=MODEL_CACHE).to(initial_load_device)
        
        # Transformer Model
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16, cache_dir=MODEL_CACHE).to(initial_load_device)

        # Set models to evaluation mode and disable gradients
        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.image_encoder.eval()
        self.transformer.eval()

        self.transformer.high_quality_fp32_output_for_inference = True

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        if self.low_vram_mode and self.device.type == "cuda":
            print("Low VRAM mode: Enabling VAE slicing/tiling and DynamicSwapInstaller.")
            self.vae.enable_slicing()
            self.vae.enable_tiling()
            try:
                # DynamicSwapInstaller for transformer and text_encoder as per demo
                DynamicSwapInstaller.install_model(self.transformer, device=self.device) # Pass target device
                DynamicSwapInstaller.install_model(self.text_encoder, device=self.device)
                print("DynamicSwapInstaller applied to transformer and text_encoder.")
            except Exception as e:
                print(f"Failed to apply DynamicSwapInstaller (continuing without it): {e}")
        
        if not self.low_vram_mode and self.device.type == "cuda":
            print("High VRAM mode: All models are on GPU.")
        
        print("All models loaded and configured.")

    def _managed_text_encode(self, prompt, negative_prompt_mode=False):
        """Handles text encoding with model loading/unloading in low VRAM mode."""
        if self.low_vram_mode and self.device.type == "cuda":
            print("Low VRAM: Loading text encoders to GPU...")
            # demo_gradio.py used fake_diffusers_current_device for text_encoder
            # and load_model_as_complete for text_encoder_2.
            # We'll explicitly load both for clarity if they are on CPU.
            if self.text_encoder.device == self.cpu_device:
                 fake_diffusers_current_device(self.text_encoder, self.device) # Moves if on CPU
            if self.text_encoder_2.device == self.cpu_device:
                load_model_as_complete(self.text_encoder_2, target_device=self.device) # Ensures full model on device
        
        # Perform encoding (models are now on self.device if cuda enabled)
        # On CPU, they are already on self.device (which is cpu)
        target_text_encoder = self.text_encoder.to(self.device)
        target_text_encoder_2 = self.text_encoder_2.to(self.device)

        vec, pooler = encode_prompt_conds(prompt, target_text_encoder, target_text_encoder_2, self.tokenizer, self.tokenizer_2)
        
        if self.low_vram_mode and self.device.type == "cuda":
            print("Low VRAM: Offloading text encoders from GPU...")
            # Offload back to CPU. unload_complete_models might be too aggressive if other models are needed.
            # Moving them explicitly to CPU is safer here.
            self.text_encoder.to(self.cpu_device)
            self.text_encoder_2.to(self.cpu_device)
        return vec, pooler

    def _managed_vae_encode(self, image_pt):
        """Handles VAE encoding with model loading/unloading in low VRAM mode."""
        if self.low_vram_mode and self.device.type == "cuda" and self.vae.device == self.cpu_device:
            print("Low VRAM: Loading VAE to GPU for encoding...")
            load_model_as_complete(self.vae, target_device=self.device)

        target_vae = self.vae.to(self.device)
        latents = vae_encode(image_pt.to(self.device, dtype=target_vae.dtype), target_vae) # image_pt to device
        
        if self.low_vram_mode and self.device.type == "cuda":
            print("Low VRAM: Offloading VAE from GPU after encoding...")
            self.vae.to(self.cpu_device)
        return latents

    def _managed_clip_vision_encode(self, image_np):
        """Handles CLIP Vision encoding with model loading/unloading in low VRAM mode."""
        if self.low_vram_mode and self.device.type == "cuda" and self.image_encoder.device == self.cpu_device:
            print("Low VRAM: Loading Image Encoder to GPU...")
            load_model_as_complete(self.image_encoder, target_device=self.device)
        
        target_image_encoder = self.image_encoder.to(self.device)
        output = hf_clip_vision_encode(image_np, self.feature_extractor, target_image_encoder)
        
        if self.low_vram_mode and self.device.type == "cuda":
            print("Low VRAM: Offloading Image Encoder from GPU...")
            self.image_encoder.to(self.cpu_device)
        return output

    def _managed_vae_decode(self, latents_on_gpu):
        """Handles VAE decoding with model loading/unloading in low VRAM mode."""
        if self.low_vram_mode and self.device.type == "cuda" and self.vae.device == self.cpu_device:
            print("Low VRAM: Loading VAE to GPU for decoding...")
            load_model_as_complete(self.vae, target_device=self.device)
        
        target_vae = self.vae.to(self.device)
        pixels = vae_decode(latents_on_gpu.to(self.device, dtype=target_vae.dtype), target_vae) # latents already on gpu
        
        if self.low_vram_mode and self.device.type == "cuda":
            print("Low VRAM: Offloading VAE from GPU after decoding...")
            self.vae.to(self.cpu_device)
        return pixels.cpu() # Always return pixels on CPU

    @torch.no_grad()
    def predict(
        self,
        input_image: Path = Input(description="Input image for video generation."),
        prompt: str = Input(description="Text prompt describing the desired video content."),
        negative_prompt: str = Input(description="Negative text prompt to specify what to avoid.", default=""),
        seed: Optional[int] = Input(description="Random seed. Leave blank for a random seed.", default=None),
        total_video_length_seconds: float = Input(description="Total video length in seconds. Max 10s for API stability.", default=3.0, ge=1.0, le=10.0),
        latent_window_size: int = Input(description="Latent window size. Advanced setting, default is recommended.", default=9, ge=1, le=16),
        steps: int = Input(description="Number of inference steps. Advanced setting, default is recommended.", default=25, ge=1, le=50),
        cfg_scale: float = Input(description="Classifier-Free Guidance scale. Advanced setting, default is recommended.", default=1.0, ge=1.0, le=32.0),
        distilled_cfg_scale: float = Input(description="Distilled CFG scale. Advanced setting, default is recommended.", default=10.0, ge=1.0, le=32.0),
        cfg_rescale: float = Input(description="CFG rescale factor. Advanced setting, default is recommended.", default=0.0, ge=0.0, le=1.0),
        use_teacache: bool = Input(description="Use TeaCache for potentially faster speed (may slightly alter results).", default=True),
        mp4_crf: int = Input(description="MP4 Constant Rate Factor (0-51, lower is better quality, ~23 is a good balance).", default=23, ge=0, le=51)
    ) -> Path:
        """Generates a video based on an initial image and a text prompt."""

        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using random seed: {seed}")

        temp_dir = tempfile.mkdtemp()
        job_id = generate_timestamp()
        final_output_path_str = os.path.join(temp_dir, f"{job_id}_video.mp4")

        total_latent_sections = (total_video_length_seconds * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
        
        print("Encoding text prompts...")
        llama_vec, clip_l_pooler = self._managed_text_encode(prompt)
        if cfg_scale == 1.0:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = self._managed_text_encode(negative_prompt, negative_prompt_mode=True)
        
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        print("Processing input image...")
        input_image_pil = Image.open(input_image).convert("RGB")
        input_image_np_orig = np.array(input_image_pil)
        H, W, _ = input_image_np_orig.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np_resized = resize_and_center_crop(input_image_np_orig, target_width=width, target_height=height)
        
        input_image_pt = torch.from_numpy(input_image_np_resized).float() / 127.5 - 1.0
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None] # VAE expects B C F H W (F=1 for image)

        print("Encoding initial image with VAE...")
        # _managed_vae_encode handles moving image_pt to device
        start_latent = self._managed_vae_encode(input_image_pt) 
        start_latent = start_latent.to(self.device, dtype=self.transformer.dtype) # Ensure on target device for transformer

        print("Encoding image with CLIP Vision model...")
        image_encoder_output = self._managed_clip_vision_encode(input_image_np_resized)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.to(self.device, dtype=self.transformer.dtype)

        transformer_dtype = self.transformer.dtype
        llama_vec = llama_vec.to(self.device, dtype=transformer_dtype)
        llama_vec_n = llama_vec_n.to(self.device, dtype=transformer_dtype)
        clip_l_pooler = clip_l_pooler.to(self.device, dtype=transformer_dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(self.device, dtype=transformer_dtype)
        llama_attention_mask = llama_attention_mask.to(self.device)
        llama_attention_mask_n = llama_attention_mask_n.to(self.device)

        print("Starting video generation loop...")
        rnd_generator = torch.Generator(self.device).manual_seed(seed)
        num_pixel_frames_per_segment = latent_window_size * 4 - 3 

        history_latents_cpu = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32, device=self.cpu_device)
        total_generated_latent_frames = 0 
        history_pixels_on_cpu = None    

        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        # --- Main Sampling Loop ---
        for i, latent_padding in enumerate(latent_paddings):
            is_last_section_in_padding_logic = (latent_padding == 0)
            current_latent_padding_size = latent_padding * latent_window_size
            print(f"  Processing section {i+1}/{total_latent_sections} (padding: {latent_padding}), is_last_section_logic: {is_last_section_in_padding_logic}")

            # Load Transformer for sampling if in low VRAM mode
            if self.low_vram_mode and self.device.type == "cuda":
                print("Low VRAM: Clearing other models and loading Transformer to GPU...")
                # Unload other large models if they were on GPU. VAE, TextEncoders, ImageEncoder.
                # Explicitly move them to CPU to be sure, rather than relying on unload_complete_models' default list.
                self.vae.to(self.cpu_device)
                self.text_encoder.to(self.cpu_device); self.text_encoder_2.to(self.cpu_device)
                self.image_encoder.to(self.cpu_device)
                # Ensure enough memory is preserved for the transformer
                move_model_to_device_with_memory_preservation(self.transformer, target_device=self.device, 
                                                              preserved_memory_gb=DEFAULT_GPU_PRESERVED_MEMORY_TRANSFORMER_LOAD_GB)
            
            target_transformer = self.transformer.to(self.device) # Ensure it's on device for sampling

            indices = torch.arange(0, sum([1, current_latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0).to(self.device)
            idx_parts = indices.split([1, current_latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices_pre, _, current_latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = idx_parts
            current_clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre_cond = start_latent # Already on self.device
            history_slice_for_cond_cpu = history_latents_cpu[:, :, :1 + 2 + 16, :, :]
            clean_latents_post_cond_cpu, clean_latents_2x_cond_cpu, clean_latents_4x_cond_cpu = history_slice_for_cond_cpu.split([1, 2, 16], dim=2)
            
            current_clean_latents_cond = torch.cat([clean_latents_pre_cond, clean_latents_post_cond_cpu.to(self.device, dtype=transformer_dtype)], dim=2)
            current_clean_latents_2x_cond = clean_latents_2x_cond_cpu.to(self.device, dtype=transformer_dtype)
            current_clean_latents_4x_cond = clean_latents_4x_cond_cpu.to(self.device, dtype=transformer_dtype)

            if use_teacache:
                target_transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                target_transformer.initialize_teacache(enable_teacache=False)
            
            def sampling_callback_log(d):
                print(f"    Sampling step {d['i']+1}/{steps} for section {i+1}")

            generated_latents_gpu = sample_hunyuan(
                transformer=target_transformer, sampler='unipc', width=width, height=height, frames=num_pixel_frames_per_segment,
                real_guidance_scale=cfg_scale, distilled_guidance_scale=distilled_cfg_scale, guidance_rescale=cfg_rescale,
                num_inference_steps=steps, generator=rnd_generator,
                prompt_embeds=llama_vec, prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n, negative_prompt_poolers=clip_l_pooler_n,
                device=self.device, dtype=transformer_dtype, image_embeddings=image_encoder_last_hidden_state,
                latent_indices=current_latent_indices, clean_latents=current_clean_latents_cond, clean_latent_indices=current_clean_latent_indices,
                clean_latents_2x=current_clean_latents_2x_cond, clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=current_clean_latents_4x_cond, clean_latent_4x_indices=clean_latent_4x_indices,
                callback=sampling_callback_log)
            
            # Offload Transformer after sampling if in low VRAM mode
            if self.low_vram_mode and self.device.type == "cuda":
                print("Low VRAM: Offloading Transformer from GPU...")
                offload_model_from_device_for_memory_preservation(self.transformer, target_device=self.device, # target_device is where it is NOW
                                                                  preserved_memory_gb=DEFAULT_GPU_PRESERVED_MEMORY_TRANSFORMER_OFFLOAD_GB)
                self.transformer.to(self.cpu_device) # Ensure it's marked as on CPU

            generated_latents_current_segment_cpu = generated_latents_gpu.cpu()
            if is_last_section_in_padding_logic:
                generated_latents_current_segment_cpu = torch.cat([start_latent.cpu(), generated_latents_current_segment_cpu], dim=2)

            num_new_latents_in_segment = generated_latents_current_segment_cpu.shape[2]
            total_generated_latent_frames += num_new_latents_in_segment
            history_latents_cpu = torch.cat([generated_latents_current_segment_cpu, history_latents_cpu], dim=2)
            
            current_total_valid_latents_cpu = history_latents_cpu[:, :, :total_generated_latent_frames, :, :]
            
            # VAE decoding with management
            # _managed_vae_decode expects latents on GPU and returns pixels on CPU
            if history_pixels_on_cpu is None: 
                history_pixels_on_cpu = self._managed_vae_decode(current_total_valid_latents_cpu.to(self.device))
            else:
                newly_decoded_pixels_segment_cpu = self._managed_vae_decode(generated_latents_current_segment_cpu.to(self.device))
                history_pixels_on_cpu = soft_append_bcthw(newly_decoded_pixels_segment_cpu, history_pixels_on_cpu, num_pixel_frames_per_segment)

            save_bcthw_as_mp4(history_pixels_on_cpu, final_output_path_str, fps=30, crf=mp4_crf)
            print(f"  Updated video saved: {final_output_path_str}, total pixel frames: {history_pixels_on_cpu.shape[2]}")

            if is_last_section_in_padding_logic:
                break 
        
        if not os.path.exists(final_output_path_str) or (history_pixels_on_cpu is not None and history_pixels_on_cpu.shape[2] == 0):
            if history_pixels_on_cpu is not None and history_pixels_on_cpu.shape[2] > 0:
                 save_bcthw_as_mp4(history_pixels_on_cpu, final_output_path_str, fps=30, crf=mp4_crf) 
                 if not os.path.exists(final_output_path_str): raise RuntimeError("Video generation failed to save output.")
            else:
                raise RuntimeError("Video generation failed: No frames were produced or saved.")
            
        # Final cleanup in low VRAM: ensure all large models are on CPU if not already.
        if self.low_vram_mode and self.device.type == "cuda":
            print("Low VRAM: Ensuring all models are offloaded from GPU post-prediction...")
            self.transformer.to(self.cpu_device)
            self.vae.to(self.cpu_device)
            self.text_encoder.to(self.cpu_device); self.text_encoder_2.to(self.cpu_device)
            self.image_encoder.to(self.cpu_device)
            torch.cuda.empty_cache() # Clear any cached memory

        return Path(final_output_path_str)
