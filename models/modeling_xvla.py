#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

import logging
import traceback
import time
from typing import Any, Dict, List
import logging

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn
import json_numpy
import cv2

from transformers import PreTrainedModel
from .modeling_florence2 import Florence2ForConditionalGeneration
from .transformer import SoftPromptedTransformer
from .action_hub import build_action_space
from .configuration_xvla import XVLAConfig

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- New Pydantic Models for Batched Request ---

class SingleActPayload(BaseModel):
    """Pydantic model for a single request payload."""
    proprio: str  # json_numpy dump
    language_instruction: str
    image0: str   # json_numpy dump
    image1: str | None = None
    image2: str | None = None
    domain_id: int
    steps: int = 10
    skip_action_generation: bool = False

class BatchActResponse(BaseModel):
    """Pydantic model for a single item in the batched response."""
    action: list
    embedding: list

# --- End New Pydantic Models ---


class XVLA(PreTrainedModel):
    """
    XVLA: HuggingFace-compatible Vision-Language-Action policy.
    ... (rest of the class is identical to xvla_model.py) ...
    """
    config_class = XVLAConfig
    base_model_prefix = "xvla"
    supports_gradient_checkpointing = True

    def __init__(self, config: XVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        # Action space (dimensions + hooks)
        self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        # Florence2 backbone (encoder only)
        self.vlm = Florence2ForConditionalGeneration(config.florence_config).to(torch.float32)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        # Temporal/action head
        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )

        # Deferred FastAPI app
        self.app: FastAPI | None = None

    # ============================= Florence2 encoder =============================
    def forward_vlm(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text + multi-view images via Florence2 encoder.

        Returns:
          { "vlm_features": [B, T_enc, D], "aux_visual_inputs": [B, (V-1)*N, D] }
        """
        B, V = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(torch.bool)         # [B*V]
        flat_images = pixel_values.flatten(0, 1)                # [B*V, C, H, W]

        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]                   # [#valid, C, H, W]
        valid_feats = self.vlm._encode_image(valid_images)      # [#valid, N, D]
        N, D = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((B * V, N, D))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(B, V, N, D)        # [B, V, N, D]

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)  # [B, L, D]

        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],  # first view: [B, N, D]
            inputs_embeds,         # [B, L, D]
        )

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]  # [B, T_enc, D]

        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)  # remaining views flattened
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    # ================================= training =================================
    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,  # [B, T=num_actions, D=dim_action]
    ) -> Dict[str, torch.Tensor]:
        """
        1) Encode multimodal inputs.
        2) Diffusion-style noisy mixture of actions: x_t = t*noise + (1-t)*gt.
        3) Space-specific preprocessing, prediction, and supervised loss.
        """
        enc = self.forward_vlm(input_ids, image_input, image_mask)

        B = input_ids.shape[0]
        t = (torch.rand(1, device=input_ids.device)
             + torch.arange(B, device=input_ids.device) / B) % (1 - 1e-5)

        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )
        return self.action_space.compute_loss(pred_action, action)

    # ================================= inference =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Iterative denoising (linear schedule).
        Applies action_space.postprocess at the end (e.g., sigmoid on gripper).
        """
        self.eval()
        
        # Determine if we need to synchronize (only for CUDA devices)
        is_cuda = proprio.is_cuda 
            
        t_vlm_start = time.perf_counter()
        enc = self.forward_vlm(input_ids, image_input, image_mask)
        log.info(f"  VLM Encode Time: {(time.perf_counter() - t_vlm_start) * 1000:.2f} ms")


        B = input_ids.shape[0]
        D = self.action_space.dim_action

        x1 = torch.randn(B, self.num_actions, D, device=proprio.device, dtype=proprio.dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        
        # Start total denoising timer
        t_denoise_total_start = time.perf_counter()
        
        for i in range(steps, 0, -1):
            t_step_start = time.perf_counter()
            
            t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
            
            t_step_end = time.perf_counter()
            # Step index: steps - i + 1 goes from 1 to `steps`
            log.info(f"  Denoise Step {steps - i + 1}/{steps}: {(t_step_end - t_step_start) * 1000:.2f} ms")

        t_denoise_total_end = time.perf_counter()
        log.info(f"  Total Denoise Steps Time: {(t_denoise_total_end - t_denoise_total_start) * 1000:.2f} ms")
        
        return self.action_space.postprocess(action), enc

    # =============================== FastAPI service =============================
    def _build_app(self, processor):
        """
        Minimal FastAPI app for XVLA inference.
        NOW includes a batched endpoint.

        Args:
            processor: callable(images, text) -> Dict[str, torch.Tensor]
                       expected keys: "input_ids", "image_input", "image_mask"
        """
        if self.app is not None:
            return

        app = FastAPI()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        def _decode_images(payload: SingleActPayload) -> List[Image.Image]:
            """Decodes images from a single payload."""
            images = []
            # Use .model_dump() for Pydantic v2
            payload_dict = payload.model_dump() 
            for key in ("image0", "image1", "image2"):
                if key not in payload_dict or payload_dict[key] is None: 
                    continue
                v = json_numpy.loads(payload_dict[key])
                if isinstance(v, np.ndarray):
                    if v.ndim == 1:  # encoded bytes
                        v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                    images.append(Image.fromarray(v))
                elif isinstance(v, (list, tuple)):
                    images.append(Image.fromarray(np.array(v)))
                elif isinstance(v, str):
                    images.append(Image.open(v))
            if not images:
                raise ValueError("No valid images found in payload.")
            return images

        def to_model(t: torch.Tensor) -> torch.Tensor:
            """Helper to move tensor to correct device and dtype."""
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            # cast floats to model dtype, keep integral/bool as-is
            return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

        # --- NEW BATCHED ENDPOINT ---
        # --- FIX: Changed route to match your error log ---
        @app.post("/act_batched", response_model=List[BatchActResponse])
        def act_batch(payloads: List[SingleActPayload]):
            """
            Processes a batch of inference requests at once.
            """
            t_start = time.perf_counter() # Start total timer
            try:
                self.eval()
                
                batch_size = len(payloads)
                if batch_size == 0:
                    return []

                # --- 1. Build batches by looping over payloads (Collation) ---
                t_collate_start = time.perf_counter()

                # --- 1. Build batches by looping over payloads ---
                all_proc_inputs = []
                all_proprio = []
                all_domain_id = []
                steps = payloads[0].steps # Assume all have the same step count
                
                for payload in payloads:
                    # 1a. Decode images for this sample
                    images = _decode_images(payload)
                    
                    # 1b. Call processor for this sample (B=1)
                    inputs = processor(images, payload.language_instruction)
                    all_proc_inputs.append(inputs)
                    
                    # 1c. Decode proprio
                    proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload.proprio)))
                    all_proprio.append(proprio)
                    
                    # 1d. Get domain ID
                    all_domain_id.append(torch.tensor([int(payload.domain_id)], dtype=torch.long))

                t_collate_end = time.perf_counter()
                
                # --- 2. Collate into single batched tensors ---
                t_stack_start = time.perf_counter()

                batched_inputs = {
                    "input_ids": to_model(torch.stack([d["input_ids"].squeeze(0) for d in all_proc_inputs])),
                    "image_input": to_model(torch.stack([d["image_input"].squeeze(0) for d in all_proc_inputs])),
                    "image_mask": to_model(torch.stack([d["image_mask"].squeeze(0) for d in all_proc_inputs])),
                    "proprio": to_model(torch.stack(all_proprio)),
                    "domain_id": to_model(torch.cat(all_domain_id)),
                }
                
                t_stack_end = time.perf_counter()
                
                # --- 3. Run single batched inference ---
                skip_action_generation = all([p.skip_action_generation for p in payloads])
                
                t_infer_start = time.perf_counter()

                if skip_action_generation:
                    action_dim = self.action_space.dim_action
                    action_len = self.num_actions
                    action_tensor = torch.zeros(
                        (batch_size, action_len, action_dim), 
                        dtype=dtype, 
                        device=device
                    )
                    
                    with torch.no_grad():
                        enc = self.forward_vlm(
                            batched_inputs["input_ids"],
                            batched_inputs["image_input"],
                            batched_inputs["image_mask"],
                        )
                    vlm_features = enc["vlm_features"]  # Shape: [B, Seq, Dim]
                    vlm_embedding_tensor = vlm_features[:, -1, :] # Shape: [B, Dim]
                else:
                    action_tensor, enc = self.generate_actions(**batched_inputs, steps=steps)

                    # Extract VLM embeddings
                    vlm_features = enc["vlm_features"]  # Shape: [B, Seq, Dim]
                    vlm_embedding_tensor = vlm_features[:, -1, :] # Shape: [B, Dim]

                t_infer_end = time.perf_counter()

                # --- 4. Un-batch results and format response ---
                t_format_start = time.perf_counter()

                response_list = []
                for i in range(batch_size):
                    action = action_tensor[i].float().cpu().numpy()
                    embedding = vlm_embedding_tensor[i].float().cpu().numpy()
                    response_list.append(
                        BatchActResponse(action=action.tolist(), embedding=embedding.tolist())
                    )
                
                t_format_end = time.perf_counter()
                
                # --- 5. Log Timings ---
                t_total_end = time.perf_counter()
                
                log.info(f"--- Batch Profile (B={batch_size}) ---")
                log.info(f"Collation Loop: {(t_collate_end - t_collate_start) * 1000:.2f} ms")
                log.info(f"Tensor Stacking:  {(t_stack_end - t_stack_start) * 1000:.2f} ms")
                log.info(f"GPU Inference:    {(t_infer_end - t_infer_start) * 1000:.2f} ms")
                log.info(f"Response Format:  {(t_format_end - t_format_start) * 1000:.2f} ms")
                log.info(f"---------------------------------")
                log.info(f"Total Request:  {(t_total_end - t_start) * 1000:.2f} ms")
                log.info(f"Time Per Item:  {((t_total_end - t_start) * 1000) / batch_size:.2f} ms/item")
                
                return response_list

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Batched request failed"}, status_code=500)
        
        # --- Original Single Endpoint (for compatibility) ---
        @app.post("/act")
        def act(payload: Dict[str, Any]):
            # This is a fallback wrapper for the new batched Pydantic model
            try:
                single_payload = SingleActPayload(**payload)
            except Exception as e:
                logging.error(f"Payload validation error: {e}")
                return JSONResponse({"error": f"Invalid payload: {e}"}, status_code=400)
                
            # Call the batched endpoint with a list of one
            batched_response = act_batch([single_payload])
            
            # Return the first (and only) result
            if isinstance(batched_response, list) and len(batched_response) > 0:
                # Need to convert Pydantic model back to dict for JSONResponse
                return batched_response[0].model_dump() 
            else:
                logging.error("Batched endpoint failed to return valid single response.")
                return JSONResponse({"error": "Request failed during batched processing"}, status_code=500)

        self.app = app

    def run(self, processor, host: str = "0.0.0.0", port: int = 8000):
        """
        Launch the FastAPI service.
        """
        self._build_app(processor)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)