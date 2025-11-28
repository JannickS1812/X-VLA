#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# This script benchmarks local XVLA inference (no client/server)
# by running it in a loop and calculating statistics.
# ------------------------------------------------------------------------------

import argparse
import numpy as np
import torch
from PIL import Image
import os
import time  # <-- Import time for benchmarking

# --- Assumed Local Imports ---
try:
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor
except ImportError:
    print("Error: Could not import from `models/`. \n"
          "Please run this script from the root of your X-VLA fork, \n"
          "or ensure the `models` directory is in your PYTHONPATH.")
    exit(1)

def get_default_device():
    """Selects the default device (CUDA if available, otherwise CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def main():
    parser = argparse.ArgumentParser(description="Local XVLA Inference Benchmark")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="2toINF/X-VLA-WidowX",
        help="Path or Hugging Face Hub ID of the X-VLA model checkpoint."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        help="Device to load the model on (e.g., 'cuda', 'cpu', 'mps')."
    )
    parser.add_argument(
        "--num_requests",
        type=int,
        default=20,
        help="Number of inference requests to run."
    )
    
    args = parser.parse_args()

    # --- 1. Load Model and Processor ---
    print(f"Loading model and processor from: {args.model_path}")
    device = torch.device(args.device)
    
    processor = XVLAProcessor.from_pretrained(args.model_path)
    model = XVLA.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    
    num_actions = model.config.num_actions
    print(f"Model loaded successfully on {device}. (Action sequence length: {num_actions})")

    # --- THIS IS THE NEW LINE ---
    print("Compiling the model with torch.compile()... This may take a moment.")
    # Use 'reduce-overhead' mode for inference, which compiles faster.
    # For maximum speed (at the cost of longer compile time), use 'max-autotune'.
    model = torch.compile(model, mode="max-autotune")
    print("Model compilation complete.")
    # ----------------------------

    # --- 2. Prepare *Raw* Inputs (once) ---
    print("Preparing raw test inputs...")
    proprio_np = np.zeros(20, dtype=np.float32) # Using 20-dim proprio
    image_np = np.zeros((256, 256, 3), dtype=np.uint8)
    image_pil = Image.fromarray(image_np)
    instruction = "Move the gripper to the target position"
    domain_id = 0
    steps = 10
    images = [image_pil]
    
    timings_ms = []  # <-- List to store all timings
    
    # --- 3. Warm-up Request ---
    # Run the full process once to warm up CUDA, JIT caches, etc.
    print("Sending warm-up request...")
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Process
            inputs = processor(images, instruction)
            proprio_tensor = torch.as_tensor(proprio_np).unsqueeze(0).to(dtype=torch.bfloat16)
            domain_id_tensor = torch.tensor([domain_id], dtype=torch.long)
            inputs.update({"proprio": proprio_tensor, "domain_id": domain_id_tensor})
            
            # Move to device
            dtype = next(model.parameters()).dtype
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device=device)
                    if v.is_floating_point():
                        inputs[k] = inputs[k].to(dtype=dtype)
            
            # Infer
            model.generate_actions(**inputs, steps=steps)
        print("Warm-up complete. Starting benchmark.")
    except Exception as e:
        print(f"⚠️ Warm-up request failed: {e}. Aborting.")
        print("This might be the proprio_np dimension (expected 20) or action_hub.py mismatch.")
        exit()

    print("-" * 30)

    # --- 4. Main Test Loop ---
    print(f"🚀 Sending {args.num_requests} local inference requests...")
    for i in range(args.num_requests):
        start_time = time.perf_counter()
        
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # --- 3. Process Inputs ---
                inputs = processor(images, instruction)
                
                # --- FIX: Build proprio tensor with correct sequence length [B, T, D_proprio] ---
                proprio_tensor = torch.as_tensor(proprio_np).unsqueeze(0).to(dtype=torch.bfloat16)  # Shape: [1, 20]
                
                domain_id_tensor = torch.tensor([domain_id], dtype=torch.long)
                inputs.update({"proprio": proprio_tensor, "domain_id": domain_id_tensor})

                # Move all tensor inputs to the correct device
                dtype = next(model.parameters()).dtype
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device=device)
                        if v.is_floating_point():
                            inputs[k] = inputs[k].to(dtype=dtype)

                # --- 4. Run Inference ---
                action_tensor, enc = model.generate_actions(
                    **inputs, 
                    steps=steps
                )
            
            # --- Stop Timer ---
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            timings_ms.append(elapsed_ms)
            
            print(f"✅ Request {i+1}/{args.num_requests}: OK ({elapsed_ms:.2f} ms)")

        except Exception as e:
            print(f"⚠️ Request {i+1}/{args.num_requests} Failed: {e}")
            pass

    print("-" * 30)

    # --- 5. Calculate and Print Statistics ---
    if timings_ms:
        timings_np = np.array(timings_ms)
        print("📊 Local Benchmark Statistics:")
        print(f"  Total successful requests: {len(timings_np)}")
        print(f"  Average time: {np.mean(timings_np):.2f} ms")
        print(f"  Median time:  {np.median(timings_np):.2f} ms")
        print(f"  Min time:     {np.min(timings_np):.2f} ms")
        print(f"  Max time:     {np.max(timings_np):.2f} ms")
        print(f"  Std. Dev.:    {np.std(timings_np):.2f} ms")
    else:
        print("No successful requests were timed.")

if __name__ == "__main__":
    main()