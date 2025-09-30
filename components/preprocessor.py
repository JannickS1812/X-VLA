# ------------------------------------------------------------------------------
# Copyright (c) 2022–∞, 2toINF
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

import torch
import torch.nn as nn
from typing import List
from transformers import AutoProcessor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ----------------------------- Language Preprocessor --------------------------

class LanguagePreprocessor:
    """
    A lightweight wrapper for tokenizing natural language instructions.

    Parameters
    ----------
    encoder_name : str
        Hugging Face model name or local path (e.g. "bert-base-uncased").
    device : str, default="cuda"
        Device to move the tokenized tensors to ("cuda" or "cpu").
    max_length : int, default=50
        Maximum sequence length for tokenization.

    Methods
    -------
    encode_language(language_instruction: List[str]) -> Dict[str, torch.Tensor]
        Tokenizes a batch of instructions into input IDs.
    """

    def __init__(self, encoder_name: str, max_length: int = 50):
        self.preprocessor = AutoProcessor.from_pretrained(encoder_name, trust_remote_code=True)
        self.max_length = max_length

    @torch.no_grad()
    def encode_language(self, language_instruction: List[str]):
        """
        Tokenize a list of language instructions.

        Parameters
        ----------
        language_instruction : List[str]
            List of natural language commands/instructions.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with:
            - "input_ids": tensor of shape [B, max_length], on self.device.
        """
        if isinstance(language_instruction, str): language_instruction = [language_instruction]
        inputs = self.preprocessor.tokenizer(
            language_instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        return {"input_ids": inputs["input_ids"]}


# ----------------------------- Image Preprocessor -----------------------------


class ImagePreprocessor:
    """
    Prepares a fixed number of image views with normalization.

    Parameters
    ----------
    num_views : int, default=3
        Number of image views expected. If fewer are provided, zero-padded
        placeholders are added.

    Methods
    -------
    __call__(images: List[Image]) -> Dict[str, torch.Tensor]
        Applies preprocessing to a batch of images.
    """

    def __init__(self, num_views: int = 3, version="v1"):
        self.num_views = num_views
        if version == "v0":
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    inplace=True,
                ),
            ])
        elif version == "v1":
            self.image_transform = transforms.Compose([
                transforms.Resize((236, 236), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    inplace=True,
                ),
            ])
        else: raise ValueError(f"Unknown image preprocessor version: {version}")

    def __call__(self, images):
        """
        Preprocess and pad a list of images.

        Parameters
        ----------
        images : List[PIL.Image]
            List of input images.

        Returns
        -------
        Dict[str, torch.Tensor]
            - "image_input": tensor of shape [1, num_views, 3, 224, 224].
              If fewer than num_views are provided, zero-padding is applied.
            - "image_mask": tensor of shape [1, num_views], bool type,
              indicating which slots correspond to valid images.
        """
        # Apply transforms to each provided image
        x = torch.stack([self.image_transform(img) for img in images])

        # Pad with zero-images if fewer than num_views are given
        V_exist = x.size(0)
        if V_exist < self.num_views:
            x = torch.cat(
                [x, x.new_zeros(self.num_views - V_exist, *x.shape[1:])],
                dim=0,
            )

        # Build image mask: True for valid slots, False for padded ones
        image_mask = torch.zeros(self.num_views, dtype=torch.bool, device=x.device)
        image_mask[:V_exist] = True

        return {
            "image_input": x.unsqueeze(0),   # [1, num_views, 3, 224, 224]
            "image_mask": image_mask.unsqueeze(0),  # [1, num_views]
        }