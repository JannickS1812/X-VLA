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
import numpy as np, torch, random
from mmengine import fileio
from scipy.interpolate import interp1d
from ..utils import read_video_to_frames, read_parquet, quat_to_rotate6d, euler_to_rotate6d
from PIL import Image
from .base import DomainHandler
from PIL import Image
from typing import Any

class AGIBOTLeRobotHandler(DomainHandler):
    dataset_name = "libero_lerobot"

    @staticmethod
    def _pil_from_arr(arr: Any) -> Image.Image:
        from ..utils import decode_image_from_bytes
        return decode_image_from_bytes(arr) if not isinstance(arr, Image.Image) else arr

    def iter_episode(self, traj_idx: int, *, num_actions: int, training: bool,
                     image_aug, lang_aug_map: dict | None, **kwargs):
        item = self.meta["datalist"][traj_idx]
        ep = item["episode_index"]; chunk = f"chunk-{ep//1000:03d}"; key = f"episode_{ep:06d}"

        pq_path = fileio.join_path(item["top_path"], "data", chunk, key + ".parquet")
        vkeys = ["observation.images.image", "observation.images.wrist_image"]
        vid_paths = [fileio.join_path(item["top_path"], "videos", chunk, k, key + ".mp4") for k in vkeys]
        images = [read_video_to_frames(p) for p in vid_paths]
        image_mask = torch.ones(self.num_views, dtype=torch.bool)

        data = read_parquet(pq_path)
        left_pos = np.asarray(data["action"])[:, :3]
        left_rot = euler_to_rotate6d(np.asarray(data["action"])[:, 3:6])
        left_grip = np.asarray(data["action"])[:, -1:]
        left = np.concatenate([left_pos, left_rot, left_grip], axis=-1)
        right = np.zeros_like(left)
        start = 1; end = item["length"]
        idxs = list(range(start, end, 1))
        if training: random.shuffle(idxs)

        # We need eef position, rotation as rotate6d and gripper (10 dimensional vector)
        # Then we use it as the left arm and make the right arm np.zeros_like(left)
        # Then use base hdf5 version

        freq = 30.0; qdur = 1.0; 
        ins = item["tasks"][0].split(" | ")[0]
        
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:len(images)] = True
        lt = np.arange(left.shape[0], dtype=np.float64) / float(freq)
        rt = np.arange(right.shape[0], dtype=np.float64) / float(freq)

        # Interpolators; clamp to endpoints
        L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
        R = interp1d(rt, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
        ref = (lt + rt) / 2.0

        V = min(self.num_views, len(images))
        for idx in idxs:

            # Query future window
            cur = ref[idx]
            q = np.linspace(cur, min(cur + qdur, float(ref.max())), num_actions + 1, dtype=np.float32)
            lseq = torch.tensor(L(q))
            rseq = torch.tensor(R(q))

            # Skip static segments
            if (lseq[1] - lseq[0]).abs().max() < 1e-5 and (rseq[1] - rseq[0]).abs().max() < 1e-5: continue
            
            # Language augmentation
            if training and lang_aug_map and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])
            
            imgs = [image_aug(Image.fromarray(images[v][idx])) for v in range(V)]
            while len(imgs) < self.num_views: imgs.append(torch.zeros_like(imgs[0]))
            image_input = torch.stack(imgs, dim=0)

            yield {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": torch.cat([lseq, rseq], -1).float()
            }