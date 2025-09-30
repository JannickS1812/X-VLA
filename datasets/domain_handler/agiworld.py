from __future__ import annotations
import numpy as np
import torch
import random
from mmengine import fileio
from scipy.interpolate import interp1d
from datasets.common import open_h5, quat_to_rotate6d
from PIL import Image
from .base import DomainHandler
import json
# Human Labeled Split file
SPLITFILE = "/mnt/petrelfs/zhengjinliang/Data/agiworld/on-site/human_split.json"

# Task instruction per domain
DOMAIN2INS = {
    "agiworld-on-site-pack": "Pick up the object and place it in the bag.",
    "agiworld-on-site-pack-extra": "Pick up the object and place it in the bag.",
    "agiworld-on-site-conveyor": "Pick objects from the conveyor belt and place them in the box.",
    "agiworld-on-site-conveyor-extra": "Pick objects from the conveyor belt and place them in the box.",
    "agiworld-on-site-restock": "Hang the snacks on the shelf.",
    "agiworld-on-site-pour": "pour the water into the cup.",
    "agiworld-on-site-microwave": "Open the microwave, put the food in, and close the microwave.",
    "agiworld-on-site-cloth": "fold the clothes.",
}

# Max chunk length per domain
DOMAIN2CHUNKSIZE = {
    "agiworld-on-site-pack": 60,
    "agiworld-on-site-pack-extra": 60,
    "agiworld-on-site-conveyor": 60,
    "agiworld-on-site-conveyor-extra": 60,
    "agiworld-on-site-restock": 60,
    "agiworld-on-site-pour": 90,
    "agiworld-on-site-microwave": 90,
    "agiworld-on-site-cloth": 90,
}

# Min chunk length around gripper-state changes
DOMAIN2MINCHUNK = {
    "agiworld-on-site-pack": 15,
    "agiworld-on-site-pack-extra": 15,
    "agiworld-on-site-conveyor": 10,
    "agiworld-on-site-conveyor-extra": 10,
    "agiworld-on-site-restock": 5,
    "agiworld-on-site-pour": 10,
    "agiworld-on-site-microwave": 15,
    "agiworld-on-site-cloth": 10,
}

class AGIWolrdHandler(DomainHandler):  # Note: "Wolrd" looks like a typo; kept for compatibility
    def read_action(self, item: str):
        """Read actions and end-effector states; return joint+gripper and 6D EE features."""
        # Different file layout for "extra" datasets
        action_path = (fileio.join_path(item, 'aligned_joints.h5')
                       if 'extra' not in self.meta['dataset_name']
                       else fileio.join_path(item, 'proprio_stats.h5'))
        
        with open_h5(str(action_path)) as f:
            try:
                # Some versions: grippers under action/effector/position (two columns: L, R)
                gripper_left = f['action']['effector']['position'][:, 0]   # [T]
                gripper_right = f['action']['effector']['position'][:, 1]  # [T]
            except Exception:
                # Fallback: split under state/left_effector and state/right_effector
                gripper_left = f['action']['left_effector']['position'][:, 0]
                gripper_right = f['action']['right_effector']['position'][:, 0]

            # assert gripper_left.max() <= 1.0 and gripper_left.min() >= 0.0, "gripper out of range [0, 1]"
            # assert gripper_right.max() <= 1.0 and gripper_right.min() >= 0.0, "gripper out of range [0, 1]"
            
            joints = f['state']['joint']['position'][:]                  # [T, 14]
            assert len(gripper_left) == joints.shape[0], "gripper/joint length mismatch"

            xyz_position_left = f['state']['end']['position'][:, 0]      # [T, 3]
            xyz_position_right = f['state']['end']['position'][:, 1]     # [T, 3]
            
            orientation_left = f['state']['end']['orientation'][:, 0]    # [T, 4]
            orientation_right = f['state']['end']['orientation'][:, 1]   # [T, 4]
            
            # Concatenate joints and grippers -> [T, 16]
            abs_joint = np.concatenate([joints,
                                        gripper_left[:, None],
                                        gripper_right[:, None]], axis=-1)

            # Build 6D rotations + XYZ + gripper for both arms
            abs_ee6d = np.concatenate([
                xyz_position_left, quat_to_rotate6d(orientation_left, scalar_first=True), gripper_left[:, None],
                xyz_position_right, quat_to_rotate6d(orientation_right, scalar_first=True), gripper_right[:, None]
            ], axis=-1)

        return abs_joint, abs_ee6d

    def iter_episode(self, traj_idx: int, *, num_actions: int, training: bool, action_mode,
                     image_aug, lang_aug_map: dict | None):
        """
        Sample multiple start indices within a trajectory, crop a window, resample to fixed steps,
        and yield model-ready samples.
        Windows are limited around gripper-state changes to avoid spanning multiple sub-tasks.
        """
        item = self.meta["datalist"][traj_idx]

        abs_joint, abs_ee6d = self.read_action(item)

        # Mark gripper changes across time (boolean over T-1)
        grippers = abs_joint[:, -2:]
        chg = np.any(grippers[1:] != grippers[:-1], axis=1)

        # Domain instruction
        ins = DOMAIN2INS[self.meta['dataset_name']]

        # Build (start, end) segment list; allow external split_list if provided
        
        current_ep_idx = int(item.split('/')[-1])
        try:
            with open(SPLITFILE, "r") as f: 
                split_data = json.load(f) 
        except: split_data = {}
        split_list = [0]
        if current_ep_idx in split_data:
            split_list.extend(split_data[current_ep_idx])
        split_list.append(len(abs_joint))
        # Drop very short segments
        split_list = [(a, b) for a, b in zip(split_list[:-1], split_list[1:]) if b - a > 10]
        random.shuffle(split_list)

        for traj_start_idx, traj_end_idx in split_list:
            # Candidate start indices; keep tail room
            index_list = list(range(traj_start_idx, 
                                        max(traj_start_idx, 
                                        traj_end_idx - DOMAIN2MINCHUNK[self.meta['dataset_name']])))
            random.shuffle(index_list)

            for idx in index_list:
                # Skip near-static consecutive frames (by EE6D)
                if np.abs(abs_ee6d[idx + 1] - abs_ee6d[idx]).max() < 5e-4:
                    continue

                # Initial window upper bound
                rel = min(DOMAIN2CHUNKSIZE[self.meta['dataset_name']] + 1,
                          traj_end_idx - idx)

                # Future nearest gripper change: ensure at least MINCHUNK after it
                right = np.flatnonzero(chg[idx:])  # offsets relative to idx
                if right.size > 0:
                    rel = min(right[0] + DOMAIN2MINCHUNK[self.meta['dataset_name']], rel)

                # Past nearest gripper change (use most recent: left[-1])
                left = np.flatnonzero(chg[:idx])
                if left.size > 0:
                    rel = min(idx - left[-1] + DOMAIN2MINCHUNK[self.meta['dataset_name']], rel)
                
                # Choose representation
                seg = abs_ee6d[idx:idx + rel] if action_mode == 'ee6d' else abs_joint[idx:idx + rel]

                # Linear interpolate to fixed length (inclusive endpoints -> +1)
                t_old = np.linspace(0.0, 1.0, seg.shape[0])
                t_new = np.linspace(0.0, 1.0, num_actions + 1)
                abs_trajectory = interp1d(t_old, seg, axis=0, kind='linear', bounds_error=False)(t_new)

                # Build image paths (different layouts for "extra")
                if 'extra' not in self.meta['dataset_name']:
                    image_names = ['head_color.jpg', 'hand_left_color.jpg', 'hand_right_color.jpg']
                    image_path = [fileio.join_path(item, f'camera/{idx}/{name}') for name in image_names]
                else:
                    image_names = ['head_color', 'hand_left_color', 'hand_right_color']
                    # Fix: remove stray extra arg, path must be videos/{name}/frame_{idx}.jpg
                    image_path = [fileio.join_path(item, f'videos/{name}/frame_{idx}.jpg') for name in image_names]

                # Mask size matches the number of loaded views
                image_mask = torch.ones(len(image_path), dtype=torch.bool)

                # Load, convert to RGB, apply augmentation, and stack
                image_input = torch.stack([
                    image_aug(Image.open(p).convert('RGB')) for p in image_path
                ])

                yield {
                    "language_instruction": ins,
                    "image_input": image_input,
                    "image_mask": image_mask,
                    "abs_trajectory": torch.from_numpy(abs_trajectory).float(),
                }