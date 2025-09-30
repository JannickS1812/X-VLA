from __future__ import annotations
import numpy as np, torch, random
from mmengine import fileio
from scipy.interpolate import interp1d
from datasets.common import read_video_to_frames, read_parquet, quat_to_rotate6d, open_h5
from PIL import Image
from .base import DomainHandler
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

class AGIWolrdHandler(DomainHandler):
    dataset_name = "AGIBOT"
    
    def action_reader(self, traj_idx: int):
        item = self.meta["datalist"][traj_idx]
        candidates = [Path(fileio.join_path(item, n)) 
                    for n in ("aligned_joints.h5", "proprio_stats.h5")]
        path = next((p for p in candidates if p.exists()), None)
        if path is None: raise FileNotFoundError(f"No h5 file found in {item!r}")

        f = open_h5(str(path))
        data = {
            "actions.end.position": f["actions/end/position"][:],
            "actions.end.orientation": f["actions/end/orientation"][:],
            "actions.joint.position": f["actions/joint/position"][:],
            "actions.effector.position": f["actions/effector/position"][:],
            
            "state.joint.position": f["state/joint/position"][:],
            "state.end.position": f["state/end/position"][:],
            "state.effector.position": f["state/effector/position"][:]
        }
        return data
    
    def iter_episode(self, traj_idx: int, *, num_actions: int, training: bool,
                     image_aug, lang_aug_map: dict | None):
        item = self.meta["datalist"][traj_idx]
        
        ep = item["episode_index"]; chunk = f"chunk-{ep//1000:03d}"; key = f"episode_{ep:06d}"

        pq_path = fileio.join_path(item["top_path"], "data", chunk, key + ".parquet")
        vkeys = ["observation.images.head", "observation.images.hand_left", "observation.images.hand_right"]
        vid_paths = [fileio.join_path(item["top_path"], "videos", chunk, k, key + ".mp4") for k in vkeys]
        images = [read_video_to_frames(p) for p in vid_paths]
        image_mask = torch.ones(self.num_views, dtype=torch.bool)

        data = read_parquet(pq_path)
        pos = np.asarray(data["actions.end.position"])     # [T,2,3]
        ori = np.asarray(data["actions.end.orientation"])  # [T,2,4]
        grip = np.asarray(data["actions.effector.position"])  # [T,2]
        left = np.concatenate([pos[:,0], quat_to_rotate6d(ori[:,0]), grip[:, :1]], -1)
        right = np.concatenate([pos[:,1], quat_to_rotate6d(ori[:,1]), grip[:, 1:]], -1)

        freq = 30.0; qdur = 4.0; t = np.arange(left.shape[0], dtype=np.float64) / freq
        L = interp1d(t, left,  axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
        R = interp1d(t, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
        start = item["action_config"][0]["start_frame"]; end = item["action_config"][-1]["end_frame"] - 30
        idxs = list(range(start, end, 4 if training else 120))
        if training: random.shuffle(idxs)

        ins = item["tasks"][0].split(" | ")[0]

        for idx in idxs:
            imgs = []
            for v in range(min(self.num_views, len(images))):
                imgs.append(image_aug(Image.fromarray(images[v][idx])))
            while len(imgs) < self.num_views: imgs.append(torch.zeros_like(imgs[0]))
            image_input = torch.stack(imgs, 0)
            cur = t[idx]
            q = np.linspace(cur, min(cur + qdur, float(t.max())), num_actions + 1, dtype=np.float32)
            lseq, rseq = torch.tensor(L(q)), torch.tensor(R(q))
            if (lseq[1]-lseq[0]).abs().max() < 1e-5 and (rseq[1]-rseq[0]).abs().max() < 1e-5:continue
            if lang_aug_map is not None and ins in lang_aug_map: ins = random.choice(lang_aug_map[ins])
            yield {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": torch.cat([lseq, rseq], -1).float(),
            }
