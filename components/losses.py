from __future__ import annotations
from typing import Iterable, Tuple
import torch
import torch.nn as nn


def _ensure_indices_valid(D: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= D]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim D={D}")

class EE6DLoss(nn.Module):
    """
    End-effector layout with xyz, 6D rotation, and gripper channels.
    Uses:
      - position (MSE) on xyz pairs
      - rotation-6D (MSE) on two 6D segments
      - gripper (BCE-with-logits) on two gripper indices
    All hyperparameters/indices are hard-coded.
    """

    # ---- Hard-coded hyperparameters/indices ----
    DIM_ACTION: int = 20  # Expected action dimension
    GRIPPER_SCALE: float = 0.1
    GRIPPER_IDX: Tuple[int, int] = (9, 19)

    XYZ_SCALE: float = 100.0
    ROT_SCALE: float = 5.0

    POS_IDX_1: Tuple[int, int, int] = (0, 1, 2)
    POS_IDX_2: Tuple[int, int, int] = (10, 11, 12)

    ROT_IDX_1: Tuple[int, int, int, int, int, int] = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2: Tuple[int, int, int, int, int, int] = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_action: torch.Tensor, target_action: torch.Tensor):
        assert pred_action.shape == target_action.shape, "pred/target shapes must match"
        assert pred_action.dim() == 3, "expected [B, T, D]"
        B, T, D = pred_action.shape  # noqa: F841

        # Validate indices
        _ensure_indices_valid(D, self.GRIPPER_IDX, "gripper_idx")
        _ensure_indices_valid(D, self.POS_IDX_1, "pos_idx_1")
        _ensure_indices_valid(D, self.POS_IDX_2, "pos_idx_2")
        _ensure_indices_valid(D, self.ROT_IDX_1, "rot_idx_1")
        _ensure_indices_valid(D, self.ROT_IDX_2, "rot_idx_2")

        # Gripper BCE (average over both indices)
        g_losses = [
            self.bce(pred_action[:, :, gi], target_action[:, :, gi]) for gi in self.GRIPPER_IDX
        ]
        gripper_loss = sum(g_losses) / len(self.GRIPPER_IDX)
        gripper_loss = gripper_loss * self.GRIPPER_SCALE

        # Position xyz (two triplets)
        pos_loss_1 = self.mse(pred_action[:, :, self.POS_IDX_1], target_action[:, :, self.POS_IDX_1])
        pos_loss_2 = self.mse(pred_action[:, :, self.POS_IDX_2], target_action[:, :, self.POS_IDX_2])
        position_loss = (pos_loss_1 + pos_loss_2) * self.XYZ_SCALE

        # Rotation 6D (two segments)
        rot_loss_1 = self.mse(pred_action[:, :, self.ROT_IDX_1], target_action[:, :, self.ROT_IDX_1])
        rot_loss_2 = self.mse(pred_action[:, :, self.ROT_IDX_2], target_action[:, :, self.ROT_IDX_2])
        rotate6D_loss = (rot_loss_1 + rot_loss_2) * self.ROT_SCALE

        total = position_loss + rotate6D_loss + gripper_loss
        return {
            "position_loss": position_loss,
            "rotate6D_loss": rotate6D_loss,
            "gripper_loss": gripper_loss,
            # "total_loss": total
        }


class JointLoss(nn.Module):
    """
    Joint-space layout with joints + gripper only.
    Uses:
      - joints (MSE) over all non-gripper channels
      - gripper (BCE-with-logits) over two gripper indices
    All hyperparameters/indices are hard-coded.
    """

    # ---- Hard-coded hyperparameters/indices ----
    DIM_ACTION: int = 14  # Expected action dimension
    GRIPPER_SCALE: float = 0.1
    GRIPPER_IDX: Tuple[int, int] = (6, 13)
    JOINTS_SCALE: float = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_action: torch.Tensor, target_action: torch.Tensor):
        assert pred_action.shape == target_action.shape, "pred/target shapes must match"
        assert pred_action.dim() == 3, "expected [B, T, D]"
        B, T, D = pred_action.shape  # noqa: F841

        # Validate
        _ensure_indices_valid(D, self.GRIPPER_IDX, "gripper_idx")

        # Gripper BCE (average over both indices)
        g_losses = [
            self.bce(pred_action[:, :, gi], target_action[:, :, gi]) for gi in self.GRIPPER_IDX
        ]
        gripper_loss = sum(g_losses) / len(self.GRIPPER_IDX)
        gripper_loss = gripper_loss * self.GRIPPER_SCALE

        # Joints = all except grippers
        grip_set = set(self.GRIPPER_IDX)
        joints_idx = tuple(i for i in range(D) if i not in grip_set)
        if len(joints_idx) == 0:
            raise ValueError("No joint indices inferred (D equals number of gripper indices).")

        joints_loss = self.mse(pred_action[:, :, joints_idx], target_action[:, :, joints_idx])
        joints_loss = joints_loss * self.JOINTS_SCALE

        total = joints_loss + gripper_loss
        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
            # "total_loss": total,
        }



class AGIBOTJointLoss(nn.Module):
    """
    Joint-space layout with joints + gripper only.
    Uses:
      - joints (MSE) over all non-gripper channels
      - gripper (BCE-with-logits) over two gripper indices
    All hyperparameters/indices are hard-coded.
    """

    # ---- Hard-coded hyperparameters/indices ----
    DIM_ACTION: int = 16  # Expected action dimension
    GRIPPER_SCALE: float = 0.1
    GRIPPER_IDX: Tuple[int, int] = (14, 15)
    JOINTS_SCALE: float = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_action: torch.Tensor, target_action: torch.Tensor):
        assert pred_action.shape == target_action.shape, "pred/target shapes must match"
        assert pred_action.dim() == 3, "expected [B, T, D]"
        B, T, D = pred_action.shape  # noqa: F841

        # Validate
        _ensure_indices_valid(D, self.GRIPPER_IDX, "gripper_idx")

        # Gripper BCE (average over both indices)
        g_losses = [
            self.bce(pred_action[:, :, gi], target_action[:, :, gi]) for gi in self.GRIPPER_IDX
        ]
        gripper_loss = sum(g_losses) / len(self.GRIPPER_IDX)
        gripper_loss = gripper_loss * self.GRIPPER_SCALE

        # Joints = all except grippers
        grip_set = set(self.GRIPPER_IDX)
        joints_idx = tuple(i for i in range(D) if i not in grip_set)
        if len(joints_idx) == 0:
            raise ValueError("No joint indices inferred (D equals number of gripper indices).")

        joints_loss = self.mse(pred_action[:, :, joints_idx], target_action[:, :, joints_idx])
        joints_loss = joints_loss * self.JOINTS_SCALE

        total = joints_loss + gripper_loss
        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
            # "total_loss": total,
        }


