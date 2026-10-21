# src/losses.py

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss：

        loss = mean(sqrt((prediction - target)^2 + eps^2))

    它是平滑且數值穩定的 L1 Loss 近似，
    適合影像重建與影片補幀。
    """

    def __init__(
        self,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()

        if eps <= 0:
            raise ValueError(
                f"eps must be greater than 0, got {eps}"
            )

        self.eps = float(eps)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(prediction, torch.Tensor):
            raise TypeError(
                "prediction must be a torch.Tensor."
            )

        if not isinstance(target, torch.Tensor):
            raise TypeError(
                "target must be a torch.Tensor."
            )

        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have the same shape, "
                f"got prediction={tuple(prediction.shape)} and "
                f"target={tuple(target.shape)}."
            )

        if not prediction.is_floating_point():
            raise TypeError(
                "prediction must use a floating-point dtype."
            )

        if not target.is_floating_point():
            raise TypeError(
                "target must use a floating-point dtype."
            )

        difference = prediction - target

        loss = torch.sqrt(
            difference.square()
            + self.eps * self.eps
        )

        return loss.mean()