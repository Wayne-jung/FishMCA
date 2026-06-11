import torch
import torch.nn as nn
import torch.nn.functional as F


class FastReIDTeacherWrapper(nn.Module):
    """Thin inference wrapper around a FastReID model.

    The Mask-ReID Teacher is trained outside the detector with FastReID on
    SAM2-masked instance crops. During student training this wrapper is frozen
    and only returns normalized teacher embeddings.
    """

    def __init__(self, config_file, checkpoint, device=None):
        super().__init__()
        try:
            from fastreid.config import get_cfg
            from fastreid.modeling import build_model
            from fastreid.utils.checkpoint import Checkpointer
        except ImportError as exc:
            raise ImportError(
                "FastReID is required when reid_teacher_config/checkpoint is enabled. "
                "Install FastReID or leave the ReID teacher config unset."
            ) from exc

        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.freeze()

        model = build_model(cfg)
        Checkpointer(model).load(checkpoint)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        if device is not None:
            model.to(device)
        self.model = model

    @torch.inference_mode()
    def forward(self, crops):
        outputs = self.model(crops)
        if isinstance(outputs, dict):
            for key in ("features", "feat", "embeddings", "embedding"):
                if key in outputs:
                    outputs = outputs[key]
                    break
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        return F.normalize(outputs, dim=-1)


def build_mask_reid_teacher(config_file=None, checkpoint=None, device=None):
    if not config_file or not checkpoint:
        return None
    return FastReIDTeacherWrapper(config_file, checkpoint, device=device)
