# efficient_sam_backbone.py
import torch
import torch.nn as nn

from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec

# Adjust import if your EfficientSAM code is in a different location
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

@BACKBONE_REGISTRY.register()
def build_efficient_sam_backbone(cfg, input_shape: ShapeSpec):
    """
    Custom backbone builder for Detectron2 that wraps an EfficientSAM encoder.
    Reads cfg.MODEL.EFFICIENTSAM.VARIANT to decide between 'vitt' (ViT-Tiny) or 'vits' (ViT-Small).
    """
    variant = cfg.MODEL.EFFICIENTSAM.VARIANT  # e.g. "vitt" or "vits"

    if variant == "vitt":
        sam_model = build_efficient_sam_vitt()
    elif variant == "vits":
        sam_model = build_efficient_sam_vits()
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Optionally freeze or partially freeze the backbone
    # e.g. to keep it in eval mode:
    # sam_model.eval()

    return EfficientSAMBackbone(sam_model, cfg, input_shape)


class EfficientSAMBackbone(Backbone):
    """
    Wraps the EfficientSAM image_encoder as a Detectron2 Backbone.
    By default, we produce a single feature map "res5".
    For multi-scale (FPN) you'd adapt this code to produce multiple scales.
    """

    def __init__(self, sam_model, cfg, input_shape: ShapeSpec):
        super().__init__()
        self.sam_model = sam_model

        # We'll define one output feature: "res5".
        self._out_features = ["res5"]
        # Typically, the SAM encoder's neck outputs 256 channels
        self._out_feature_channels = {"res5": 256}
        # If patch_size=16, stride might be 16
        self._out_feature_strides = {"res5": 16}

    def forward(self, x: torch.Tensor):
        """
        x: shape [B, C, H, W], input images from Detectron2.
        We'll call the SAM image encoder.
        """
        # The SAM image_encoder typically expects 1024x1024, but can handle other sizes
        # via positional embedding interpolation. For instance segmentation on COCO,
        # you might set input size to 800x1333 or something similar.
        feats = self.sam_model.image_encoder(x)  # shape [B, 256, H', W']

        # Return a dict of {feature_name: feature_map}
        return {"res5": feats}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }
