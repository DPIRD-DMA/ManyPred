"""A model with two inputs at different resolutions and three heads: a segmentation head, a
quarter-resolution regression head and an image-level classifier.

Inputs are a dict; the 20 m input's window is the 10 m window halved. Outputs come back per
head: dense heads blended on their own grid, the image-level head as TilePredictions.
"""

import numpy as np
import torch
import torch.nn.functional as F

from manypred import TilePredictions, predict_array, predict_batch


class TwoInputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.s2 = torch.nn.Conv2d(4, 8, 3, padding=1)
        self.s1 = torch.nn.Conv2d(2, 8, 3, padding=1)
        self.seg = torch.nn.Conv2d(8, 3, 1)
        self.height = torch.nn.Conv2d(8, 1, 1)
        self.scene = torch.nn.Linear(8, 5)

    def forward(self, x):
        s1 = F.interpolate(self.s1(x["s1"]), scale_factor=2, mode="nearest")
        features = F.gelu(self.s2(x["s2"]) + s1)
        return {
            "seg": self.seg(features),
            "height": F.avg_pool2d(self.height(features), 4),  # 1/4 resolution
            "scene": self.scene(features.mean(dim=(2, 3))),  # image-level
        }


rng = np.random.default_rng(0)
scene = {
    "s2": rng.random((4, 400, 480), dtype=np.float32),  # 10 m, the primary input
    "s1": rng.random((2, 200, 240), dtype=np.float32),  # 20 m: ratio 0.5, from the shapes
}
out = predict_array(
    TwoInputModel(),
    scene,
    patch_size=128,
    overlap=32,
    device="cpu",
    nodata=None,
    activation={"seg": "softmax", "scene": "softmax"},
    output={"seg": "argmax", "scene": "probs"},
)
print("seg", out["seg"].shape, "height", out["height"].shape)
assert out["seg"].shape == (1, 400, 480) and out["height"].shape == (1, 100, 120)

tiles = out["scene"]
assert isinstance(tiles, TilePredictions)
print("scene class probabilities per patch", tiles.values.shape)
print("scene mean over valid pixels", tiles.reduce("mean").round(3))
print("coarse map", tiles.to_map(stride=16).shape)

# postprocess runs once on the blended result: derived outputs are never blended
count = predict_array(
    TwoInputModel(),
    scene,
    patch_size=128,
    overlap=32,
    device="cpu",
    nodata=None,
    activation={"seg": "softmax", "scene": "softmax"},
    output={"seg": "argmax", "scene": "probs"},
    postprocess=lambda result: int((result["seg"] == 2).sum()),
    progress=False,
)
print("pixels of class 2:", count)

# batch mode with dict chips: image-level heads give (N, C)
chips = {"s2": scene["s2"][None, :, :128, :128], "s1": scene["s1"][None, :, :64, :64]}
batch = predict_batch(
    TwoInputModel(), chips, device="cpu", nodata=None, output={"scene": "logits"}, progress=False
)
assert batch["scene"].shape == (1, 5)
