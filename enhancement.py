"""Exposure-adaptive low-light enhancement for BGR camera frames, without weights."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EnhancementResult:
    image: np.ndarray
    applied: bool
    brightness: float
    gamma: float


class AdaptiveLowLightEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self._gamma = None
        self._brightness = None

    def process(self, image, strength=0.65):
        if image is None or image.size == 0 or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("暗光增强需要非空 uint8 BGR 图像")
        strength = float(np.clip(strength, 0.0, 1.0))
        # Bounded-cost exposure statistics on a thumbnail.
        h, w = image.shape[:2]
        scale = min(1.0, 256.0 / max(h, w))
        thumbnail = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        luma_small = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        brightness = float(np.percentile(luma_small, 60))
        # Leave normal/daylight frames alone; black frames contain no recoverable detail.
        if strength == 0 or brightness >= 0.42 or float(luma_small.max()) < 0.015:
            self.reset()
            return EnhancementResult(image, False, brightness, 1.0)
        target_gamma = float(np.clip(np.log(0.42) / np.log(max(brightness, 0.02)), 0.45, 1.0))
        target_gamma = 1.0 - strength * (1.0 - target_gamma)
        # Smooth modest exposure changes; reset immediately on scene cuts.
        if self._gamma is None or abs(brightness - self._brightness) > 0.12:
            gamma = target_gamma
        else:
            gamma = 0.75 * self._gamma + 0.25 * target_gamma
        self._gamma, self._brightness = gamma, brightness
        # Mild edge-preserving denoising only in frames needing enhancement.
        # OpenCV's uint8 SIMD path avoids several full-resolution float RGB buffers.
        smooth = cv2.bilateralFilter(image, 5, 10, 3)
        base = cv2.addWeighted(image, 1 - 0.2 * strength, smooth, 0.2 * strength, 0)
        illumination = cv2.GaussianBlur(luma_small, (0, 0), 3)
        illumination = cv2.resize(illumination, (w, h), interpolation=cv2.INTER_LINEAR)
        gain = np.power(np.maximum(illumination, 0.025), gamma - 1)
        gain = np.minimum(gain, 1 + 2.0 * strength)
        # Scale B/G/R equally, capping gain at local highlight headroom.
        blue, green, red = cv2.split(base)
        brightest = cv2.max(cv2.max(blue, green), red).astype(np.float32)
        headroom = 255.0 / np.maximum(brightest, 1.0)
        gain = np.minimum(gain, headroom)
        corrected = cv2.multiply(base, cv2.merge((gain, gain, gain)), dtype=cv2.CV_32F)
        output = cv2.convertScaleAbs(corrected)
        applied = cv2.norm(output, image, cv2.NORM_L1) / image.size >= 0.5
        return EnhancementResult(output if applied else image, applied, brightness, gamma)

    def enhance(self, image, strength=0.65):
        return self.process(image, strength).image
