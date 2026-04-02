"""Core Python reference implementation for the course submission."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class CannyResult:
    gradient: np.ndarray
    nms: np.ndarray
    edges: np.ndarray


@dataclass
class MatchResult:
    keypoints_a: Sequence[cv2.KeyPoint]
    keypoints_b: Sequence[cv2.KeyPoint]
    matches: List[cv2.DMatch]
    inliers: List[cv2.DMatch]
    homography: Optional[np.ndarray]
    aligned: Optional[np.ndarray]


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def manual_nms(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    h, w = magnitude.shape
    out = np.zeros_like(magnitude, dtype=np.float32)
    angle = np.rad2deg(angle) % 180
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            current = magnitude[y, x]
            direction = angle[y, x]
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                q, r = magnitude[y, x + 1], magnitude[y, x - 1]
            elif 22.5 <= direction < 67.5:
                q, r = magnitude[y + 1, x - 1], magnitude[y - 1, x + 1]
            elif 67.5 <= direction < 112.5:
                q, r = magnitude[y + 1, x], magnitude[y - 1, x]
            else:
                q, r = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
            if current >= q and current >= r:
                out[y, x] = current
    return out


def run_canny_pipeline(image: np.ndarray, low: int = 45, high: int = 110) -> CannyResult:
    gray = to_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    angle = cv2.phase(gx, gy, angleInDegrees=False)
    nms = manual_nms(magnitude, angle)
    edges = cv2.Canny(blurred, low, high)
    return CannyResult(gradient=magnitude, nms=nms, edges=edges)


def draw_keypoints(image: np.ndarray, keypoints: Sequence[cv2.KeyPoint], color: Tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    for kp in keypoints:
        center = (int(round(kp.pt[0])), int(round(kp.pt[1])))
        radius = max(3, int(round(kp.size / 2)))
        cv2.circle(canvas, center, radius, color, 1, lineType=cv2.LINE_AA)
    return canvas


def detect_harris(image: np.ndarray, max_points: int = 200):
    gray = np.float32(to_gray(image))
    response = cv2.cornerHarris(gray, 2, 3, 0.04)
    response = cv2.dilate(response, None)
    threshold = np.percentile(response, 98)
    ys, xs = np.where(response >= threshold)
    keypoints = [cv2.KeyPoint(float(x), float(y), 7, response=float(response[y, x])) for y, x in zip(ys.tolist(), xs.tolist())]
    keypoints.sort(key=lambda kp: kp.response, reverse=True)
    return keypoints[:max_points], response


def detect_sift(image: np.ndarray, max_points: int = 200):
    gray = to_gray(image)
    sift = cv2.SIFT_create(nfeatures=max_points)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints[:max_points], descriptors[:max_points] if descriptors is not None else None


def match_two_images(image_a: np.ndarray, image_b: np.ndarray, max_points: int = 250) -> MatchResult:
    keypoints_a, desc_a = detect_sift(image_a, max_points)
    keypoints_b, desc_b = detect_sift(image_b, max_points)
    if desc_a is None or desc_b is None:
        return MatchResult(keypoints_a, keypoints_b, [], [], None, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for m, n in raw_pairs if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return MatchResult(keypoints_a, keypoints_b, good, [], None, None)

    src = np.float32([keypoints_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([keypoints_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    inliers = [m for m, keep in zip(good, mask.ravel().tolist()) if keep] if mask is not None else []
    aligned = cv2.warpPerspective(image_a, homography, (image_b.shape[1], image_b.shape[0])) if homography is not None else None
    return MatchResult(keypoints_a, keypoints_b, good, inliers, homography, aligned)


def blend_pair(base: np.ndarray, warped: np.ndarray, mode: str = "average") -> np.ndarray:
    base_mask = (base.sum(axis=2) > 0).astype(np.float32)[..., None]
    warp_mask = (warped.sum(axis=2) > 0).astype(np.float32)[..., None]
    if mode == "overlay":
        return np.where(warp_mask > 0, warped, base)
    if mode == "average":
        total = base_mask + warp_mask
        total[total == 0] = 1
        return ((base.astype(np.float32) + warped.astype(np.float32)) / total).astype(np.uint8)

    x = np.linspace(0.0, 1.0, base.shape[1], dtype=np.float32)
    feather = np.minimum(x, x[::-1])[None, :, None]
    weight_base = np.where(base_mask > 0, feather + 1e-3, 0)
    weight_warp = np.where(warp_mask > 0, feather[:, ::-1, :] + 1e-3, 0)
    total = weight_base + weight_warp
    total[total == 0] = 1
    return ((base.astype(np.float32) * weight_base + warped.astype(np.float32) * weight_warp) / total).astype(np.uint8)


def stitch_images(images: Sequence[np.ndarray], mode: str = "average") -> np.ndarray:
    panorama = images[0]
    for next_image in images[1:]:
        result = match_two_images(panorama, next_image)
        if result.homography is None:
            continue
        h = max(panorama.shape[0], next_image.shape[0])
        w = panorama.shape[1] + next_image.shape[1]
        warped = cv2.warpPerspective(panorama, result.homography, (w, h))
        canvas = np.zeros_like(warped)
        canvas[: next_image.shape[0], : next_image.shape[1]] = next_image
        panorama = blend_pair(canvas, warped, mode)
    return panorama
