"""
lip_sync_pipeline.py

Full replacement pipeline for real-time audio-driven mouth animation.
Landmark-aware (MediaPipe) with robust rectangle fallback.
Designed to be inserted into your existing frame-by-frame renderer.

Key features:
- Adaptive audio normalization (running max)
- Temporal smoothing (EMA) for stable mouth motion
- MediaPipe landmark path for accurate lips (if mediapipe installed)
- Fallback rectangular mouth deformation with texture-preserving blending
- Hooks for integration with Bark TTS (streaming audio)
- Optional notes where to plug model-based approaches (Wav2Lip/MuseTalk/Hunyuan)

Drop-in: Replace your original large file with this, or merge the functions/classes into your project.
"""

import argparse
import logging
import math
from typing import Optional, Tuple, List

import cv2
import numpy as np

# Torch is optional for audio_features; code supports numpy arrays too.
try:
    import torch
except Exception:
    torch = None

# Try to import MediaPipe for landmarks; if not present, we use rectangle fallback.
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

logger = logging.getLogger("lip_sync_pipeline")
logging.basicConfig(level=logging.INFO)


# -----------------------------
# Utilities: audio handling
# -----------------------------
def rms_of_array(a: np.ndarray) -> float:
    """Compute RMS (root mean square) loudness for an audio buffer."""
    if a is None or a.size == 0:
        return 0.0
    a = a.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(a)) + 1e-12))


# -----------------------------
# Landmarks helper (mediapipe)
# -----------------------------
class MediapipeFaceMesh:
    """
    Lightweight class to provide lip landmarks from an image using MediaPipe FaceMesh.
    If MediaPipe is not installed or detection fails, return None.
    """

    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=True):
        if not MP_AVAILABLE:
            raise RuntimeError("MediaPipe not available. Install with `pip install mediapipe` for landmarks.")
        self.mp_face_mesh = mp.solutions.face_mesh
        self._proc = self.mp_face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                                max_num_faces=max_num_faces,
                                                refine_landmarks=refine_landmarks,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5)

    def get_lip_landmarks(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns (N,2) numpy array of lip landmarks in pixel coords (x,y) or None.
        We choose commonly used indices for inner/outer lips. If multiple faces found,
        returns first.
        """
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._proc.process(img_rgb)
        if not results.multi_face_landmarks:
            return None

        # FaceMesh has 468 landmarks; indices for lips (outer + inner) can be used.
        # We'll take a subset: upper/lower outer lips and inner lips.
        # Indices chosen (common): outer: [61,146,91,181,84,17,314,405,321,375,291,308]
        # inner: [78,95,88,178,87,14,317,402,318,324,308]
        indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
                   78, 95, 88, 178, 87, 14, 317, 402, 318, 324]  # mix outer & inner
        lm = results.multi_face_landmarks[0]
        h, w = image_bgr.shape[:2]
        coords = []
        for idx in indices:
            p = lm.landmark[idx]
            coords.append([int(p.x * w), int(p.y * h)])
        return np.array(coords, dtype=np.int32)


# -----------------------------
# Mouth Animator (core)
# -----------------------------
class MouthAnimator:
    """
    Mouth animator that can be driven by small audio windows (numpy or torch).
    Features:
     - Adaptive normalization via running max with decay
     - Temporal smoothing (EMA)
     - Landmarks-based deformation (if provided)
     - Rectangle fallback with vertical stretch + blending
    """

    def __init__(self,
                 use_landmarks: bool = True,
                 mediapipe_instance: Optional[MediapipeFaceMesh] = None,
                 device: str = "cpu"):
        self.use_landmarks = use_landmarks and MP_AVAILABLE and mediapipe_instance is not None
        self.mp = mediapipe_instance if mediapipe_instance is not None else None

        # Running max normalization (keeps mouth response stable across recordings)
        self._audio_max_running = 1e-6
        self._audio_decay = 0.995

        # Temporal smoothing
        self.mouth_ema = 0.0
        self.mouth_smooth_alpha = 0.55  # 0..1, higher = more responsive
        self.min_speech_thresh = 0.01   # below this threshold, treat as silence
        self.noise_floor = 1e-4

        # Visual tuning
        self.ease_power = 0.8
        self.blend_min = 0.55
        self.blend_max = 0.95

        # runtime
        self.debug = False
        self.device = device

    # ------ Audio ingestion helper -----
    def ingest_audio(self, audio_buf: np.ndarray) -> float:
        """
        Accepts a chunk of audio samples (np.ndarray, mono). Returns a normalized smoothed float 0..1.
        This is a helper you can call from a Bark stream callback (small windows ~ 200-500ms).
        """
        rms = rms_of_array(audio_buf)
        # update running max
        self._audio_max_running = max(self._audio_max_running * self._audio_decay, rms, 1e-8)
        normalized = 0.0
        if self._audio_max_running > 0:
            normalized = float(np.clip((rms - self.noise_floor) / (self._audio_max_running + 1e-8), 0.0, 1.0))

        # EMA smoothing
        a_alpha = float(self.mouth_smooth_alpha)
        self.mouth_ema = a_alpha * normalized + (1.0 - a_alpha) * self.mouth_ema
        # faster close when silence
        if normalized < self.min_speech_thresh:
            self.mouth_ema *= 0.92

        smoothed = float(np.clip(self.mouth_ema, 0.0, 1.0))
        if self.debug:
            logger.info(f"[ingest_audio] rms={rms:.6f} norm={normalized:.4f} smoothed={smoothed:.4f}")
        return smoothed

    # ------ Main apply function (drop-in replacement for your old _apply_mouth_animation) -----
    def apply_to_frame(self,
                       image: np.ndarray,
                       face_bbox: Tuple[int, int, int, int],
                       audio_features: Optional[np.ndarray] = None,
                       frame_idx: Optional[int] = None,
                       landmarks_override: Optional[np.ndarray] = None
                       ) -> np.ndarray:
        """
        image: BGR uint8 frame
        face_bbox: (x1,y1,x2,y2)
        audio_features: optional short audio chunk (numpy or torch). If provided, it will be ingested.
                        If None, it will use existing EMA state (useful if audio processed elsewhere).
        landmarks_override: optional (N,2) pixel coords for lips (if you compute landmarks elsewhere).
        """
        if audio_features is not None:
            # support torch tensors as well
            if torch is not None and isinstance(audio_features, torch.Tensor):
                arr = audio_features.detach().cpu().numpy()
            else:
                arr = np.asarray(audio_features)
            smoothed = self.ingest_audio(arr)
        else:
            smoothed = float(np.clip(self.mouth_ema, 0.0, 1.0))

        # quick skip if no face bbox or invalid
        H, W = image.shape[:2]
        x1, y1, x2, y2 = face_bbox
        x1 = int(np.clip(x1, 0, W - 1)); x2 = int(np.clip(x2, 1, W))
        y1 = int(np.clip(y1, 0, H - 1)); y2 = int(np.clip(y2, 1, H))
        if x2 <= x1 or y2 <= y1:
            return image

        # Try landmarks path first if enabled
        if self.use_landmarks or (landmarks_override is not None):
            lm = landmarks_override
            if lm is None and self.mp is not None:
                try:
                    lm = self.mp.get_lip_landmarks(image)
                except Exception:
                    lm = None
            if lm is not None:
                try:
                    return self._apply_landmark_deformation(image, lm, smoothed, face_bbox, frame_idx)
                except Exception:
                    logger.exception("Landmark deformation failed; falling back to rectangle deformation.")

        # Fallback rectangle path
        try:
            return self._apply_rectangle_deformation(image, face_bbox, smoothed, frame_idx)
        except Exception:
            logger.exception("Rectangle deformation failed; returning original frame.")
            return image

    # ------ Landmark-based mouth deformation ------
    def _apply_landmark_deformation(self,
                                    image: np.ndarray,
                                    lip_landmarks: np.ndarray,
                                    smoothed: float,
                                    face_bbox: Tuple[int, int, int, int],
                                    frame_idx: Optional[int] = None) -> np.ndarray:
        """
        Given a set of lip landmarks (N,2), apply a simple vertical displacement
        to lower and upper lip regions proportionally to `smoothed`. This is a
        lightweight approach that doesn't do full mesh warping (TPS), but already
        gives much better alignment than a rectangle.

        For best results replace this with a thin-plate spline warp using the
        mapped lip region points (left as an optional future improvement).
        """
        img = image.copy()
        H, W = img.shape[:2]
        # lip_landmarks expected as Nx2: we'll compute upper and lower lip boxes
        xs = lip_landmarks[:, 0]
        ys = lip_landmarks[:, 1]
        lx1 = int(np.clip(np.min(xs) - 2, 0, W - 1))
        lx2 = int(np.clip(np.max(xs) + 2, 1, W))
        ly1 = int(np.clip(np.min(ys) - 2, 0, H - 1))
        ly2 = int(np.clip(np.max(ys) + 2, 1, H))

        # define ROI and split into upper/lower halves
        roi = img[ly1:ly2, lx1:lx2].copy()
        if roi.size == 0:
            return img

        rh, rw = roi.shape[:2]
        half = rh // 2
        upper = roi[:half, :, :].copy()
        lower = roi[half:, :, :].copy()

        # move lower lip down and upper lip slightly up proportionally to smoothed
        # Convert smoothed -> pixel displacement using face height heuristic
        face_h = face_bbox[3] - face_bbox[1]
        max_disp = max(1, int(face_h * 0.14))  # maximum pixel displacement
        disp = int(max_disp * (smoothed ** self.ease_power))

        # Resize (vertical stretch) lower and upper parts to simulate opening
        # Lower lip gets stretched down
        lower_h = lower.shape[0]
        new_lower_h = min(lower_h + disp, lower_h * 3)
        lower_resized = cv2.resize(lower, (lower.shape[1], int(new_lower_h)), interpolation=cv2.INTER_LINEAR)

        # Upper lip gets slightly compressed (or moved up visually)
        upper_h = upper.shape[0]
        new_upper_h = max(1, upper_h - (disp // 2))
        upper_resized = cv2.resize(upper, (upper.shape[1], int(new_upper_h)), interpolation=cv2.INTER_LINEAR)

        # Reconstruct ROI: pad/crop to original roi height
        top_pad = max(0, (upper.shape[0] - upper_resized.shape[0]) // 2)
        bottom_pad = max(0, upper.shape[0] - upper_resized.shape[0] - top_pad)
        upper_padded = cv2.copyMakeBorder(upper_resized, top_pad, bottom_pad, 0, 0, borderType=cv2.BORDER_REPLICATE)

        lower_h_target = lower.shape[0]
        if lower_resized.shape[0] >= lower_h_target:
            y_off = (lower_resized.shape[0] - lower_h_target) // 2
            lower_cropped = lower_resized[y_off:y_off + lower_h_target, :, :]
        else:
            # pad
            top = (lower_h_target - lower_resized.shape[0]) // 2
            bottom = lower_h_target - lower_resized.shape[0] - top
            lower_cropped = cv2.copyMakeBorder(lower_resized, top, bottom, 0, 0, borderType=cv2.BORDER_REPLICATE)

        new_roi = np.vstack([upper_padded, lower_cropped])

        # Blend
        factor = float(smoothed ** self.ease_power)
        alpha = float(np.clip(self.blend_min + (self.blend_max - self.blend_min) * factor, 0.0, 0.98))
        new_roi_u8 = new_roi.astype(np.uint8)
        roi_u8 = roi.astype(np.uint8)
        blended = cv2.addWeighted(new_roi_u8, alpha, roi_u8, 1.0 - alpha, 0)

        img[ly1:ly2, lx1:lx2] = blended
        if self.debug and frame_idx is not None and frame_idx % 25 == 0:
            logger.info(f"[landmark] applied lip deformation disp={disp} alpha={alpha:.3f}")
        return img

    # ------ Rectangle fallback deformation (texture preserving) ------
    def _apply_rectangle_deformation(self,
                                     image: np.ndarray,
                                     face_bbox: Tuple[int, int, int, int],
                                     smoothed: float,
                                     frame_idx: Optional[int] = None) -> np.ndarray:
        img = image.copy()
        H, W = img.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = x2 - x1
        face_h = y2 - y1

        # mouth center heuristics
        mouth_cx = x1 + face_w // 2
        mouth_cy = y1 + int(face_h * 0.78)

        base_w = max(4, int(face_w * 0.45))
        base_h = max(4, int(face_h * 0.07))
        max_open_h = max(base_h + 1, int(face_h * 0.22))

        mouth_x1 = max(0, mouth_cx - base_w // 2)
        mouth_x2 = min(W, mouth_cx + base_w // 2)
        mouth_y1 = max(0, mouth_cy - base_h // 2)
        mouth_y2 = min(H, mouth_cy + base_h // 2)

        if mouth_x2 <= mouth_x1 or mouth_y2 <= mouth_y1:
            return img

        mouth_region = img[mouth_y1:mouth_y2, mouth_x1:mouth_x2].copy()
        if mouth_region.size == 0:
            return img

        # target height based on smoothed audio (eased)
        factor = float(smoothed ** self.ease_power)
        target_h = int(base_h + (max_open_h - base_h) * factor)
        target_h = max(1, target_h)

        orig_h, orig_w = mouth_region.shape[:2]
        mouth_resized = cv2.resize(mouth_region, (orig_w, target_h), interpolation=cv2.INTER_LINEAR)

        # center-crop or pad back to orig_h
        if target_h >= orig_h:
            y_off = (target_h - orig_h) // 2
            output_mouth = mouth_resized[y_off:y_off + orig_h, :, :]
        else:
            top = (orig_h - target_h) // 2
            bottom = orig_h - target_h - top
            output_mouth = cv2.copyMakeBorder(mouth_resized, top, bottom, 0, 0, borderType=cv2.BORDER_REPLICATE)

        # blending alpha scaled by factor
        alpha = float(np.clip(self.blend_min + (self.blend_max - self.blend_min) * factor, 0.0, 0.98))

        blended = cv2.addWeighted(output_mouth.astype(np.uint8), alpha, mouth_region.astype(np.uint8), 1.0 - alpha, 0)
        img[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = blended

        if self.debug and frame_idx is not None and frame_idx % 25 == 0:
            logger.info(f"[rect] applied rect deformation target_h={target_h} alpha={alpha:.3f}")

        return img


# -----------------------------
# Integration notes & helpers
# -----------------------------
def connect_bark_stream_to_animator(animator: MouthAnimator, bark_audio_callback):
    """
    Example placeholder for connecting Bark TTS streaming output into the animator.
    The bark_audio_callback should call animator.ingest_audio(np_audio_chunk) for each small chunk.

    Real-time tip:
    - Use small windows (e.g., 16000 Hz * 0.08s => 1280 samples) so mouth reacts with low latency.
    - Adjust animator.mouth_smooth_alpha for smoother vs more responsive motion.
    """
    raise NotImplementedError("This helper is illustrative. Integrate with your Bark streaming code by calling animator.ingest_audio(chunk).")


# -----------------------------
# Optional model suggestions & where to plug them
# -----------------------------
#
# If you want even higher-quality lip-sync (photoreal), consider:
#  - Wav2Lip: deterministic and high-quality mouth frames. Accepts audio + face crop -> returns synced face.
#    Integration: call Wav2Lip per short window (e.g. 0.5s), blend resultant mouth onto original frame.
#    Downsides: adds latency and GPU compute (not fully real-time unless optimized).
#
#  - HunyuanVideo-Avatar / MuseTalk: model-based full-face animation, often higher realism but heavy.
#    Integration: use them for offline/high-quality generation rather than streaming.
#
# For real-time Bark-driven avatar, the landmark method + texture-preserving deformation usually achieves
# a good balance of latency and visual quality.
#

# -----------------------------
# Smoke test / demo harness
# -----------------------------
def demo_smoke_test(output_prefix: str = "demo_out"):
    """Creates two images (quiet and loud) to verify mouth deformation works end-to-end."""
    logger.info("Running smoke test...")

    # create sample skin-like base image
    H, W = 480, 640
    base = np.full((H, W, 3), 200, dtype=np.uint8)
    # draw a simple face rectangle for visualization
    face_bbox = (160, 60, 480, 420)
    cv2.rectangle(base, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (170, 140, 120), thickness=2)

    # If mediapipe available, create instance
    mp_inst = None
    if MP_AVAILABLE:
        try:
            mp_inst = MediapipeFaceMesh(static_image_mode=True)
        except Exception as e:
            logger.warning("Mediapipe FaceMesh init failed; landmarks disabled.")

    animator = MouthAnimator(use_landmarks=(mp_inst is not None), mediapipe_instance=mp_inst)
    animator.debug = True

    # Generate fake audio: loud and quiet
    loud = (np.random.randn(1600) * 0.1 + 0.2).astype(np.float32)
    quiet = (np.random.randn(1600) * 0.01).astype(np.float32)

    # Apply loud
    out_loud = animator.apply_to_frame(base.copy(), face_bbox, audio_features=loud, frame_idx=0)
    cv2.imwrite(f"{output_prefix}_loud.png", out_loud)
    logger.info(f"Wrote {output_prefix}_loud.png")

    # Apply quiet
    out_quiet = animator.apply_to_frame(base.copy(), face_bbox, audio_features=quiet, frame_idx=1)
    cv2.imwrite(f"{output_prefix}_quiet.png", out_quiet)
    logger.info(f"Wrote {output_prefix}_quiet.png")

    logger.info("Smoke test complete.")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lip sync pipeline smoke-test / runner")
    parser.add_argument("--test", action="store_true", help="Run smoke test and write sample outputs")
    parser.add_argument("--no-landmarks", action="store_true", help="Disable mediapipe landmarks even if installed")
    args = parser.parse_args()

    if args.test:
        demo_smoke_test()
