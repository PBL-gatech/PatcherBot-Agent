"""Optional local learned correspondences for difficult saved scan pairs.

Only returns image points. Acceptance and coordinate transforms belong to the
registration validator; this module never downloads weights or operates devices.
"""
from functools import lru_cache
from threading import Lock

import numpy as np

_MODEL_LOCK = Lock()
_MODEL_ID = "ETH-CVG/lightglue_superpoint"


@lru_cache(maxsize=1)
def _local_model():
    import torch
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching
    processor = AutoImageProcessor.from_pretrained(_MODEL_ID, local_files_only=True)
    model = AutoModelForKeypointMatching.from_pretrained(_MODEL_ID, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return processor, model.eval().to(device), device


def learned_correspondences(scan, reference):
    """Return matched points in supplied overview pixels plus confidence scores.

    Missing optional packages or locally cached weights raise their normal errors
    for the caller to report. Low confidence correspondences are excluded, while
    all geometric acceptance checks remain the caller's responsibility.
    """
    import torch
    from PIL import Image
    source = np.asarray(scan["pixels"] if isinstance(scan, dict) else scan, dtype=np.uint8)
    target = np.asarray(reference["pixels"] if isinstance(reference, dict) else reference, dtype=np.uint8)
    with _MODEL_LOCK:
        processor, model, device = _local_model()
        batch = processor(images=[Image.fromarray(source), Image.fromarray(target)],
                          return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**batch)
        result = processor.post_process_keypoint_matching(
            output, [[source.shape[:2], target.shape[:2]]], threshold=0.1)[0]
        return (result["keypoints0"].cpu().numpy().astype(np.float32),
                result["keypoints1"].cpu().numpy().astype(np.float32),
                result["matching_scores"].cpu().numpy().astype(np.float32))
