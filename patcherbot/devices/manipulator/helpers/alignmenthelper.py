import numpy as np

from patcherbot.deepLearning.cellSegmentor import CellSegmentor2
from patcherbot.deepLearning.PatchMatcher import PatchMatcher


class AlignmentHelper:
    def __init__(self, segmentor=None, matcher=None):
        self.segmentor = CellSegmentor2() if segmentor is None else segmentor
        self.matcher = PatchMatcher() if matcher is None else matcher
        self.observation = None
        self.cell = None

    def update(self, observation, cell):
        self.observation = observation
        self.cell = cell

    def segment(self, image, point):
        """Segment one prompted cell and return its mask and interior targets."""
        mask = self.segmentor.segment(
            image=self.segmentor._prepare_for_segmentation(image),
            input_point=np.asarray([point], dtype=np.float32),
            input_label=np.asarray([1], dtype=np.int32),
            multimask_output=False,
        )
        return self.segmentor.get_mask_target(mask)

    def match(self, reference_image, image):
        """Return reference-to-current translation; inputs are paths or CHW tensors."""
        return self.matcher.find_target(reference_image, image)["translation"]

    @staticmethod
    def get_error(target_xy, desired_xy):
        """Return the XY correction from the target to the desired image position."""
        return np.asarray(desired_xy, dtype=float) - np.asarray(target_xy, dtype=float)

    def get_alignment(self):
        if self.observation is None or self.cell is None:
            raise ValueError("Observation and cell must be set before obtaining alignment.")
        # Implement the 
        return None

    