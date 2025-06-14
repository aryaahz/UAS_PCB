import cv2
import numpy as np
from tqdm import tqdm  # Tambahkan tqdm

class Inpainter:
    CHECK_VALID = 1
    CHECK_INVALID = 0

    def __init__(self, image, mask, halfPatchWidth=4):
        self.originalImage = image.astype(np.float32)
        self.inpaintMask = (mask > 0).astype(np.uint8)  # binary mask: 1 for inpaint region
        self.halfPatchWidth = halfPatchWidth
        self.confidence = np.where(self.inpaintMask == 0, 1.0, 0.0)
        self.result = self.originalImage.copy()
        self.height, self.width = mask.shape

    def checkValidInputs(self):
        if self.originalImage is None or self.inpaintMask is None:
            return self.CHECK_INVALID
        if self.originalImage.shape[:2] != self.inpaintMask.shape:
            return self.CHECK_INVALID
        return self.CHECK_VALID

    def inpaint(self):
        total_pixels_to_fill = np.sum(self.inpaintMask)
        pbar = tqdm(total=total_pixels_to_fill, desc="Inpainting Pixels", unit="px")

        while np.any(self.inpaintMask):
            # 1. Compute the fill front
            front = cv2.Laplacian(self.inpaintMask, cv2.CV_64F)
            frontPoints = np.argwhere(front > 0)

            if len(frontPoints) == 0:
                break

            priorities = []
            patches = []

            for y, x in frontPoints:
                if not self._isPatchValid(x, y):
                    continue
                patch = self._getPatch(x, y)
                confidence = self._computeConfidence(patch)
                data = 1.0  # Simplified data term
                priority = confidence * data
                priorities.append(priority)
                patches.append((x, y, patch))

            if not priorities:
                break

            best = np.argmax(priorities)
            targetX, targetY, targetPatch = patches[best]

            bestMatchX, bestMatchY = self._findBestMatch(targetPatch)

            # Hitung jumlah piksel yang akan diisi
            yslice, xslice = targetPatch
            to_fill = self.inpaintMask[yslice, xslice] == 1
            num_filled = np.sum(to_fill)

            self._copyPatch(targetX, targetY, bestMatchX, bestMatchY)

            # Update progress bar
            pbar.update(num_filled)

        pbar.close()

    def _isPatchValid(self, x, y):
        return (self.halfPatchWidth <= x < self.width - self.halfPatchWidth) and \
               (self.halfPatchWidth <= y < self.height - self.halfPatchWidth)

    def _getPatch(self, x, y):
        x0 = x - self.halfPatchWidth
        y0 = y - self.halfPatchWidth
        x1 = x + self.halfPatchWidth + 1
        y1 = y + self.halfPatchWidth + 1
        return (slice(y0, y1), slice(x0, x1))

    def _computeConfidence(self, patch):
        yslice, xslice = patch
        patchMask = self.inpaintMask[yslice, xslice]
        patchConf = self.confidence[yslice, xslice]
        return np.sum(patchConf * (1 - patchMask)) / ((2*self.halfPatchWidth + 1)**2)

    def _findBestMatch(self, targetPatch):
        yslice, xslice = targetPatch
        targetRegion = self.result[yslice, xslice]
        targetMask = self.inpaintMask[yslice, xslice]

        minError = float('inf')
        bestMatch = (0, 0)

        for y in range(self.halfPatchWidth, self.height - self.halfPatchWidth):
            for x in range(self.halfPatchWidth, self.width - self.halfPatchWidth):
                patch = self._getPatch(x, y)
                ysl, xsl = patch

                if np.any(self.inpaintMask[ysl, xsl]):
                    continue

                candidate = self.result[ysl, xsl]
                error = np.sum(((targetRegion - candidate) * (1 - targetMask)[..., None])**2)

                if error < minError:
                    minError = error
                    bestMatch = (x, y)

        return bestMatch

    def _copyPatch(self, tx, ty, sx, sy):
        t_patch = self._getPatch(tx, ty)
        s_patch = self._getPatch(sx, sy)

        t_yslice, t_xslice = t_patch
        s_yslice, s_xslice = s_patch

        mask = self.inpaintMask[t_yslice, t_xslice]
        to_fill = (mask == 1)

        self.result[t_yslice, t_xslice][to_fill] = self.result[s_yslice, s_xslice][to_fill]
        self.confidence[t_yslice, t_xslice][to_fill] = self._computeConfidence(t_patch)
        self.inpaintMask[t_yslice, t_xslice][to_fill] = 0
