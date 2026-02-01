from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


class BackgroundSampler:
    def __init__(self, background_dir: str = 'assets/backgrounds'):
        self.background_dir = str(background_dir)
        self.paths: List[Path] = []
        self._scan()

    def _scan(self):
        p = Path(self.background_dir)
        if not p.exists():
            self.paths = []
            return
        self.paths = sorted([f for f in p.rglob('*') if f.is_file() and f.suffix.lower() in IMG_EXTS])

    def available(self) -> bool:
        return len(self.paths) > 0

    def _read_square(self, path: Path, size: int) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return np.full((size, size, 3), 235, np.uint8)

        h, w = img.shape[:2]
        if min(h, w) < size:
            scale = size / float(min(h, w))
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        y0 = max(0, (h - size) // 2)
        x0 = max(0, (w - size) // 2)
        img = img[y0:y0 + size, x0:x0 + size].copy()
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def sample(self, batch_size: int, size: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.available():
            return None
        idx = np.random.randint(0, len(self.paths), size=batch_size)
        imgs = [self._read_square(self.paths[i], size=size) for i in idx]
        arr = np.stack(imgs, axis=0)
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float() / 255.0
        return x.to(device)
