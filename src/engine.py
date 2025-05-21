import numpy as np
import random
from collections import deque
from utils import extract_patch


class AntEnv:
    """
    A simple “ant” that at each step sees a local patch and its own memory,
    takes one of 8 discrete moves, and receives a cosine-based reward
    relative to a precomputed vector field.
    """
    def __init__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        vector_field: np.ndarray,
        kernel_size: tuple[int,int] = (16,16),
        memory_len: int = 100,
        boundary: str = 'torus',
        max_steps: int = 50000
    ):
        self.image = image
        self.mask = mask
        self.vf = vector_field
        self.kernel_size = kernel_size
        self.boundary   = boundary            # ← store it
        self.memory_len = memory_len
        self.max_steps = max_steps

        # 8 directions: N, NE, E, SE, S, SW, W, NW
        self.actions = [
            (0, -5), (5, -5), (5, 0), (5, 5),
            (0, 5),  (-5, 5), (-5, 0), (-5, -5)
        ]

        self.reset()

    def reset(self):
        H, W = self.mask.shape
        # random start
        self.pos = (random.randint(0, W-1), random.randint(0, H-1))
        self.step_count = 0
        self.memory = deque(maxlen=self.memory_len)
        self.trajectory = [self.pos]
        patch = extract_patch(self.image, self.pos, self.kernel_size)
        return patch, list(self.memory)

    def step(self, action_idx: int):
        """
        Apply action, compute reward = cosine(action_vec, vf_at_new_pos),
        append (prev_patch, action_vec) to memory, return (obs, mem), reward, done, info.
        """
        # get previous local patch
        prev_patch = extract_patch(self.image, self.pos, self.kernel_size)

        dx, dy = self.actions[action_idx]
        x, y = self.pos
        H, W = self.mask.shape

        # either torus‐wrap or clip
        if self.boundary == 'torus':
            new_x = (x + dx) % W
            new_y = (y + dy) % H
        else:
            new_x = np.clip(x + dx, 0, W - 1)
            new_y = np.clip(y + dy, 0, H - 1)

        self.pos = (new_x, new_y)
        self.trajectory.append(self.pos)
        self.step_count += 1

        # reward: cosine similarity
        vf_vec = self.vf[new_y, new_x]
        action_vec = np.array([dx, dy], dtype=np.float32)
        norm = np.linalg.norm(action_vec)
        if norm > 0:
            action_unit = action_vec / norm
            reward = float(np.dot(action_unit, vf_vec))
        else:
            reward = 0.0

        # update memory
        self.memory.append((prev_patch, action_unit))

        # new observation
        obs_patch = extract_patch(self.image, self.pos, self.kernel_size)
        obs = (obs_patch, list(self.memory))

        done = (self.step_count >= self.max_steps)
        info = {'position': self.pos, 'steps': self.step_count}

        return obs, reward, done, info
    
    def is_in_cell(self) -> bool:
        """True iff the ant’s current center position lies on a mask pixel==1."""
        x, y = self.pos
        return bool(self.mask[y, x])

    def render(self, overlay_mask: bool = True, figsize=(5,5)):
        """
        Return an RGB array with trajectory and optional mask overlay.
        """
        import cv2
        import numpy as np

        canvas = self.image.copy()
        if overlay_mask:
            m_rgb = np.zeros_like(canvas)
            m_rgb[..., 0] = self.mask * 255
            canvas = ((canvas * 0.5) + (m_rgb * 0.5)).astype(np.uint8)

        # draw trajectory
        for (x0, y0), (x1, y1) in zip(self.trajectory[:-1], self.trajectory[1:]):
            cv2.line(canvas, (x0, y0), (x1, y1), (0,255,0), 1)

        return canvas