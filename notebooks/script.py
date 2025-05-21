#!/usr/bin/env python3
import os
import sys
import csv
import random
from pathlib import Path

import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

# ─── Project setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent        # .../AntNet/notebooks
PROJECT_ROOT = SCRIPT_DIR.parent                      # .../AntNet
SRC_PATH     = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_PATH))

from utils      import load_image, load_mask, compute_vector_field
from engine     import AntEnv
from model      import AntModel

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
N_AGENTS       = 1
LOG_INTERVAL   = 10
IMAGE_INTERVAL = 100
MAX_ENERGY     = 10000
INITIAL_ENERGY = 5000
EVAL_STEPS     = 50   # rollout length for snapshots

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Helpers ───────────────────────────────────────────────────────────────────

def pack_inputs(patch_np, memory, device):
    pt = (
        torch.from_numpy(patch_np.astype(np.float32)/255)
             .permute(2,0,1)
             .unsqueeze(0)
             .to(device)
    )
    mem_t = []
    for p_np, a_np in memory:
        p_t = (
            torch.from_numpy(p_np.astype(np.float32)/255)
                 .permute(2,0,1)
                 .to(device)
        )
        a_t = torch.from_numpy(a_np.astype(np.float32)).to(device)
        mem_t.append((p_t, a_t))
    return pt, mem_t


def compute_iou(mask: np.ndarray, trajectory: list[tuple[int,int]], kernel_size: tuple[int,int]):
    H, W = mask.shape
    kh, kw = kernel_size
    coverage = np.zeros((H, W), dtype=bool)
    for x, y in trajectory:
        x1 = max(0, x - kw//2)
        x2 = min(W, x1 + kw)
        y1 = max(0, y - kh//2)
        y2 = min(H, y1 + kh)
        coverage[y1:y2, x1:x2] = True
    mask_bool = mask.astype(bool)
    inter = np.logical_and(coverage, mask_bool).sum()
    union = np.logical_or(coverage, mask_bool).sum()
    return inter/union if union>0 else 0.0


class Agent:
    def __init__(self, img, mask, vf, device):
        self.env    = AntEnv(img, mask, vf,
                             kernel_size=(16,16),
                             memory_len=10,
                             max_steps=1000,
                             boundary='torus')
        self.model  = AntModel(
            in_channels=3,
            patch_size=(16,16),
            emb_dim=128,
            n_actions=len(self.env.actions),
            n_heads=4
        ).to(device)
        self.opt    = optim.Adam(self.model.parameters(), lr=1e-4)
        self.energy = INITIAL_ENERGY
        self.obs_patch, self.mem = self.env.reset()
        self.prev_act_unit = None

    def step_and_learn(self, device):
        pt, mem_t = pack_inputs(self.obs_patch, self.mem, device)
        logits    = self.model(pt, mem_t)
        dist      = torch.distributions.Categorical(logits=logits)
        action    = dist.sample()
        logp      = dist.log_prob(action)
        entropy   = dist.entropy()

        (obs2, mem2), cosine_r, done, _ = self.env.step(int(action.item()))

        in_cell = self.env.is_in_cell()
        occ_r   = 1 if in_cell else -1
        self.energy += occ_r

        dx, dy = self.env.actions[int(action.item())]
        act_vec = torch.tensor([dx, dy], dtype=torch.float32, device=device)
        act_unit = act_vec / (act_vec.norm() + 1e-8)
        if self.prev_act_unit is not None:
            turn_bonus = (1.0 - torch.dot(act_unit, self.prev_act_unit)).item()
        else:
            turn_bonus = 0.0
        self.prev_act_unit = act_unit

        r_total = cosine_r + occ_r + 0.1 * turn_bonus
        loss = - (logp * r_total + 0.5 * entropy)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.obs_patch, self.mem = obs2, mem2


def snapshot_and_log(agents, dataset, device, step, out_dir, iou_writer):
    """
    Pick a random slide, evaluate all agents on it, save image with mask & paths,
    compute per-agent IoUs, log them.
    """
    # 1) choose random slide
    img, mask, vf = random.choice(dataset)

    # 2) prepare plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img, alpha=0.6)
    ax.imshow(mask, cmap='Reds', alpha=0.3)  # semi-transparent mask

    # 3) rollout each agent
    ious = []
    for ag in agents:
        eval_env = AntEnv(img, mask, vf,
                          kernel_size=(16,16),
                          memory_len=10,
                          max_steps=EVAL_STEPS,
                          boundary='torus')
        obs, mem = eval_env.reset()
        for _ in range(EVAL_STEPS):
            pt, mem_t = pack_inputs(obs, mem, device)
            with torch.no_grad():
                logits = ag.model(pt, mem_t)
            action = int(logits.argmax(dim=-1).item())
            (obs, mem), _, done, _ = eval_env.step(action)
            if done:
                break

        traj = eval_env.trajectory
        xs, ys = zip(*traj)
        T = len(traj)
        for t, (x, y) in enumerate(traj):
            alpha = 0.1 + 0.9 * (t/(T-1))
            ax.scatter(x, y, s=8, color='blue', alpha=alpha)

        iou = compute_iou(mask, traj, eval_env.kernel_size)
        ious.append(iou)

    # 4) title and save
    title = f"Step {step} | IoUs: " + ", ".join(f"{iou:.2f}" for iou in ious)
    ax.set_title(title)
    os.makedirs(out_dir, exist_ok=True)
    fname = Path(out_dir)/f"snapshot_{step:05d}.png"
    fig.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # 5) log IoUs
    iou_writer.writerow([step] + ious)


def main():
    # load up to 20 slides
    DATA_DIR   = PROJECT_ROOT / 'data' / 'Fold1_jpg'
    imgs_dir   = DATA_DIR / 'images'
    masks_dir  = DATA_DIR / 'masks'
    img_paths  = sorted(imgs_dir.glob('*.jpg'), key=lambda p: int(p.stem))
    mask_paths = sorted(masks_dir.glob('*.jpg'), key=lambda p: int(p.stem))
    pairs = list(zip(img_paths, mask_paths))
    if len(pairs) > 20:
        pairs = random.sample(pairs, 20)

    dataset = []
    for ip, mp in pairs:
        im  = load_image(str(ip))
        mk  = load_mask(str(mp))
        vf  = compute_vector_field(mk)
        dataset.append((im, mk, vf))

    # prepare output dirs & logs
    Agents_DIR = PROJECT_ROOT/'Agents'
    Images_DIR = PROJECT_ROOT/'images_history'
    Agents_DIR.mkdir(exist_ok=True)
    Images_DIR.mkdir(exist_ok=True)

    en_log = PROJECT_ROOT/'Energies_log.csv'
    iou_log = PROJECT_ROOT/'IoU_log.csv'
    with open(en_log, 'w', newline='') as f:
        csv.writer(f).writerow(['step'] + [f'agent_{i}' for i in range(N_AGENTS)])
    with open(iou_log, 'w', newline='') as f:
        csv.writer(f).writerow(['step'] + [f'agent_{i}' for i in range(N_AGENTS)])

    # spawn agents
    agents = []
    for _ in range(N_AGENTS):
        im, mk, vf = random.choice(dataset)
        agents.append(Agent(im, mk, vf, DEVICE))

    global_step = 0
    best_energy = -float('inf')

    try:
        with open(en_log, 'a', newline='') as ef, open(iou_log, 'a', newline='') as iof:
            en_writer  = csv.writer(ef)
            iou_writer = csv.writer(iof)

            while True:
                global_step += 1
                for i, ag in enumerate(agents):
                    ag.step_and_learn(DEVICE)
                    best_energy = max(best_energy, ag.energy)
                    if ag.energy <= 0:
                        im, mk, vf = random.choice(dataset)
                        agents[i] = Agent(im, mk, vf, DEVICE)
                    if ag.energy >= MAX_ENERGY:
                        print(f"🏆 Agent {i} reached {ag.energy} at step {global_step}!")
                        raise StopIteration

                if global_step % LOG_INTERVAL == 0:
                    energies = [int(a.energy) for a in agents]
                    print(f"[Step {global_step:05d}] Energies: {energies}")
                    en_writer.writerow([global_step] + energies)
                    ef.flush()

                if global_step % IMAGE_INTERVAL == 0:
                    print(f"[Step {global_step:05d}] Generating snapshot and IoUs")
                    snapshot_and_log(agents, dataset, DEVICE,
                                     global_step, Images_DIR, iou_writer)
                    iof.flush()

    except (KeyboardInterrupt, StopIteration):
        print(f"⏹ Stopped at step {global_step} (best energy={best_energy})")

    finally:
        for idx, ag in enumerate(agents):
            torch.save(ag.model.state_dict(),
                       Agents_DIR/f'agent_{idx}_step{global_step}.pth')
        print(f"💾 Saved all {len(agents)} agents to {Agents_DIR}")

if __name__ == '__main__':
    main()
