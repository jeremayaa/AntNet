#!/usr/bin/env python3
import os
import sys
import csv
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

# ─── Project setup ─────────────────────────────────────────────────────────────
#!/usr/bin/env python3
import os, sys
from pathlib import Path

# ─── Fix project paths ─────────────────────────────────────────────────────────
# If script.py is in `project/notebooks/`, then:
SCRIPT_DIR   = Path(__file__).resolve().parent        # .../AntNet/notebooks
PROJECT_ROOT = SCRIPT_DIR.parent                      # .../AntNet
SRC_PATH     = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_PATH))

# Now imports from `src/` will work:
from utils      import load_image, load_mask, compute_vector_field
from engine     import AntEnv
from model      import AntModel

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
N_AGENTS       = 10
LOG_INTERVAL   = 10
IMAGE_INTERVAL = 100
MAX_ENERGY     = 5000
INITIAL_ENERGY = 50
EVAL_STEPS     = 50   # rollout length when making snapshots

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Helpers ───────────────────────────────────────────────────────────────────

def pack_inputs(patch_np, memory, device):
    """
    Turn a single (H×W×3) patch + list of (patch,action) into
    torch tensors for the model.
    """
    # normalize & rearrange: 1×3×H×W
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


class Agent:
    """Wraps an AntEnv + AntModel + optimizer + energy counter."""
    def __init__(self, img, mask, vf, device):
        self.env    = AntEnv(img, mask, vf,
                             kernel_size=(16,16),
                             memory_len=100,
                             max_steps=1000,
                             boundary='torus')
        self.model  = AntModel(in_channels=3,
                               patch_size=(16,16),
                               emb_dim=128,
                               n_actions=8,
                               n_heads=4).to(device)
        self.opt    = optim.Adam(self.model.parameters(), lr=1e-4)
        self.energy = INITIAL_ENERGY
        # fresh reset
        self.obs_patch, self.mem = self.env.reset()

    def step_and_learn(self, device):
        # forward & sample
        pt, mem_t = pack_inputs(self.obs_patch, self.mem, device)
        logits    = self.model(pt, mem_t)
        dist      = torch.distributions.Categorical(logits=logits)
        action    = dist.sample()
        logp      = dist.log_prob(action)
        entropy = dist.entropy() 
        

        # env step → get cosine reward
        (obs2, mem2), cosine_r, done, _ = self.env.step(int(action.item()))
        in_cell = self.env.is_in_cell()
        # occupancy bonus
        occ_r = 1 if in_cell else -1

        # evolutionary energy update
        self.energy += occ_r

        # combined learning signal
        r_total = cosine_r + occ_r

        entropy_coef = 0.2 
        loss = - (logp * r_total + entropy_coef * entropy)

        # one-step PG update
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # advance obs
        self.obs_patch, self.mem = obs2, mem2


def snapshot_trajectories(agents, img, mask, vf, device, step, out_dir):
    """
    For each agent, roll out EVAL_STEPS steps greedily and scatter-
    plot its trajectory (brighter=later), all on one figure.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img, alpha=0.3)
    ax.axis('off')
    ax.set_title(f'Agent Positions at step {step}')

    for ag in agents:
        # fresh eval env
        eval_env = AntEnv(img, mask, vf,
                          kernel_size=(16,16),
                          memory_len=100,
                          max_steps=EVAL_STEPS,
                          boundary='torus')
        obs_patch, mem = eval_env.reset()

        # greedy rollout
        for _ in range(EVAL_STEPS):
            pt, mem_t = pack_inputs(obs_patch, mem, device)
            with torch.no_grad():
                logits = ag.model(pt, mem_t)
            action = int(logits.argmax(dim=-1).item())
            (obs_patch, mem), _, done, _ = eval_env.step(action)
            if done:
                break

        traj = np.array(eval_env.trajectory)
        T    = len(traj)
        for t, (x, y) in enumerate(traj):
            alpha = 0.1 + 0.9 * (t / (T - 1))
            ax.scatter(x, y, s=8, color='blue', alpha=alpha)

    out_path = out_dir / f'traj_step_{step:05d}.png'
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    # prepare data
    img_path  = PROJECT_ROOT / 'data/Fold1_jpg/images/1.jpg'
    mask_path = PROJECT_ROOT / 'data/Fold1_jpg/masks/1.jpg'
    img  = load_image(str(img_path))
    mask = load_mask(str(mask_path))
    vf   = compute_vector_field(mask)

    # make output dirs
    (PROJECT_ROOT / 'Agents').mkdir(exist_ok=True)
    img_hist_dir = PROJECT_ROOT / 'images_history'
    img_hist_dir.mkdir(exist_ok=True)

    # spawn population
    agents = [Agent(img, mask, vf, DEVICE) for _ in range(N_AGENTS)]

    # CSV logger
    csv_path = PROJECT_ROOT / 'Energies_log.csv'
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    # header: step, agent_0, agent_1, ..., agent_9
    header = ['step'] + [f'agent_{i}' for i in range(N_AGENTS)]
    writer.writerow(header)
    csv_file.flush()

    global_step = 0
    best_energy = -float('inf')

    try:
        while True:
            global_step += 1

            # step & learn
            for i, ag in enumerate(agents):
                ag.step_and_learn(DEVICE)

                # track best overall
                if ag.energy > best_energy:
                    best_energy = ag.energy

                # respawn dead ants
                if ag.energy <= 0:
                    agents[i] = Agent(img, mask, vf, DEVICE)

                # stop if any mastered
                if ag.energy >= MAX_ENERGY:
                    print(f"🏆 Agent {i} reached {ag.energy} energy at step {global_step}!")
                    raise StopIteration

            # logging
            if global_step % LOG_INTERVAL == 0:
                energies = [int(ag.energy) for ag in agents]
                print(f"[Step {global_step:05d}] Energies: {energies}")
                writer.writerow([global_step] + energies)
                csv_file.flush()

            # snapshot images
            if global_step % IMAGE_INTERVAL == 0:
                snapshot_trajectories(agents, img, mask, vf, DEVICE,
                                      step=global_step,
                                      out_dir=img_hist_dir)

    except (KeyboardInterrupt, StopIteration):
        print(f"⏹ Stopping at step {global_step} (best energy={best_energy})")

    finally:
        # save all final agent models
        for idx, ag in enumerate(agents):
            path = PROJECT_ROOT / 'Agents' / f'agent_{idx}_step{global_step}.pth'
            torch.save(ag.model.state_dict(), path)
        print(f"💾 Saved all {len(agents)} agents to './Agents/'")
        csv_file.close()


if __name__ == '__main__':
    main()
