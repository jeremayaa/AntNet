import torch
import torch.nn as nn


class AntModel(nn.Module):
    """
    CNN encoder + attention over past (patch,action) pairs → logits over 8 moves.
    """
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: tuple[int,int] = (16,16),
        emb_dim: int = 128,
        n_actions: int = 8,
        n_heads: int = 4
    ):
        super().__init__()
        H, W = patch_size

        # patch encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * H * W, emb_dim),
            nn.ReLU(),
        )

        # action vector → embedding
        self.action_emb = nn.Linear(2, emb_dim)

        # multi-head attention: query=current patch, kv=memory
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=n_heads,
            batch_first=True
        )

        # final policy head
        self.policy_head = nn.Linear(emb_dim, n_actions)

    def forward(self, patch: torch.Tensor, memory: list[tuple[torch.Tensor, torch.Tensor]]):
        """
        patch: [B, C, H, W]
        memory: list of (patch_tensor [C,H,W] or [1,C,H,W], action_tensor [2])
        returns: logits [B, n_actions]
        """
        B = patch.shape[0]
        # encode current patch → [B, emb_dim]
        z = self.encoder(patch)                # (B, D)
        z = z.unsqueeze(1)                     # (B, 1, D) as query

        # build memory embeddings
        if len(memory) > 0:
            mem_list = []
            for (mp, a) in memory:
                # ensure batch dim
                if mp.dim() == 3:       # [C,H,W]
                    mp = mp.unsqueeze(0)  # → [1,C,H,W]
                zmp = self.encoder(mp)  # [1, D]
                za  = self.action_emb(a.unsqueeze(0))  # [1, D]
                mem_list.append((zmp + za).unsqueeze(0))  # [1,1,D]
            mem_seq = torch.cat(mem_list, dim=1)  # [1, L, D]
            # attend: query=z, keys=mem_seq, values=mem_seq
            attn_out, _ = self.attn(z, mem_seq, mem_seq)  # [1,1,D]
            out = attn_out.squeeze(1)                     # [1, D]
        else:
            out = z.squeeze(1)  # no memory yet

        logits = self.policy_head(out)  # [B, n_actions]
        return logits