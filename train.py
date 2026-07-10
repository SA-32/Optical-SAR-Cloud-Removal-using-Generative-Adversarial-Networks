"""
Example training script for DADIGAN, following the hyperparameters
reported in Section 4.2 of the paper:

    - Adam optimizer, lr = 4e-5 for both generator and discriminator
    - ~200 epochs, batch size 8
    - lambda1 = lambda2 = 100 in the combined loss
    - T = 3 unfolding stages
    - Optical: 13 bands (SEN12MS-CR), SAR: 2 polarizations (VV, VH)
    - Image size 256 x 256

This script provides a runnable skeleton: a `CloudRemovalDataset` stub
you should adapt to your actual data layout (e.g. SEN12MS-CR triplets of
cloudy Sentinel-2 / Sentinel-1 SAR / cloud-free Sentinel-2), plus the
training loop wiring the Generator, Discriminator and DADIGANLoss
together.
"""

import os
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .generator import Generator
    from .discriminator import Discriminator
    from .losses import DADIGANLoss
except ImportError:  # allow running as a standalone script: `python train.py`
    from generator import Generator
    from discriminator import Discriminator
    from losses import DADIGANLoss


class CloudRemovalDataset(Dataset):
    """Stub dataset. Replace `__len__` / `__getitem__` with real data loading.

    Expected outputs per sample:
        cloudy   : FloatTensor (C1, H, W) in [0, 1]   -- cloud-covered optical
        sar      : FloatTensor (C2, H, W)             -- SAR image (VV, VH)
        cloudfree: FloatTensor (C1, H, W) in [0, 1]   -- target cloud-free optical

    For SEN12MS-CR (Section 4.1): crop VV to [-25, 0], VH to [0, 32], then
    normalize; crop optical to [0, 10000] and normalize to [0, 1].
    """

    def __init__(self, root: str, split: str = "train", patch_size: int = 256,
                 c1: int = 13, c2: int = 2):
        self.root = root
        self.split = split
        self.patch_size = patch_size
        self.c1 = c1
        self.c2 = c2
        # TODO: populate self.samples with a list of (cloudy_path, sar_path, cloudfree_path)
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        # TODO: load and preprocess real data here.
        cloudy = torch.rand(self.c1, self.patch_size, self.patch_size)
        sar = torch.rand(self.c2, self.patch_size, self.patch_size)
        cloudfree = torch.rand(self.c1, self.patch_size, self.patch_size)
        return cloudy, sar, cloudfree


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(
        c1=args.c1, c2=args.c2, feat_ch=args.feat_ch,
        num_stages=args.num_stages, num_heads=args.num_heads,
        attn_reduction_ratio=args.attn_reduction_ratio,
    ).to(device)
    D = Discriminator(c1=args.c1, c2=args.c2).to(device)

    loss_fn = DADIGANLoss(lambda1=args.lambda1, lambda2=args.lambda2)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    train_set = CloudRemovalDataset(args.data_root, split="train", patch_size=args.patch_size,
                                     c1=args.c1, c2=args.c2)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)

    for epoch in range(args.epochs):
        for step, (cloudy, sar, cloudfree) in enumerate(train_loader):
            cloudy, sar, cloudfree = cloudy.to(device), sar.to(device), cloudfree.to(device)

            # ---- Train Discriminator ----
            with torch.no_grad():
                fake = G(cloudy, sar)
            real_logits = D(cloudy, sar, cloudfree)
            fake_logits = D(cloudy, sar, fake)
            d_loss = loss_fn.discriminator_loss(real_logits, fake_logits)

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # ---- Train Generator ----
            fake = G(cloudy, sar)
            fake_logits = D(cloudy, sar, fake)
            g_loss, g_terms = loss_fn.generator_loss(fake_logits, fake, cloudfree)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} | D: {d_loss.item():.4f} | "
                      f"G: {g_loss.item():.4f} (adv={g_terms['adv']:.4f} "
                      f"l1={g_terms['l1']:.4f} kl={g_terms['kl']:.4f})")

        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(G.state_dict(), os.path.join(args.ckpt_dir, f"G_epoch{epoch+1}.pth"))
            torch.save(D.state_dict(), os.path.join(args.ckpt_dir, f"D_epoch{epoch+1}.pth"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--c1", type=int, default=13)
    p.add_argument("--c2", type=int, default=2)
    p.add_argument("--feat_ch", type=int, default=64)
    p.add_argument("--num_stages", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--attn_reduction_ratio", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--lambda1", type=float, default=100.0)
    p.add_argument("--lambda2", type=float, default=100.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
