# DADIGAN (PyTorch)

A PyTorch implementation of **DADIGAN: A dual attention blocks-based
disentangled iterative Generative Adversarial Network for cloud and
shadow removal on SAR and optical images** (He, Wang, et al.,
*Information Fusion*, 2026).

This closely follows the architecture and equations described in the
paper (Sections 3.2–3.4, Figs. 3–7). The paper's public code repository
(`https://github.com/NUAA-RS/DADIGAN`) was not available to consult, so
this is a from-scratch re-implementation built directly from the paper's
text, equations, and figures — **not** a copy of the authors' code.

## Files

| File | Contents |
|---|---|
| `modules.py` | `ProxNet`, `PFEB`, `VFEB`, `SFEB`, `DDIN` — the deep disentangled iterative network (unfolded PGDA steps, Eqs. 8–20, Fig. 4) |
| `attention.py` | `CAB`, `MSAB`, `PDAFM` — the progressive dual attention fusion module (Eqs. 21–31, Fig. 5) |
| `reconstruction.py` | `ResidualBlock`, `ReconstructionBlock` — the reconstruction block (Fig. 6) |
| `generator.py` | `Generator` — wires init convs + DDIN + PDAFM + RB together (Fig. 3) |
| `discriminator.py` | `Discriminator` — the 5-block PatchGAN-style conditional discriminator (Fig. 7) |
| `losses.py` | `AdversarialLoss`, `L1Loss`, `SpectralKLLoss`, `DADIGANLoss` — the combined objective (Eqs. 32–35) |
| `train.py` | Example training loop with a stub dataset, matching the paper's reported hyperparameters |

## Quick start

```python
import torch
from dadigan import Generator, Discriminator, DADIGANLoss

# SEN12MS-CR setting: 13-band optical, 2-channel (VV, VH) SAR, 256x256 patches
G_model = Generator(c1=13, c2=2, feat_ch=64, num_stages=3, num_heads=8, attn_reduction_ratio=16)
D_model = Discriminator(c1=13)
loss_fn = DADIGANLoss(lambda1=100.0, lambda2=100.0)

cloudy = torch.rand(2, 13, 256, 256)   # I_o
sar    = torch.rand(2, 2, 256, 256)    # I_s
target = torch.rand(2, 13, 256, 256)   # cloud-free ground truth

result = G(cloudy, sar)                # reconstructed cloud-free image
```

Training:

```bash
python train.py --data_root /path/to/SEN12MS-CR --epochs 200 --batch_size 8 --lr 4e-5
```

`train.py`'s `CloudRemovalDataset` is a stub — plug in real loading of
SEN12MS-CR / SMILE-CR triplets (cloudy optical, SAR, cloud-free optical),
following the pre-processing in Section 4.1 of the paper (VV cropped to
`[-25, 0]`, VH to `[0, 32]`, optical bands cropped to `[0, 10000]`, all
normalized to `[0, 1]`).

## Hyperparameters used (from Section 4.2)

- Optimizer: Adam, lr = `4e-5` for both G and D
- ~200 epochs, batch size 8
- `lambda1 = lambda2 = 100` in the combined loss
- `T = 3` DDIN unfolding stages
- Input: 13-band optical + 2-channel dual-pol SAR, 256×256 patches

## Where the paper is descriptive rather than fully specified

The paper explains the architecture through prose and equations rather
than full pseudocode. A few points required an explicit, documented
engineering decision on my part — flagged here so you can revisit them
against your own reading of the paper or the authors' released code:

1. **`SFEB`'s 128-channel `L_s` filters.** The paper states all filters
   are 64-channel except `L_s`, which is 128. Concatenating the raw
   image-space residuals (13 + 2 = 15 channels) doesn't reach 128, so
   this implementation first encodes each modality's residual into a
   64-channel latent and concatenates those (64+64=128) before
   regressing `L_s(S)` against it.

2. **Attention cost / "matrix factorization."** The paper mentions using
   "a matrix factorization method to reduce the amount of calculation"
   for the CAB/MSAB attention maps, without giving the exact scheme.
   Full O(N²) attention over a 256×256 feature map (65,536 tokens) is
   very memory-heavy, so `CAB` and `MSAB` here use a spatial-reduction
   trick (average-pooling the Key/Value branch by `reduction_ratio`,
   similar to PVT's SRA) as a practical, documented stand-in. Increase
   `attn_reduction_ratio` (e.g. 16–32) if you hit memory limits at full
   resolution; decrease it for higher fidelity at smaller patch sizes.

3. **`ProxNet` (Fig. 4d).** Described as "we use Res-Net to implicitly
   learn the prior information." Implemented as 4× (Conv3×3 → ReLU)
   wrapped in one global residual connection.

4. **Discriminator conditioning (Fig. 7 / Eq. 33).** The figure labels
   the two stacked inputs "Reference image" and "Generated image," but
   Eq. 33's `D(m, n)` / `D(m, G(m))` makes clear the cloudy image `m` is
   the conditioning signal. This is implemented as a conditional
   discriminator: `D(I_o, candidate)`, where `candidate` is either
   the real cloud-free image or the generator's output.

5. **`eta_p`, `eta_v`, `eta_s`** (PGDA step sizes, "reciprocal of the
   Lipschitz constant") are implemented as learnable scalar parameters
   rather than fixed/precomputed constants, which is standard practice
   in deep-unfolding networks and consistent with the paper calling them
   part of the learned pipeline.

6. **No final activation on the generator's output.** The paper doesn't
   mention one on the Reconstruction Block. If your target images are
   normalized to `[0, 1]` (as in Section 4.1), consider adding a
   `Sigmoid` after `ReconstructionBlock.final_conv`, or clip/normalize
   post-hoc, for training stability.

## Verified

All modules were forward/backward-tested with dummy tensors (including
at the paper's native 256×256, 13/2-channel setting) to confirm shapes
and gradients flow correctly through the full Generator → Discriminator
→ loss pipeline.
