"""
Loss functions for DADIGAN. Eqs. (32)-(35).

    loss = argmin_G max_D  L_cGAN(G, D) + lambda1 * L1(G) + lambda2 * L_KL

    L_cGAN(G, D) = E[log D(m, n)] + E[log(1 - D(m, G(m)))]
    L1(G)        = (1 / (H*W*C1)) * ||n - G(m)||_1
    L_KL         = KL( p(G(m)) || q(n) )

We implement the adversarial term with the standard non-saturating BCE
formulation (BCEWithLogitsLoss against the discriminator's scalar
logit), which is the common, numerically-stable realization of the
vanilla GAN objective written in Eq. (33).

For L_KL, since p(.) and q(.) are described only as "probability
distributions" over the generated/target images without a concrete
binning scheme, we implement a standard, differentiable approximation:
per-image, per-channel spatial softmax of pixel intensities is treated
as an empirical distribution, and KL divergence is computed between the
generated and target distributions, averaged over channels and batch.
This is a documented modeling choice for an underspecified step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        loss_real = self.bce(real_logits, real_labels)
        loss_fake = self.bce(fake_logits, fake_labels)
        return loss_real + loss_fake

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        real_labels = torch.ones_like(fake_logits)
        return self.bce(fake_logits, real_labels)


class L1Loss(nn.Module):

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_pixels = generated.shape[1] * generated.shape[2] * generated.shape[3]
        return torch.sum(torch.abs(target - generated)) / (generated.shape[0] * n_pixels)


class SpectralKLLoss(nn.Module):

    def forward(self, generated: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        B, C, H, W = generated.shape
        gen_flat = generated.reshape(B, C, H * W)
        tgt_flat = target.reshape(B, C, H * W)

        log_p = F.log_softmax(gen_flat, dim=-1)
        q = F.softmax(tgt_flat, dim=-1)

        # KL(p || q) = sum p * (log p - log q); using kl_div(input=log_q, target=p)
        # torch's kl_div(input, target) computes target * (log(target) - input),
        # i.e. expects `input` = log(q) and `target` = p for KL(p || q).
        log_q = torch.log(q + eps)
        p = torch.exp(log_p)
        kl = F.kl_div(log_q, p, reduction="batchmean")
        return kl


class DADIGANLoss(nn.Module):

    def __init__(self, lambda1: float = 100.0, lambda2: float = 100.0):
        super().__init__()
        self.adv = AdversarialLoss()
        self.l1 = L1Loss()
        self.kl = SpectralKLLoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def generator_loss(self, fake_logits: torch.Tensor, generated: torch.Tensor, target: torch.Tensor):
        adv_loss = self.adv.generator_loss(fake_logits)
        l1_loss = self.l1(generated, target)
        kl_loss = self.kl(generated, target)
        total = adv_loss + self.lambda1 * l1_loss + self.lambda2 * kl_loss
        return total, {"adv": adv_loss.item(), "l1": l1_loss.item(), "kl": kl_loss.item()}

    def discriminator_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor):
        return self.adv.discriminator_loss(real_logits, fake_logits)
