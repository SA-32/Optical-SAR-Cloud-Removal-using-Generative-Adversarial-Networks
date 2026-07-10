import torch
import torch.nn as nn


class ProxNet(nn.Module):

    def __init__(self, channels: int = 64, n_layers: int = 4):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PFEB(nn.Module):

    def __init__(self, feat_ch: int = 64, img_ch: int = 13):
        super().__init__()
        self.conv_xs = nn.Conv2d(feat_ch, img_ch, kernel_size=3, padding=1)
        self.conv_xp = nn.Conv2d(feat_ch, img_ch, kernel_size=3, padding=1)
        self.deconv_xp = nn.ConvTranspose2d(img_ch, feat_ch, kernel_size=3, padding=1)
        self.eta_p = nn.Parameter(torch.tensor(0.1))
        self.prox = ProxNet(feat_ch)

    def forward(self, S_prev: torch.Tensor, P_prev: torch.Tensor, I_o: torch.Tensor) -> torch.Tensor:
        Xs_S = self.conv_xs(S_prev)
        Xp_P = self.conv_xp(P_prev)
        residual = Xs_S + Xp_P - I_o
        grad = self.deconv_xp(residual)
        P_t = self.prox(P_prev - self.eta_p * grad)
        return P_t


class VFEB(nn.Module):

    def __init__(self, feat_ch: int = 64, img_ch: int = 2):
        super().__init__()
        self.conv_ys = nn.Conv2d(feat_ch, img_ch, kernel_size=3, padding=1)
        self.conv_yv = nn.Conv2d(feat_ch, img_ch, kernel_size=3, padding=1)
        self.deconv_yv = nn.ConvTranspose2d(img_ch, feat_ch, kernel_size=3, padding=1)
        self.eta_v = nn.Parameter(torch.tensor(0.1))
        self.prox = ProxNet(feat_ch)

    def forward(self, S_prev: torch.Tensor, V_prev: torch.Tensor, I_s: torch.Tensor) -> torch.Tensor:
        Ys_S = self.conv_ys(S_prev)
        Yv_V = self.conv_yv(V_prev)
        residual = Ys_S + Yv_V - I_s
        grad = self.deconv_yv(residual)
        V_t = self.prox(V_prev - self.eta_v * grad)
        return V_t


class SFEB(nn.Module):
    
    def __init__(self, feat_ch: int = 64, c1: int = 13, c2: int = 2, latent_ch: int = 64):
        super().__init__()
        self.conv_xp = nn.Conv2d(feat_ch, c1, kernel_size=3, padding=1)
        self.conv_yv = nn.Conv2d(feat_ch, c2, kernel_size=3, padding=1)
        self.enc_o = nn.Conv2d(c1, latent_ch, kernel_size=3, padding=1)
        self.enc_s = nn.Conv2d(c2, latent_ch, kernel_size=3, padding=1)

        self.conv_ls = nn.Conv2d(feat_ch, latent_ch * 2, kernel_size=3, padding=1)  # 128 channels
        self.deconv_ls = nn.ConvTranspose2d(latent_ch * 2, feat_ch, kernel_size=3, padding=1)
        self.eta_s = nn.Parameter(torch.tensor(0.1))
        self.prox = ProxNet(feat_ch)

    def forward(self, P_t: torch.Tensor, V_t: torch.Tensor, S_prev: torch.Tensor,
                I_o: torch.Tensor, I_s: torch.Tensor) -> torch.Tensor:
        I_o_tilde = I_o - self.conv_xp(P_t)
        I_s_tilde = I_s - self.conv_yv(V_t)
        I_tilde = torch.cat([self.enc_o(I_o_tilde), self.enc_s(I_s_tilde)], dim=1)  # 128 ch

        Ls_S = self.conv_ls(S_prev)
        residual = Ls_S - I_tilde
        grad = self.deconv_ls(residual)
        S_t = self.prox(S_prev - self.eta_s * grad)
        return S_t


class DDIN(nn.Module):

    def __init__(self, feat_ch: int = 64, c1: int = 13, c2: int = 2, num_stages: int = 3):
        super().__init__()
        self.num_stages = num_stages
        self.pfebs = nn.ModuleList([PFEB(feat_ch, c1) for _ in range(num_stages)])
        self.vfebs = nn.ModuleList([VFEB(feat_ch, c2) for _ in range(num_stages)])
        self.sfebs = nn.ModuleList([SFEB(feat_ch, c1, c2) for _ in range(num_stages)])

    def forward(self, P0: torch.Tensor, V0: torch.Tensor, S0: torch.Tensor,
                I_o: torch.Tensor, I_s: torch.Tensor):
        P, V, S = P0, V0, S0
        for t in range(self.num_stages):
            P_new = self.pfebs[t](S, P, I_o)
            V_new = self.vfebs[t](S, V, I_s)
            S_new = self.sfebs[t](P_new, V_new, S, I_o, I_s)
            P, V, S = P_new, V_new, S_new
        return P, V, S
