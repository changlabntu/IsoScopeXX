# ed023eMSdet: ed023eMS with DETACHED coarse-head taps (read-only heads).
#
# Identical architecture and weight layout to ed023eMS.Generator — only the decode
# forward differs: out64/out128 read their trunk features through .detach(), so no
# gradient from any coarse-head loss can reach the trunk. Together with a detached
# distillation target in the model (models/vqcleanMH.py), the full-res output's
# training signal is provably identical to a head-less (vqclean-recipe) run at every
# step — the heads are observers, not participants.

import torch

from networks.EncoderDecoder.ed023eMS import Generator as MSGenerator


class Generator(MSGenerator):

    def forward(self, x, method=None):
        # x (1, C, X, Y, Z)
        if method != 'decode':
            x = x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)
            feat = []
            for i in range(len(self.encoder)):
                if i > 0:
                    x = x.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
                    x = self.max3_pool(x)
                    x = x.squeeze(0).permute(3, 0, 1, 2)  # (Z, C, X, Y)
                x = self.encoder[i](x)
                feat.append(x.permute(1, 2, 3, 0).unsqueeze(0))
            if method == 'encode':
                return feat
            x = x.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)

        # Decode stages as in ed023eMS, but the coarse heads tap DETACHED features.
        x = self.up3(x)
        x = self.conv5(x)          # 1/4 res, 4*nf channels
        out64 = self.conv7_64(x.detach())
        x = self.up2(x)
        x = self.conv6(x)          # 1/2 res, 2*nf channels
        out128 = self.conv7_128(x.detach())
        x = self.up1(x)            # full res, nf channels
        x70 = self.conv7_k(x)
        x71 = self.conv7_g(x)

        return {'out0': x70, 'out1': x71, 'out128': out128, 'out64': out64}


if __name__ == '__main__':
    g = Generator(n_channels=1, norm_type='group', final='tanh', use_upsample=False)
    f = g(torch.rand(1, 1, 128, 128, 128), method='encode')
    out = g(f[-1], method='decode')
    print({k: tuple(v.shape) for k, v in out.items()})
