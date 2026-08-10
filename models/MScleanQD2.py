from models.MSclean import GAN as MScleanGAN
from models.MScleanQD import GAN as MScleanQDGAN
import torch


# MScleanQD2: MScleanQD with the two wall-clock fixes (same objective, same
# flags, same expected loss — only WHERE and WHEN the prefix decode happens):
#
# M1 — lazy prefix decode. MScleanQD decodes the prefix inside generation(),
#   which base.py calls for BOTH optimizer steps; with --adv_prefix 0 the
#   d-step never reads recon_prefix, so half of all prefix decodes were
#   thrown away. Here generation() never decodes; backward_g() (and, only
#   when --adv_prefix > 0, backward_d()) draws and decodes on demand.
# M2 — rank-synchronized draw. MScleanQD drew per-rank np.random, so under
#   DDP the allreduce barrier made every step wait for the slowest rank:
#   with 4 ranks at p_prefix 0.5, ~94% of steps contained a firing rank and
#   wall-clock behaved like p_prefix≈1. Here the draw (fire + k) comes from
#   a torch.Generator seeded by global_step, identical on every rank: all
#   ranks fire together on the expected p_prefix fraction of steps. Side
#   effect: the draw is deterministic per optimizer step (reproducible);
#   consecutive accumulation micro-batches share a global_step and hence a
#   draw — benign correlation, marginal rate unchanged.
#
# Measured MScleanQD overhead vs MSclean was ~2x; expected after M1+M2 is
# ~1.25x (the residual is the genuine cost of the prefix decode + LPIPS on
# half the g-steps).
#
# NOTE (snapshot): train.py copies only models/MScleanQD2.py into the
# checkpoint dir; this file subclasses MScleanQD.py -> MSclean.py, so
# reproducing from a snapshot needs all three at the run's logged git hash.
class GAN(MScleanQDGAN):
    def generation(self, batch, deterministic=False):
        # skip MScleanQD's eager prefix decode; state is reset here so a
        # backward that does not draw sees no stale prefix from a prior step
        MScleanGAN.generation(self, batch, deterministic)
        self.recon_prefix = None
        self.prefix_scale = None

    def _maybe_prefix(self):
        """Rank-synchronized lazy draw + decode (fixes M1+M2)."""
        if not self.training or getattr(self.hparams, 'p_prefix', 0) <= 0 \
                or self.num_scales < 2 or self.recon_prefix is not None:
            return
        g = torch.Generator()
        g.manual_seed(0x51D0 + int(self.global_step))
        if float(torch.rand((), generator=g)) >= self.hparams.p_prefix:
            return
        k = int(torch.randint(1, self.num_scales, (), generator=g))
        self.prefix_scale = self.scale_factors[k - 1]
        self.recon_prefix = self.decode(sum(self.scale_latents[:k]))

    def backward_g(self):
        self._maybe_prefix()
        return super().backward_g()

    def backward_d(self):
        # only the prefix-adversarial variant needs the decode on d-steps;
        # with --adv_prefix 0 (the default) d-steps pay nothing
        if getattr(self.hparams, 'adv_prefix', 0) > 0:
            self._maybe_prefix()
        return super().backward_d()
