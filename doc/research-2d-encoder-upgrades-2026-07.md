# Research: upgrading the 2D VQ encoder (2026-07-12)

Deep-research synthesis (three parallel investigations: modern tokenizer/autoencoder encoders
2023–2026; efficient 2D attention & block designs for restoration; repo integration constraints)
on how to improve the 2D encoder — currently shallow, minimal attention — **while keeping it 2D**.

Status: research/proposal only. No code changed.

---

## 1. Current encoder — facts (verified in code)

`ldm/modules/diffusionmodules/modelcut.py::Encoder`, configured by `ldm/vqgan.yaml`:

| item | value | note |
|---|---|---|
| ch / ch_mult | 64 / [1,2,2,4] | feature maps 192²×64 → 96²×128 → 48²×128 → 24²×256 |
| num_res_blocks | **1** | SD-VAE uses 2 at ch=128 — we are ~10× smaller (≈4.5M params; decoder 7.4M) |
| attn_resolutions | `[]` | strips the **down-path** attention only |
| mid-block attention | **PRESENT & ACTIVE** | `modelcut.py:421` builds `mid.attn_1` unconditionally — global softmax attention at 24×24×256 runs today. "No attention" is only true above the bottleneck. |
| latent | z_channels=4, 24×24 | into per-scale `quant_convs` → VQ (n_embed **256**, embed_dim **4**) → multi-scale residual sum |
| input per forward | (24, 1, 192, 192) | Z-stack becomes the batch dim (cropz/dsp = 24 slices in both roiD and fuse regimes) — every per-slice cost is paid 24× |
| pretrained weights | none | everything trains from scratch; state-dict compatibility is a non-issue |

`modelcut` vs upstream LDM `model.py`: byte-identical except `Encoder.forward` returns
`(h, hbranch, hs)`; only `h` is consumed (`vqcleanM0aMSskipE.py:227`). `make_attn` supports
`vanilla` / `linear` / `none`.

Gradients reaching the encoder: (1) 2D VQGAN self-reconstruction (L1 + LPIPS + codebook;
`VQLPIPSWithDiscriminator`'s internal disc is delayed by `disc_start: 250001`), and
(2) the full 3D objective through `quant → decoder.conv_in → net_g → XupX` (six-way adv,
L1 projection, skipE pyramid/coarse terms).

### Hard constraints for any replacement

1. `in_channels=1` (2 with `--tc`).
2. **Exactly 3 spatial down-stages** (192→24). Coupled to net_g's 3 decode up-stages — changing
   encoder depth breaks the isotropic-cube geometry (`uprate=8`) unless net_g changes too.
3. `double_z=False` (else `h` gets 2·z_channels and mismatches `quant_convs`).
4. First return value must be a **spatial** `(N, z_channels, 24, 24)` map — the residual-VQ loop
   bilinear-resizes it; a flattened/1D-token latent breaks `encode()`.
5. **`ch·ch_mult[-1] = 256`** — `decoder.conv_in` output must match net_g's decode entry
   (`8·ngf = 256` at default `ngf=32`). Widening to block_in 512 (e.g. vqganB's [1,2,4,8])
   requires `--ngf 64` in lockstep.

Soft (self-adjusting from ddconfig, no net_g edits): `z_channels`, `embed_dim`, `n_embed`,
`num_res_blocks`, internal block types, mid `attn_type`. Swap point: the Encoder is hard-imported
(`from ldm...modelcut import Encoder, Decoder`) at the top of each `vqclean*` model — ~3 lines
per model file, no registry.

---

## 2. The reframing finding

The strongest cross-cutting result in the 2023–2026 tokenizer literature: **encoder capacity is
usually NOT the reconstruction bottleneck.**

- LiteVAE (NeurIPS 2024, [2405.14477](https://arxiv.org/abs/2405.14477)): an encoder **6× smaller**
  than SD-VAE's matches its reconstruction quality.
- GigaTok scaling study ([2504.08736](https://arxiv.org/abs/2504.08736)): "prioritize decoder
  scaling"; encoders kept smaller than decoders.
- SD3 / FLUX VAE: architecture essentially unchanged from SD-VAE — the reconstruction leap came
  from **latent channels 4→16(→32)**, i.e. relieving the z-channel bottleneck, not the encoder.
- MAGVIT-v2 ([2310.05737](https://arxiv.org/abs/2310.05737)): gains from the quantizer (LFQ) and
  decoder capacity, attention-free encoder.

This matches our setup uncomfortably well: `z_channels=4` into a **256-code, dim-4 VQ** is a hard
information bottleneck *before* encoder capacity even matters. The highest-evidence levers are at
the **latent interface**, not encoder depth. (In our pipeline the latent's real consumer is the 3D
`net_g` — latent quality matters relatively more, 2D-decoder polish relatively less, than in the
diffusion-tokenizer literature.)

---

## 3. Options, ranked

### A. Relieve the latent interface: z_channels 4→8/16 + FSQ (or LFQ) — highest evidence, ~zero cost

- **FSQ** ([2309.15505](https://arxiv.org/abs/2309.15505)): project to ~4–6 bounded dims, round
  each to L levels (e.g. levels [8,5,5,5] ≈ 1000 codes), straight-through gradients. **No
  codebook, no commitment loss, no EMA, no collapse; near-100% utilization by design.** Our tiny
  256-code VQ is exactly the regime where dead codes silently cap encoder gradient quality — any
  encoder-capacity work done before this can be eaten by a sick quantizer.
- **Compatibility with our VAR-style multi-scale residual VQ is proven**: Infinity
  ([2412.04431](https://arxiv.org/abs/2412.04431), CVPR 2025 oral) runs multi-scale residual
  quantization with a binary spherical quantizer (LFQ variant); ImageFolder
  ([2410.01756](https://arxiv.org/abs/2410.01756)) likewise. LFQ needs its entropy penalty to stay
  healthy and its huge implicit vocab (2^14+) may be oversized for one materials-texture class —
  FSQ is the simpler, safer fit.
- z_channels/embed_dim/n_embed are fully soft in this repo — `quant_convs`, `post_quant_convs`,
  `inject_convs`, `decoder.conv_in` all self-adjust from ddconfig; net_g untouched.
- Cheap adjuncts from ViT-VQGAN ([2110.04627](https://arxiv.org/abs/2110.04627)) if staying with
  VQ: L2-normalized (cosine) codebook lookup; we already have factorized low-dim codes.
- **This is also the cheapest falsification test** of whether the encoder/latent is the binding
  constraint at all.

### B. Bring the encoder to SD-VAE parity with modern blocks — the direct answer to "too shallow"

- **Depth**: `num_res_blocks` 1→2–3, using **NAFNet blocks**
  ([2204.04676](https://arxiv.org/abs/2204.04676)) rather than more ResnetBlocks. NAF block
  (LN → 1×1 expand → 3×3 depthwise → SimpleGate → simplified channel attention → 1×1, no
  activations) is *cheaper* than our current block (≈0.9 vs 2.7 GMAC/slice at 192²×64), is the
  reference restoration-efficiency baseline, conv-only (zero GAN risk), and was trained from
  scratch on single-domain data (SIDD) — exactly our regime.
- **Attention**: add **window attention at 48²** (2–4 Swin layers, window 8
  [2108.10257](https://arxiv.org/abs/2108.10257), or neighborhood attention k=7 via NATTEN
  [2204.07143](https://arxiv.org/abs/2204.07143)) and keep/enrich the existing 24² global
  mid-attention. Direct tokenizer precedent: **Efficient-VQGAN**
  ([2310.05400](https://arxiv.org/abs/2310.05400)) — local attention in a VQGAN first stage beat
  global on both reconstruction and speed. Cost at 48²/24² is ~0.3 GMAC/slice per block — noise.
  DiNAT-IR ([2507.17892](https://arxiv.org/abs/2507.17892)): pair local attention with channel
  attention (the NAF blocks already supply it).
- **Do NOT** put spatial softmax attention at 192²/96² — the 24-slice Z-batching pays every
  per-slice cost 24×. If global context is wanted at full res, **Restormer MDTA**
  (channel-wise attention, linear in N, [2111.09881](https://arxiv.org/abs/2111.09881)) is the
  affordable option (~1.5 GMAC/slice, no attention-map memory).
- **Width under the constraint**: `ch=128` with current mult → block_in 512 → needs `--ngf 64`.
  Cheap route keeping block_in=256: stay ch=64, deepen, and/or `ch_mult [1,2,4,4]`.
- MAGVIT-v2 hygiene applies (strided-conv down, no attention reliance, ~2× res blocks): that
  config level (≈SD-VAE parity, ~25M params) is where the field found encoder capacity stops
  mattering — a sensible target, not a starting point for further growth.

### C. Anti-aliased / lossless downsampling — targets OUR known failure mode

The encoder's stride-2 3×3 convs alias high frequencies into the latent — the same pathology we
fought in the discriminator (patchblur_16) and the upsampler (ed023emsfpnu). Two proven fixes,
both cheaper than what they replace:

- **Haar wavelet downsampling** (DWT rearranges 2×2 → 4 subband channels, lossless/invertible →
  1×1 conv; HWD, Pattern Recognition 2023, [code](https://github.com/apple1986/HWD); ~302M vs
  679M MACs at the 192→96 transition). Validated in GAN+LPIPS-trained autoencoders by LiteVAE
  ([2405.14477](https://arxiv.org/abs/2405.14477)) and NVIDIA Cosmos
  ([2501.03575](https://arxiv.org/pdf/2501.03575)) wavelet front-ends.
- **DC-AE residual space-to-channel shortcut** ([2410.10733](https://arxiv.org/abs/2410.10733),
  used by SANA): non-parametric pixel-unshuffle identity path around each downsample block
  (H×W×C → H/2×W/2×4C + channel-group averaging); convs learn only the residual, so HF survives
  downsampling losslessly. Near-free; headline gains are at f32+ so expect modest effect at our
  f8 — rank as adjunct, not standalone bet.

Given the lattice-artifact history, this option is the most likely to help *our specific* failure
mode rather than generic rFID.

### D. Cheap add-ons — one ablation each

- **EQ-VAE** ([2502.09509](https://arxiv.org/abs/2502.09509), ICML 2025): regularize the latent to
  be equivariant to rotations/scalings of the input (one extra forward per step, no architecture
  change, works on discrete AEs). Dovetails with the isotropy objective: latents consistent under
  rotation ↔ slices interchangeable across axes — and our dataloader already applies rotations.
  Speculative transfer (evidence is diffusion-consumer), but ~free to test.
- **One FFC/Fourier-Unit block at 48² on a zero-init residual branch** (LaMa,
  [2109.07161](https://arxiv.org/abs/2109.07161)): rFFT2 → 1×1 conv in frequency domain → irFFT2 —
  image-wide spectral receptive field, uniquely able to represent periodic structure, proven
  stable under PatchGAN training. **Guarded**: "Rethinking FFC" (ICCV 2023) shows BN+ReLU in the
  spectral path causes spectrum shifting and can *inject* periodic artifacts — drop BN/ReLU there,
  keep the global branch fraction ≤0.5. Conv-only fallback for receptive field: VAN large-kernel
  attention ([2202.09741](https://arxiv.org/abs/2202.09741)).

---

## 4. Explicitly not recommended

- **ViT / 1D tokenizer encoders** (ViT-VQGAN, TiTok [2406.07550](https://arxiv.org/abs/2406.07550),
  MAETok [2502.03444](https://arxiv.org/abs/2502.03444), FlowMo
  [2503.11056](https://arxiv.org/abs/2503.11056)): data-hungry (ImageNet-scale or distillation
  from pretrained tokenizers), no locality prior for one materials class, and TiTok's 1D tokens /
  FlowMo's iterative decoder break the spatial-latent contract with the 3D net_g.
- **MambaIR / VMamba** ([2402.15648](https://arxiv.org/abs/2402.15648),
  [2411.15269](https://arxiv.org/abs/2411.15269)): no evidence inside adversarially trained
  autoencoders, poor wall-clock at our short sequence lengths (N=576–2304), custom CUDA kernel
  vs our legacy PL/DDP stack; the field is converging back to attention hybrids anyway.
- **DINOv2-alignment latent regularizers** (VA-VAE [2501.01423](https://arxiv.org/abs/2501.01423),
  REPA-E [2504.10483](https://arxiv.org/abs/2504.10483)): RGB natural-image foundation-feature
  domain gap on 1-channel micrographs; revisit only if a DINOv2 probe on our slices shows
  meaningful features.
- **Linear/agent attention** (FLatten, Agent Attention): solves a large-N problem we don't have;
  known low-rank/blurring pathologies are a poor match for a HF-fidelity task.
- **RepLKNet-scale 31×31 kernels**: recognition-skewed evidence; at 24–48² maps a 13×13 kernel
  already sees half the map.

---

## 5. Recommendation & sequencing

1. **A + B as one "encoder v2" experiment**: FSQ (levels ≈ [8,5,5,5]) + z_channels 8, encoder
   deepened with NAF blocks (num_res_blocks 2–3 equivalent) + 2–4 window-attention layers at 48²,
   keeping block_in=256. A alone first if a minimal falsification test is preferred.
2. **C alongside** (wavelet or DC-AE shortcut downsampling) — low risk, cheaper than what it
   replaces, aimed at our aliasing pathway.
3. **D** as single-knob ablations after the new baseline exists.
4. Success metrics: 2D branch — `aeloss`/recon L1 + codebook utilization (FSQ makes this moot);
   3D branch — the usual val_kid / val_lpips_pred / val_lat_* (gf regime, thx-MS-384gf store);
   watch the lattice metrics specifically for option C.
5. If gains plateau after A–C, the literature says the next leverage point is **net_g/decoder
   capacity and the quantizer**, not more encoder attention (LiteVAE, GigaTok, MAGVIT-v2).

Memory note: encoder activation cost is dominated by the 192²×64 stage × 24 slices per item;
DDP means this does not amortize across GPUs. NAF blocks and wavelet downsampling both *reduce*
per-slice cost; window attention at ≤48² is negligible; avoid anything quadratic at ≥96².

## Sources

DC-AE [2410.10733](https://arxiv.org/abs/2410.10733) · DC-AE 1.5
[2508.00413](https://arxiv.org/abs/2508.00413) · LiteVAE
[2405.14477](https://arxiv.org/abs/2405.14477) · MAGVIT-v2
[2310.05737](https://arxiv.org/abs/2310.05737) · Open-MAGVIT2
[2409.04410](https://arxiv.org/abs/2409.04410) · FSQ
[2309.15505](https://arxiv.org/abs/2309.15505) · Infinity
[2412.04431](https://arxiv.org/abs/2412.04431) · ImageFolder
[2410.01756](https://arxiv.org/abs/2410.01756) · ViT-VQGAN
[2110.04627](https://arxiv.org/abs/2110.04627) · TiTok
[2406.07550](https://arxiv.org/abs/2406.07550) · MAETok
[2502.03444](https://arxiv.org/abs/2502.03444) · FlowMo
[2503.11056](https://arxiv.org/abs/2503.11056) · EQ-VAE
[2502.09509](https://arxiv.org/abs/2502.09509) · VA-VAE
[2501.01423](https://arxiv.org/abs/2501.01423) · REPA-E
[2504.10483](https://arxiv.org/abs/2504.10483) · GigaTok
[2504.08736](https://arxiv.org/abs/2504.08736) · Cosmos
[2501.03575](https://arxiv.org/pdf/2501.03575) · SD-VAE variant notes
[madebyollin gist](https://gist.github.com/madebyollin/ff6aeadf27b2edbc51d05d5f97a595d9) ·
SwinIR [2108.10257](https://arxiv.org/abs/2108.10257) · Efficient-VQGAN
[2310.05400](https://arxiv.org/abs/2310.05400) · HAT
[2205.04437](https://arxiv.org/abs/2205.04437) · NAT/NATTEN
[2204.07143](https://arxiv.org/abs/2204.07143) · DiNAT-IR
[2507.17892](https://arxiv.org/abs/2507.17892) · Restormer
[2111.09881](https://arxiv.org/abs/2111.09881) · NAFNet
[2204.04676](https://arxiv.org/abs/2204.04676) · MambaIR
[2402.15648](https://arxiv.org/abs/2402.15648) / v2
[2411.15269](https://arxiv.org/abs/2411.15269) · ConvNeXt v2
[2301.00808](https://arxiv.org/abs/2301.00808) · VAN
[2202.09741](https://arxiv.org/abs/2202.09741) · FFC/LaMa
[2109.07161](https://arxiv.org/abs/2109.07161) · Rethinking FFC
[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Chu_Rethinking_Fast_Fourier_Convolution_in_Image_Inpainting_ICCV_2023_paper.pdf) ·
HWD [Pattern Recognition 2023](https://github.com/apple1986/HWD) · BlurPool
[1904.11486](https://arxiv.org/abs/1904.11486)
