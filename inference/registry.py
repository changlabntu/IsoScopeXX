"""Model registry — named, ready-to-load model specs for inference.

Each entry pins everything needed to reproduce a model's inference behavior:
which checkpoint (a timestamped run dir, its checkpoints/ parent, or the
experiment root), which models/*.py implements the staged generation_test for
those weights, which epoch, and the normalization the run was trained with.

Fields (only `checkpoint` is required):
    checkpoint  path accepted by inference.load.resolve_checkpoint_dir
    model_file  repo-relative models/*.py overriding the run's source snapshot
                (omit to fall back to Engine's auto-map / the snapshot)
    epoch       checkpoint epoch (omit/None = latest complete one)
    nm          normalization mode ('00', '01', '11', '11p', '11g');
                omit to use the run's config.json
    gamma, gamma_lo
                nm='11g' parameters; omit to use the run's config.json

Usage:
    from inference import Engine
    eng = Engine.from_registry('skipU')
    x = eng.normalize(vol)                    # vol: (Y, X, Z) in [-1, 1]
"""

MODELS = {
    # skipU: vqcleanM0aMSskipU recipe (skipE class + resize-conv netG
    # ed023emsfpnu), thx10 roiD192gfC/max5skip4. The newer of the two complete
    # runs in this experiment (both epochs 0-1100 at epoch_save=100); pinned to
    # the timestamped dir to disambiguate. Weights copied to this box under
    # /home/cheese/workspace/logs (2026-07-18).
    'skipU': dict(
        checkpoint='/home/cheese/workspace/logs/skipU/roiD192gfC/max5skip4/checkpoints/20260716_055927',
        model_file='models/MSclean.py',
        epoch=300,                  # 0-1100 available at epoch_save=100
        nm='11g', gamma=0.7, gamma_lo=-0.8,
    ),
    # Template for further entries (e.g. a vqclean run, once its weights exist
    # on this box):
    # 'vqclean': dict(
    #     checkpoint='/path/to/vqcleanVQ/experiment',
    #     model_file='models/vqclean.py',
    #     epoch=700,
    #     nm='00',
    # ),
}


def available():
    """Registered model names."""
    return sorted(MODELS)


def get(name):
    """The spec dict for `name` (a copy — mutating it does not edit the registry)."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Registered: {', '.join(available())}")
    return dict(MODELS[name])


def register(name, **spec):
    """Add or replace a registry entry at runtime (e.g. from a notebook)."""
    if 'checkpoint' not in spec:
        raise ValueError("spec needs at least 'checkpoint'")
    MODELS[name] = spec
