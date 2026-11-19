"""KAIROS -- Kuramoto Assemblies Inferred from Rhythmic Oscillatory Synchrony.

The public entry point to the model this repository documents. KAIROS is variant ``253`` of
``full_benchmark.model.FreshModel``: a spiking core whose coupled oscillators gate the
dendritic and membrane updates, grouped into object masks by the temporal similarity of the
emitted spike traces. Variant 253 has ``head=None``; the masks come from the core's own
readout, with no mask head and no mask supervision anywhere in training.

Why the internals still carry their research names
--------------------------------------------------
``benchmark_codes/vendor/benchmark/`` is hash-locked: ``mod_benchmark.common.immutable()``
compares every file under it against ``vendor/mod_benchmark/BENCHMARK_LOCK.json`` on each call
and refuses to load the scorer if anything moved or changed. Renaming the packages around it
would break the one check that guarantees the published numbers were produced by unmodified
evaluation code. The tree is therefore shipped exactly as it ran, and this module is the
KAIROS-named surface over it.

Usage
-----
    import kairos
    model = kairos.build(seed=0)              # fresh weights, CPU
    model = kairos.load('/path/to/native64_core_253')   # the trained checkpoint

``load`` needs both ``resume.pt`` and ``initial.pt`` from that directory: the feature
normalization lives in ``initial.pt`` and the model cannot be built without it. See
``docs/checkpoints.md`` for the locations and hashes.
"""
import pathlib
import sys

__all__ = ['build', 'load', 'VARIANT', 'GRID', 'TOTAL_STEPS']

#: the variant number KAIROS is called throughout the research code and the experiment log
VARIANT = 253
#: patch grid; the model reads 128x128 RGB and predicts labels on a 64x64 grid
GRID = 64
#: default rollout length; the readout reads the second half (settle = TOTAL_STEPS // 2)
TOTAL_STEPS = 1024

_VENDOR = pathlib.Path(__file__).resolve().parent.parent / 'benchmark_codes' / 'vendor'


def _prepare_path():
    if not _VENDOR.is_dir():
        raise RuntimeError(f'benchmark_codes/vendor not found next to the kairos package: {_VENDOR}')
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))


def build(seed=0, normalization=None):
    """A fresh, untrained KAIROS on CPU.

    ``seed`` seeds the parameter initialization exactly as training did. ``normalization``
    is the encoder's feature mean/std record; leaving it ``None`` gives identity
    normalization, which is admission-only and not what the trained model uses.
    """
    _prepare_path()
    from full_benchmark.model import FreshModel
    return FreshModel(VARIANT, seed, normalization=normalization)


def load(checkpoint_dir, seed=0, map_location='cpu'):
    """KAIROS with the trained weights in ``checkpoint_dir``.

    The directory must hold ``resume.pt`` (the trained state) and ``initial.pt`` (which
    carries the feature normalization). Returns the model in eval mode.
    """
    import torch
    directory = pathlib.Path(checkpoint_dir)
    resume, initial = directory / 'resume.pt', directory / 'initial.pt'
    for path in (resume, initial):
        if not path.is_file():
            raise FileNotFoundError(f'{path} is required; see docs/checkpoints.md')
    normalization = torch.load(initial, map_location=map_location, weights_only=False).get('normalization')
    model = build(seed=seed, normalization=normalization)
    state = torch.load(resume, map_location=map_location, weights_only=False)
    model.load_state_dict(state['model'] if 'model' in state else state)
    return model.eval()
