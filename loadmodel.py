"""
Load model parameters from either PyTorch (.pt) or NumPy (.npz) format.

This module allows build scripts to work with either format, enabling
CI environments to run without PyTorch installed.
"""

import json
import numpy as np

import modelspec


def load_spec_from_model(model_path: str, default_dual_bias: int = 3) -> dict:
    """Return the FROZEN resolved model spec baked into an artifact.

    Reads '_modelspec' (.npz) / 'modelspec' (.pt). For LEGACY artifacts that
    predate modelspec, reconstructs an equivalent spec from the architecture dict
    + dual-bias threshold via modelspec.from_legacy, so old models still build/
    eval byte-for-byte as before. This is the single seam every downstream stage
    uses to obtain the spec — never the live modelspec.py."""
    if model_path.endswith('.npz'):
        data = np.load(model_path)
        if '_modelspec' in data.files:
            return modelspec.from_json(bytes(data['_modelspec']).decode('utf-8'))
        arch = json.loads(bytes(data['_architecture']).decode('utf-8'))
        dbt = (int(data['_dual_bias_threshold'])
               if '_dual_bias_threshold' in data.files else default_dual_bias)
        return modelspec.from_legacy(arch, dbt)
    elif model_path.endswith('.pt'):
        import torch
        cp = torch.load(model_path, weights_only=False, map_location='cpu')
        if cp.get('modelspec'):
            return modelspec.from_json(cp['modelspec'])
        return modelspec.from_legacy(
            cp['architecture'], cp.get('dual_bias_threshold', default_dual_bias))
    raise ValueError(f"Unknown model format: {model_path} (expected .pt or .npz)")


def load_model_params(model_path: str) -> tuple[dict, dict, str]:
    """
    Load model parameters from .pt or .npz file.

    Returns:
        params: dict of quantized weights/biases (fc1_weight, fc1_bias, etc.)
        arch: architecture dict with input_size, hidden_sizes
        charset: character set string
    """
    if model_path.endswith('.npz'):
        return _load_npz(model_path)
    elif model_path.endswith('.pt'):
        return _load_pt(model_path)
    else:
        raise ValueError(f"Unknown model format: {model_path} (expected .pt or .npz)")


def _load_npz(model_path: str) -> tuple[dict, dict, str]:
    """Load from NumPy npz format."""
    data = np.load(model_path)

    arch = json.loads(bytes(data['_architecture']).decode('utf-8'))
    charset = bytes(data['_charset']).decode('utf-8')

    params = {k: data[k] for k in data.files if not k.startswith('_')}

    return params, arch, charset


def load_dual_bias_threshold(model_path: str, default: int = 3) -> int:
    """Return the dual-bias position threshold the model was trained with.

    The device must branch on the SAME threshold the sim used (train.py's
    DUAL_BIAS_THRESHOLD), otherwise on-calc generation silently diverges. The
    value is recorded by exportmodel.py (.npz '_dual_bias_threshold') and by
    train.py (.pt 'dual_bias_threshold'); falls back to `default` if absent.
    """
    if model_path.endswith('.npz'):
        data = np.load(model_path)
        if '_dual_bias_threshold' in data.files:
            return int(data['_dual_bias_threshold'])
        return default
    elif model_path.endswith('.pt'):
        import torch
        cp = torch.load(model_path, weights_only=False, map_location='cpu')
        return int(cp.get('dual_bias_threshold', default))
    return default


def _load_pt(model_path: str) -> tuple[dict, dict, str]:
    """Load from PyTorch checkpoint format."""
    import torch
    from train import NeochatModel, filter_legacy_state

    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    arch = checkpoint['architecture']
    charset = checkpoint['charset']

    model = NeochatModel(
        input_size=arch['input_size'],
        hidden_sizes=arch['hidden_sizes'],
        num_chars=arch['num_classes'],
    )
    model.load_state_dict(filter_legacy_state(checkpoint['model_state']))
    model.eval()

    params = model.get_quantized_params()

    return params, arch, charset
