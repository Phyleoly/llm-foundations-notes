"""Run a small causal-attention check without datasets or trained weights."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "chi_2023_variance" / "src"))

from model import ModelConfig, NoPEModel


def run_smoke(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    config = ModelConfig(
        device=args.device,
        dtype=dtype,
        seq_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
    )
    model = NoPEModel(config).eval()
    shape = (args.batch_size, config.seq_length, config.hidden_dim)
    inputs = torch.randn(*shape, device=config.device, dtype=dtype, requires_grad=True)
    output = model(inputs)
    output.square().mean().backward()
    input_gradient_finite = bool(torch.isfinite(inputs.grad).all())
    parameter_gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )

    split = config.seq_length // 2
    changed = inputs.detach().clone()
    # Independent resampling can collapse to the same direction after
    # LayerNorm, especially at hidden_dim=2. Negation changes that direction.
    changed[:, split:] = -changed[:, split:]
    with torch.no_grad():
        changed_output = model(changed)
        unmasked = model(inputs.detach(), causal=False)
        unmasked_changed = model(changed, causal=False)

    reference = output.detach()
    prefix_difference = (reference[:, :split] - changed_output[:, :split]).abs().max().item()
    unmasked_difference = (unmasked[:, :split] - unmasked_changed[:, :split]).abs().max().item()
    atol, rtol, control_threshold = 1e-6, 1e-5, 1e-5
    torch.testing.assert_close(reference[:, :split], changed_output[:, :split], atol=atol, rtol=rtol)
    checks = {
        "output_shape_matches_input": tuple(output.shape) == shape,
        "output_finite": bool(torch.isfinite(output).all()),
        "input_gradient_finite": input_gradient_finite,
        "parameter_gradients_finite": parameter_gradients_finite,
        "causal_prefix_unchanged": True,
        "unmasked_control_changed": unmasked_difference > control_threshold,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Smoke checks failed: {checks}")

    source_files = ["smoke.py", "chi_2023_variance/src/model.py"]
    return {
        "scope": "operator_smoke_only",
        "description": "LayerNorm + MHA; no positional encoding, training, or paper result",
        "config": {
            "device": str(next(model.parameters()).device),
            "dtype": args.dtype,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "sequence_length": config.seq_length,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "layer_norm_eps": config.eps,
            "dropout": model.attention.dropout,
            "perturbation_start": split,
            "future_perturbation": "negate_inputs",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda,
        },
        "input": {"distribution": "standard_normal", "shape": list(shape)},
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "checks": checks,
        "metrics": {
            "causal_prefix_max_abs_diff": prefix_difference,
            "unmasked_prefix_max_abs_diff": unmasked_difference,
            "causal_comparison_atol": atol,
            "causal_comparison_rtol": rtol,
            "unmasked_control_threshold": control_threshold,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Write JSON to a new file; refuses overwrite")
    args = parser.parse_args()
    if args.batch_size < 1 or args.sequence_length < 2:
        parser.error("batch-size must be positive and sequence-length must be at least 2")
    if args.num_heads < 1 or args.hidden_dim < 2 or args.hidden_dim % args.num_heads:
        parser.error("hidden-dim must be at least 2 and divisible by a positive num-heads")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = json.dumps(run_smoke(args), ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
