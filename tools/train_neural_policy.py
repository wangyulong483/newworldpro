"""Train / continue the neural behaviour + proto-language policy.

Examples::

    python tools/train_neural_policy.py --pretrain           # first-time bootstrap
    python tools/train_neural_policy.py --mode fine --steps 220 --epochs 10
    python tools/train_neural_policy.py --out assets/neural_policy.npz

In the GUI, press **F2** to trigger the same continuous fine-tuning; the policy
is never retrained on startup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newworld.training import DEFAULT_POLICY, fine_tune, pretrain  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the newworld neural policy")
    parser.add_argument("--mode", choices=("pretrain", "fine"), default=None,
                        help="pretrain from scratch or continue from --init/--out")
    parser.add_argument("--pretrain", action="store_true", help="shortcut for --mode pretrain")
    parser.add_argument("--init", default=None, help="checkpoint to continue from (default: --out)")
    parser.add_argument("--out", default=str(DEFAULT_POLICY))
    parser.add_argument("--steps", type=int, default=300, help="simulation steps per start hour")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    source = Path(args.init) if args.init else out
    mode = "pretrain" if args.pretrain else (args.mode or ("fine" if source.exists() else "pretrain"))

    if mode == "pretrain":
        policy, metrics = pretrain(out, seed=args.seed, steps_per_hour=args.steps, epochs=args.epochs)
        print(f"pretrained {out}")
    else:
        if args.init:
            from newworld.neural import NeuralPolicy

            NeuralPolicy.load(args.init)
        policy, metrics = fine_tune(source, seed=args.seed, steps_per_hour=args.steps, epochs=args.epochs)
        if source != out:
            policy.save(out)
        print(f"fine-tuned {out} from {source}")
    print(f"mode accuracy = {metrics['mode_accuracy']:.3f}  loss = {metrics['loss']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
