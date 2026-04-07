import argparse
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="sample_submission.csv")
    parser.add_argument("--transformer-probs", type=str, required=True)
    parser.add_argument("--linear-probs", type=str, default=None)
    parser.add_argument("--transformer-weight", type=float, default=0.85)
    parser.add_argument("--linear-weight", type=float, default=0.15)
    parser.add_argument("--out", type=str, default="submission.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    sub = pd.read_csv(args.sample)
    transformer_probs = np.load(args.transformer_probs)
    probs = transformer_probs.copy()

    if args.linear_probs is not None and os.path.exists(args.linear_probs):
        linear_probs = np.load(args.linear_probs)
        w_sum = args.transformer_weight + args.linear_weight
        probs = (args.transformer_weight * transformer_probs + args.linear_weight * linear_probs) / w_sum

    sub["rate"] = probs.argmax(axis=1) + 1
    sub.to_csv(args.out, index=False)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
