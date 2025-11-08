#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Hold-out, k-fold, and OOB demo on breast_cancer dataset")
    p.add_argument("--results-dir", type=str, default="results", help="Directory to write CSV/PNG/JSON artifacts")
    p.add_argument("--splits", type=int, default=5, help="Number of folds for StratifiedKFold")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splits and model")
    p.add_argument("--jobs", type=int, default=1, help="n_jobs for RandomForest (use 1 for bitwise reproducibility)")
    p.add_argument("--pos-label", type=int, default=1, help="Positive class label used for ROC-AUC column selection and F1 thresholding")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # Ensure the current directory is importable (so that eval_generalization.py is found)
    here = os.path.abspath(os.path.dirname(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    try:
        import eval_generalization as e
    except Exception as ex:
        print("ERROR: cannot import eval_generalization.py. Make sure it resides next to main.py.")
        raise

    # Override module-level config BEFORE running, and re-seed RNG
    e.RESULTS_DIR = args.results_dir
    e.N_SPLITS = int(args.splits)
    e.RANDOM_STATE = int(args.seed)
    e.N_JOBS = int(args.jobs)
    e.POS_LABEL = int(args.pos_label)
    try:
        e.np.random.seed(e.RANDOM_STATE)
    except Exception:
        pass

    # Run
    print(f"[INFO] Running demo with: results_dir={e.RESULTS_DIR}, splits={e.N_SPLITS}, seed={e.RANDOM_STATE}, jobs={e.N_JOBS}, pos_label={e.POS_LABEL}")
    e.main()
    print("[INFO] Done. See artifacts in:", os.path.abspath(e.RESULTS_DIR))

if __name__ == "__main__":
    main()
