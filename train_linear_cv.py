import argparse
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


WORD_CFG = dict(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.99,
    sublinear_tf=True,
    max_features=120000,
)

CHAR_CFG = dict(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=180000,
)


def normalize_text(text: str) -> str:
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def decision_to_probs(clf: SGDClassifier, X):
    scores = clf.decision_function(X)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return softmax(scores, axis=1)


def train_branch(train_texts, train_labels, valid_texts, test_texts, cfg, random_state):
    vec = TfidfVectorizer(**cfg)
    x_train = vec.fit_transform(train_texts)
    x_valid = vec.transform(valid_texts)
    x_test = vec.transform(test_texts)
    clf = SGDClassifier(
        loss="log_loss",
        alpha=2e-6,
        penalty="l2",
        max_iter=60,
        tol=1e-4,
        random_state=random_state,
    )
    clf.fit(x_train, train_labels)
    val_probs = decision_to_probs(clf, x_valid)
    test_probs = decision_to_probs(clf, x_test)
    return vec, clf, val_probs, test_probs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train.csv")
    parser.add_argument("--test", type=str, default="test.csv")
    parser.add_argument("--output-dir", type=str, default="runs/linear_tfidf")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    train_df["text"] = train_df["text"].map(normalize_text)
    test_df["text"] = test_df["text"].map(normalize_text)

    oof_probs = np.zeros((len(train_df), 5), dtype=np.float32)
    test_probs = np.zeros((len(test_df), 5), dtype=np.float32)
    fold_scores = []

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df["text"], train_df["rate"]), start=1):
        print(f"fold {fold}/{args.folds}")
        tr_text = train_df.iloc[tr_idx]["text"].tolist()
        va_text = train_df.iloc[va_idx]["text"].tolist()
        te_text = test_df["text"].tolist()
        y_tr = train_df.iloc[tr_idx]["rate"].to_numpy()
        y_va = train_df.iloc[va_idx]["rate"].to_numpy()

        word_vec, word_clf, word_val, word_test = train_branch(tr_text, y_tr, va_text, te_text, WORD_CFG, args.seed + fold)
        char_vec, char_clf, char_val, char_test = train_branch(tr_text, y_tr, va_text, te_text, CHAR_CFG, args.seed + 100 + fold)

        fold_val_probs = 0.5 * word_val + 0.5 * char_val
        fold_test_probs = 0.5 * word_test + 0.5 * char_test

        oof_probs[va_idx] = fold_val_probs
        test_probs += fold_test_probs / args.folds

        fold_pred = fold_val_probs.argmax(axis=1) + 1
        score = f1_score(y_va, fold_pred, average="weighted")
        fold_scores.append(score)
        print(f"val_weighted_f1={score:.6f}")

        joblib.dump(word_vec, os.path.join(args.output_dir, f"word_vectorizer_fold_{fold}.pkl"))
        joblib.dump(word_clf, os.path.join(args.output_dir, f"word_model_fold_{fold}.pkl"))
        joblib.dump(char_vec, os.path.join(args.output_dir, f"char_vectorizer_fold_{fold}.pkl"))
        joblib.dump(char_clf, os.path.join(args.output_dir, f"char_model_fold_{fold}.pkl"))

    oof_pred = oof_probs.argmax(axis=1) + 1
    oof_score = f1_score(train_df["rate"], oof_pred, average="weighted")
    print(f"cv_weighted_f1={oof_score:.6f}")

    np.save(os.path.join(args.output_dir, "oof_probs.npy"), oof_probs)
    np.save(os.path.join(args.output_dir, "test_probs.npy"), test_probs)

    metrics = {
        "fold_scores": fold_scores,
        "mean_fold_score": float(np.mean(fold_scores)),
        "oof_weighted_f1": float(oof_score),
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
