import argparse

import pandas as pd


def infer_label(image_path: str) -> int:
    path = str(image_path)
    if "member_" in path and "nonmember_" not in path:
        return 1
    if "nonmember_" in path:
        return 0
    raise ValueError(f"Could not infer membership label from image path: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="std_results.csv")
    parser.add_argument("--output", default="std_results_labeled.csv")
    parser.add_argument("--expected-per-class", type=int, default=500)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "image" not in df.columns:
        raise ValueError("Expected an 'image' column in the feature CSV.")

    df["label"] = df["image"].apply(infer_label)
    counts = df["label"].value_counts().to_dict()

    expected = {0: args.expected_per_class, 1: args.expected_per_class}
    if counts != expected:
        raise ValueError(
            f"Expected balanced labels {expected}, but found {counts}. "
            "Regenerate features before evaluating paper results."
        )

    df.to_csv(args.output, index=False)
    print(f"Saved labeled features: {args.output}")
    print(f"Label counts: {counts}")


if __name__ == "__main__":
    main()
