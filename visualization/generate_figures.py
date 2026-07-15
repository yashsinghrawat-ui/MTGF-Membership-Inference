from pathlib import Path

import config
from visualization.plots import plot_metric_bar


def generate_figures(force: bool = False) -> Path:
    output = config.PATHS["figure8"]
    if output.exists() and not force:
        print(f"[plots] figure exists: {output}")
        return output

    path = plot_metric_bar(
        config.PATHS["metrics_csv"],
        output,
        metric="AUC",
        title="Temporal Gradient Feature Comparison",
    )
    print(f"[plots] saved figure: {path}")
    return path


def main() -> None:
    generate_figures(force=False)


if __name__ == "__main__":
    main()
