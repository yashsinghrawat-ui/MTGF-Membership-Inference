import subprocess


def main() -> None:
    subprocess.run(
        [
            "python3",
            "extract_research_features.py",
            "--output",
            "std_results.csv",
            "--n-per-class",
            "500",
            "--overwrite",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
