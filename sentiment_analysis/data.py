import subprocess

import pandas as pd


def download_data():
    # DVC fetches the data automatically if not present
    # path = dvc.api.get_url(
    #    path="market_comments.csv",
    #    repo="https://github.com/KateZernyshkina/Datasets",
    #    rev="main"
    # )
    subprocess.run(
        [
            "dvc",
            "get",
            "https://github.com/KateZernyshkina/Datasets.git",
            "market_comments.csv",
        ],
        cwd="data",
        check=True,
    )
    return pd.read_csv("data/market_comments.csv")


if __name__ == "__main__":
    print(download_data())
