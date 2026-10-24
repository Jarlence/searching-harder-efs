"""Rebuild the processed data sets from the public sources (NEEDS NETWORK ACCESS; not needed when data/processed is present).
    python experiments/get_data.py          # downloads the 13 OpenML ARFF files (md5-checked against OpenML) and builds data/processed/*.npz + data cards
The Parkinson set (UCI 470) additionally needs the RAR member extracted once with 7-Zip into data/raw/uci/pd (see efsbench/datasets.py).
After building, `python experiments/verify_data.py` compares the files with data/data_cards.json."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from efsbench import datasets as D  # noqa: E402


def main():
    cards = []
    for did in D.OPENML:
        c = D.build_openml(did)
        print("built", c["name"], c["n"], c["p"], flush=True)
        cards.append(c)
    try:
        cards.append(D.build_parkinson()); print("built parkinson", flush=True)
    except FileNotFoundError as e:
        print("parkinson skipped:", e)
    print("done; data cards are written next to each data set in data/processed/ (data/data_cards.json is the archived copy)")


if __name__ == "__main__":
    main()
