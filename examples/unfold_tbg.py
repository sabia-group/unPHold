import argparse
from pathlib import Path


def validate_file_complete(data_dpath: Path):
    pass
    # should exist: phonopy.yaml, tmat.npz, force_constants.h5, gp_pc.xyz
    # add everything required for this calculation


def main(*args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TBG unfolding example")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input data (default: None, then use ../tests/data/tbg/m2_r1_sc4); use ../docs/assets to update tutorial figures",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Directory for output figures (default: output); use ../docs/assets to update tutorial figures",
    )
    args = parser.parse_args()
    main(output=args.output)
