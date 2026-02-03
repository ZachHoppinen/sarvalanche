#!/usr/bin/env python3

import argparse
import numpy as np
import subprocess
from pathlib import Path


def run_spatial_crf_densecrf_py38(
    U_path,
    Q_path,
    script_path,
    conda_env="py38",
    sxy=2,
    compat=5,
    sdims=(5, 5),
    schan=(0.1,),
    iters=3,
):
    """
    Run DenseCRF script inside a Python 3.8 conda environment.
    """

    cmd = [
        "conda", "run", "-n", conda_env,
        "python", str(script_path),
        str(U_path),
        str(Q_path),
        "--sxy", str(sxy),
        "--compat", str(compat),
        "--sdims", ",".join(map(str, sdims)),
        "--schan", ",".join(map(str, schan)),
        "--iters", str(iters),
    ]

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    return result


def parse_tuple(arg, type_=float):
    """
    Parse comma-separated values into a tuple.
    Example: "5,5" -> (5, 5)
    """
    return tuple(type_(x) for x in arg.split(","))


def run_crf(
    U_path,
    Q_path,
    sxy=2,
    compat=5,
    sdims=(5, 5),
    schan=(0.1,),
    n_iters=3,
):
    try:
        import pydensecrf.densecrf as dcrf
        import pydensecrf.utils as crf_utils
    except ModuleNotFoundError:
        raise ValueError("Dense CRF module can only be run with python 3.8 use py38_environment.yml")

    # Load unary energy
    U = np.load(U_path)
    _, H, W = U.shape

    # Flatten unary for DenseCRF
    U_flat = U.reshape((2, -1))

    # Initialize CRF
    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(U_flat)

    # Gaussian pairwise term
    d.addPairwiseGaussian(sxy=sxy, compat=compat)

    # Build feature image from class-1 energy
    img = np.asarray(np.exp(-U[1]), dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    features = img[:, :, None]

    # Bilateral pairwise energy
    pairwise_energy = crf_utils.create_pairwise_bilateral(
        sdims=sdims,
        schan=schan,
        img=features,
        chdim=2,
    )

    d.addPairwiseEnergy(pairwise_energy, compat=compat)

    # Inference
    Q = d.inference(n_iters)
    Q = np.array(Q).reshape(2, H, W)

    # Save output
    np.save(Q_path, Q)


def main():
    parser = argparse.ArgumentParser(
        description="Run DenseCRF inference on unary energy array"
    )

    parser.add_argument("U", help="Path to input unary energy (.npy)")
    parser.add_argument("Q", help="Path to output Q array (.npy)")

    parser.add_argument("--sxy", type=float, default=2,
                        help="Spatial Gaussian kernel size (default: 2)")
    parser.add_argument("--compat", type=float, default=5,
                        help="Compatibility coefficient (default: 5)")
    parser.add_argument("--sdims", type=lambda s: parse_tuple(s, float),
                        default=(5, 5),
                        help="Spatial dims for bilateral kernel, e.g. '5,5'")
    parser.add_argument("--schan", type=lambda s: parse_tuple(s, float),
                        default=(0.1,),
                        help="Channel sensitivity for bilateral kernel, e.g. '0.1'")
    parser.add_argument("--iters", type=int, default=3,
                        help="Number of CRF inference iterations (default: 3)")

    args = parser.parse_args()

    run_crf(
        U_path=args.U,
        Q_path=args.Q,
        sxy=args.sxy,
        compat=args.compat,
        sdims=args.sdims,
        schan=args.schan,
        n_iters=args.iters,
    )


if __name__ == "__main__":
    main()
