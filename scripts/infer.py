import argparse
import sys
from scripts.docking import docking4infer
from scripts.pre_process_testdata import preprocess
from DM_P450_model.src.infer import infer_wrapper


def main():
    # ------------------------------------------------------
    # Argument parser
    # ------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Inference wrapper for Seq-Only, Pocket-Only and DM-P450 modes."
    )

    parser.add_argument(
        "-model",
        type=str,
        required=True,
        choices=["Seq-Only", "Pocket-Only", "DM-P450"],
        help="Model type: Seq-Only | Pocket-Only | DM-P450",
    )

    parser.add_argument(
        "-inputFA",
        type=str,
        help="Input FASTA file (required for Seq-Only and DM-P450)",
    )
    parser.add_argument(
        "-substrate",
        type=str,
        help="Substrate SDF file (required for Seq-Only and DM-P450)",
    )

    args = parser.parse_args()

    model = args.model
    input_fa = args.inputFA
    substrate = args.substrate

    # ------------------------------------------------------
    # Mode-specific validation
    # ------------------------------------------------------
    if model == "Seq-Only":
        if not input_fa or not substrate:
            sys.exit("[ERROR] Seq-Only mode requires -inputFA and -substrate")
        preprocess(sdf_file_path=substrate, fasta_file_path=input_fa, way="Seq-Only")
        infer_wrapper(model_type="Seq-Only")
    elif model == "Pocket-Only":
        if not substrate:
            sys.exit("[ERROR] Pocket-Only mode requires -substrate")
        docking4infer(input_fa, substrate)
        preprocess(sdf_file_path=substrate, fasta_file_path=input_fa, way="Pocket-Only")
        infer_wrapper(model_type="Pocket-Only")
    elif model == "DM-P450":
        if not input_fa or not substrate:
            sys.exit("[ERROR] DM-P450 mode requires -inputFA and -substrate")
        docking4infer(input_fa, substrate)
        preprocess(sdf_file_path=substrate, fasta_file_path=input_fa, way="DM-P450")
        infer_wrapper(model_type="DM-P450")
    else:
        sys.exit(f"[ERROR] Unsupported model type: {model}")


if __name__ == "__main__":
    main()
