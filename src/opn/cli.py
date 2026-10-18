"""Command-line options and output files shared by the simulation scripts."""

import argparse
import os


def simulation_parser(description, n_cells, n_reps, out_dir):
    """Parser for ``script.py jobid [--n-reps N] [--out-dir DIR]``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("jobid", type=int, help=f"cell of the true-parameter grid to simulate (0-{n_cells - 1})")
    parser.add_argument("--n-reps", type=int, default=n_reps, help="simulated datasets per grid cell (default: %(default)s)")
    parser.add_argument("--out-dir", default=out_dir, help="folder for output_<jobid>.csv (default: %(default)s)")
    return parser


def save_results(results_table, out_dir, jobid):
    """Write ``out_dir/output_<jobid>.csv`` (creating out_dir) and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"output_{jobid}.csv")
    results_table.to_csv(output_path, index=False)
    return output_path
