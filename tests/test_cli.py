import os

import pandas as pd

from opn.cli import save_results, simulation_parser


def test_simulation_parser_defaults_and_overrides():
    parser = simulation_parser("desc", n_cells=10, n_reps=20, out_dir="results/x")
    args = parser.parse_args(["3"])
    assert (args.jobid, args.n_reps, args.out_dir) == (3, 20, "results/x")
    args = parser.parse_args(["4", "--n-reps", "1", "--out-dir", "elsewhere"])
    assert (args.jobid, args.n_reps, args.out_dir) == (4, 1, "elsewhere")


def test_save_results_creates_folder(tmp_path):
    out_dir = tmp_path / "a" / "b"
    path = save_results(pd.DataFrame({"x": [1, 2]}), str(out_dir), 7)
    assert path == os.path.join(str(out_dir), "output_7.csv")
    assert pd.read_csv(path)["x"].tolist() == [1, 2]
