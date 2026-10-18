PYTHON ?= python

.PHONY: test smoke demo notebook

# Unit tests for the opn library, incl. equivalence with the original analysis code (~1 min)
test:
	$(PYTHON) -m pytest

# Run every analysis script end to end with tiny settings (~5 min);
# saves tests/runs/<date-time>/report.html
smoke:
	$(PYTHON) tests/quick_test.py

# Hierarchical fit of the demo dataset and comparison with the true parameters
# (writes demo/output/; see the README section "Fitting a new dataset")
demo:
	$(PYTHON) demo/fit_demo.py --out-dir demo/output
	$(PYTHON) demo/check_recovery.py demo/output

# Execute the demo notebook without a browser and save its outputs in place
# (reuses demo/output/ from `make demo` if present, otherwise fits; see the notebook)
notebook:
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 demo/fit_demo.ipynb
