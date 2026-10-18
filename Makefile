PYTHON ?= python

.PHONY: test smoke demo

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
