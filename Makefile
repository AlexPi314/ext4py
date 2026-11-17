.PHONY: test check coverage help
help:
	./ext4py --help
test:
	python3 -m unittest discover -s tests -v
check:
	python3 -m compileall -q src tests scripts
	python3 -m unittest discover -s tests -v
coverage:
	python3 scripts/coverage.py
