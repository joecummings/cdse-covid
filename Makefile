default:
	@echo "an explicit target is required"

SHELL=/usr/bin/env bash

PYTHON_FILES=$(shell git ls-files --cached --others --exclude-standard '*.py' | sort | tr '\n' ' ')

export PYTHONPATH := ..

lint:
	pylint $(PYTHON_FILES)

docstyle:
	pydocstyle --convention=google $(PYTHON_FILES)

mypy:
	mypy $(PYTHON_FILES)

flake8:
	flake8 $(PYTHON_FILES)

black-fix:
	isort $(PYTHON_FILES)
	black --config pyproject.toml $(PYTHON_FILES)

black-check:
	isort --check $(PYTHON_FILES)
	black --config pyproject.toml --check $(PYTHON_FILES)

check: black-check flake8 mypy lint docstyle

precommit: black-fix check

install:
	bash setup.sh $1