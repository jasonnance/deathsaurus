#!/bin/bash

# Run the various processes needed for CI.

set -e

isort --check-only .
black .
mypy . --ignore-missing-imports
flake8 . --config setup.cfg
