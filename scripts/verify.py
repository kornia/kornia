#! /usr/bin/env python

"""Script that runs all verification steps.
"""

import argparse

import os
import shutil
from subprocess import run
from subprocess import CalledProcessError
import sys

def main(checks):
    try:
        print("Verifying with " + str(checks))
        if "lint" in checks:
            print("Linter (flake8):", flush=True)
            run("flake8 torchgeometry test examples", shell=True, check=True)
            print("lint checks passed")

        if "build-docs" in checks:
            print("Documentation (build):", flush=True)
            run("cd docs; make clean html", shell=True, check=True)

        if "check-docs" in checks:
            print("Documentation (check):", flush=True)
            run("./scripts/check_docs.py", shell=True, check=True)
            print("check docs passed")

    except CalledProcessError:
        # squelch the exception stacktrace
        sys.exit(1)

if __name__ == "__main__":
    checks = ['lint', 'build-docs', 'check-docs',]

    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', type=str, required=False, nargs='+', choices=checks)

    args = parser.parse_args()

    if args.checks:
        run_checks = args.checks
    else:
        run_checks = checks

main(run_checks)
