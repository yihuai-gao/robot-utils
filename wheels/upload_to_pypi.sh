#! /bin/bash

set -e
rm -r dist
pip install --upgrade build
pip install twine
python -m build
twine upload dist/* --verbose