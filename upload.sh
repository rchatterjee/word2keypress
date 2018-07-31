pandoc --columns=100 --output=README --to rst README.md
python setup.py clean
rm -rf dist/*
python setup.py sdist
twine upload -r pypi dist/*
