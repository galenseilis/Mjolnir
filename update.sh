rm -r dist

python -m build

twine upload dist/*

pip install Thor-Hammer --upgrade
