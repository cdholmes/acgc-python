# use pdoc (https://pdoc3.github.io/pdoc/) to generate API docs
# Add module documentation in the __init__.py substring
# Can use '.. include documentation.md' in docstring
pdoc3 --html --force -o 'docs' acgc
cp -r docs/acgc/* docs/
