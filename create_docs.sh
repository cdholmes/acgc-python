# use pdoc (https://pdoc3.github.io/pdoc/) to generate API docs
# Add module documentation in the __init__.py substring
# Can use '.. include documentation.md' in docstring

#pdoc3
#pdoc --html --force -o 'docs' acgc
#cp -r docs/acgc/* docs/
#rm -rf docs/acgc

pdoc -d numpy --math --logo "https://acgc.eoas.fsu.edu/wiki/_media/wiki/logo.png" -o 'docs' ./acgc
