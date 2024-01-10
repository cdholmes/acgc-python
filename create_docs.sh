# use pdoc (https://pdoc.dev) to generate API docs
# Add module documentation in the __init__.py docstring
# Can use '.. include documentation.md' in docstring

# Temporarily rename loess.py because it causes an error in pdoc
mv acgc/stats/loess.py acgc/stats/loess.tmp_pdoc

# Generate documentation
pdoc --docformat numpy \
     --math \
     --logo "https://acgc.eoas.fsu.edu/wiki/_media/wiki/logo.png" -o 'docs' ./acgc !acgc.stats.loess

# Restore loess.py
mv acgc/stats/loess.tmp_pdoc acgc/stats/loess.py

#-----------------------------------
# Old version with pdoc3, https://pdoc3.github.io/pdoc
#pdoc --html --force -o 'docs' acgc
#cp -r docs/acgc/* docs/
#rm -rf docs/acgc

