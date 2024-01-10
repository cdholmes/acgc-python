# Process to distribute `acgc` to PyPI and conda-forge
1. Create a tag named `X.Y.Z[a|b|rc][n]` on the `main` branch.
2. Push the tag to GitHub: `git push --tags`.
   GitHub Actions will "Run Pytest" (test package on multiple Python versions) and "Upload to TestPyPI".
4. Confirm that all GitHub Actions complete successfully. Fix any errors.
5. Run `./create_docs.sh` to update documentation.
6. Commit and push updates to documentation.
   GitHub Actions will automatically update https://cdholmes.github.io/acgc-python
7. Draft a Release through the GitHub web interface. The release name should be `X.Y.Z`.
   GitHub Actions will "Upload to PyPI". Confirm that Actions complete successfully.
8. A conda-forge bot will automatically detect the PyPI changes and update the package on conda-forge.
   After several hours, emails should report whether the conda-forge update was successful.

## Updating the conda-forge recipe
In some cases, it may be necessary to update the conda-forge build recipe, 
such as when the package dependencies change. To update the recipe
1. Sync the fork at https://github.com/cdholmes/acgc-feedstock with upstream
2. Edit https://github.com/cdholmes/acgc-feedstock/blob/main/recipe, as needed.
3. Submit a Pull Request and approve it after all tests pass.
