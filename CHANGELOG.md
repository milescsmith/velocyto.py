Changelog

# [0.44.0]

## Fixed

- Maybe fixed the issue where nothing actually gets installed?

# [0.43.0]

## Changed

- Updated deps, including updating to numpy 2.0 and numba 0.60.0

# [0.42.0]

## Changed

- Renamed the `doc` folder `docs`
- Altered `noxfile.py` to reflect the poetry -> pdm change

## Fixed

- Apparently I moved the `init_logging` function at one point and forgot to update everything else

# [0.41.0]

## Changed

- Replaced use of poetry as package manager with pdm

# [0.40.0]

## Changed

- At this point, lots.  More structural pattern matching, lots of typing fixes for mypy
- Removed black as formatter, using ruff
- Moved ruff config to its own file

# [0.31.0]

## Changed

- Anndata now uses gene symbols instead of ensembl geneids for var_names

# [0.30.1] and [0.30.2] - 2023-08-15

## Fixed

- removed some testing code

# [0.30.0] - 2023-08-15

## Changed

- No longer save the results of `run` as a loom file - save it directly as an anndata object
- replaced more "if...elif...else" with pattern matching
- `ExInCounter.readtranscriptmodels` now returns the number of features and number of
  chromosome examined because that is all that was being used


# [0.29.0] - 2023-08-07

## Fixed

- LOOKS LIKE VELOCYTO IS BACK ON THE MENU, BOYS!

## Changed

- Updated to require Python 3.10 as minimum version
- Updated to use Typer 0.9.0
- Updated dependencies
- Pulled a lot out of `velocyto.commands._run()` into separate functions
- Removed ability to add UMAP and cluster info to the loom file - that will be regenerated in other software, so why?


# [0.28.0] - 2023-02-24

## Changed

- Code cleanup and improvements thanks to sourcery


# [0.27.0] - 2023-02-23

## Changed

- Pass the actual "samtools_memory" value to samtools instead of checking the amount of memory available.
  - Also change to passing the argument as a string, allowing for the use of a prefix instead of requireing memory to be
  defined in megabytes.


# [0.26.0] - 2023-02-22

## Changed

- Various code cleanups
- Add ruff as package linter


# [0.20.0], [0.21.0], [0.22.0], [0.23.0], [0.24.0], and [0.25.0]

## Changed

- Various. No longer sure but a lot.


# [0.19.1] - 2022-06-30

## Fixed

- Fixed version issues


# [0.19.0] - 2022-06-29

## Changed

- Removed dumping all imports into __init__.py and then just importing the entire module (which lead to circular imports) and am importing as needed
- removed wildcard importing
- removed unused imports
- commented out a lot of unused assigned variables

## Fixed

- lots of changes to fix things flagged by flake8


# [0.18.0] - 2022-06-29

## Changed

- Black formatting
- Replaced setup.py with pyproject.toml
- remove distutils from build
- lock requirements and versions


[0.44.0]: https://github.com/milescsmith/velocyto.py/compare/0.43.0..0.44.0
[0.42.0]: https://github.com/milescsmith/velocyto.py/compare/0.41.0..0.42.0
[0.41.0]: https://github.com/milescsmith/velocyto.py/compare/0.40.0..0.41.0
[0.40.0]: https://github.com/milescsmith/velocyto.py/compare/0.31.0..0.40.0
[0.31.0]: https://github.com/milescsmith/velocyto.py/compare/0.30.2..0.31.0
[0.30.2]: https://github.com/milescsmith/velocyto.py/compare/0.30.1..0.30.2
[0.30.1]: https://github.com/milescsmith/velocyto.py/compare/0.30.0..0.30.1
[0.30.0]: https://github.com/milescsmith/velocyto.py/compare/0.29.0..0.30.0
[0.29.0]: https://github.com/milescsmith/velocyto.py/compare/0.28.0..0.29.0
[0.28.0]: https://github.com/milescsmith/velocyto.py/compare/0.27.1..0.28.0
[0.27.1]: https://github.com/milescsmith/velocyto.py/compare/0.27.0..0.27.1
[0.27.0]: https://github.com/milescsmith/velocyto.py/compare/0.26.0..0.27.0
[0.26.0]: https://github.com/milescsmith/velocyto.py/compare/0.25.0..0.26.0
[0.25.0]: https://github.com/milescsmith/velocyto.py/compare/0.24.0..0.25.0
[0.24.0]: https://github.com/milescsmith/velocyto.py/compare/0.23.0..0.24.0
[0.23.0]: https://github.com/milescsmith/velocyto.py/compare/0.22.0..0.23.0
[0.22.0]: https://github.com/milescsmith/velocyto.py/compare/0.21.0..0.22.0
[0.21.0]: https://github.com/milescsmith/velocyto.py/compare/0.20.0..0.21.0
[0.20.0]: https://github.com/milescsmith/velocyto.py/compare/0.19.0..0.20.0
[0.19.1]: https://github.com/milescsmith/velocyto.py/compare/0.19.0..0.19.1
[0.19.0]: https://github.com/milescsmith/velocyto.py/compare/0.18.0..0.19.0
[0.18.0]: https://github.com/milescsmith/velocyto.py/releases/tag/0.18.0
