# TestDistributions.jl

A simple Julia package that provides a convenient interface for accessing the 16 distributions in the test density suite in the paper by [Simensen et al. (2025)](https://doi.org/10.48550/ARXIV.2505.22034).

## Functionality
In addition to the common `pdf`, `cdf`, `rand` and `support` methods for all test distributions, this package also provides a few convenience functions to determine other key quantities of the given distribution, see `?peaks`, `?discontinuities` and `?pid_tolerance`. The names of the supported distributions can be found by typing `TestDistributions.SUPPORTED_DISTRIBUTIONS`.

For simple vizualization of the test densities, we provide the functions `plot_test_density(d)` to plot the probability density function of a test distribution `d`, with reasonably chosen xlims. To plot all the supported test densities in a 4x4 grid, use `plot_all_test_densities()`.

## Installation
This package is not part of any Julia registries, and should as such be installed from github directly. To install the package, execute the two following lines in the repl (or similar).

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/oskarhs/Random-Histograms---Paper.git", subdir="TestDistributions"))
```
