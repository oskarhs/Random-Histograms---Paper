# Random Irregular Histograms

Code supplementing the paper of Simensen et al. (2025).

To run the code, a working Julia installation is required. In addition, an R installation is needed along with KernSmooth package, which is available on cran.

To get started, clone the git repo and run the following commands to install all required Julia packages in the current environment.
```julia
using Pkg
Pkg.activate(".")
```

# A note on required R packages
The simulation study requires a working R installtation together with the following packages: 'KernSmooth', 'histogram' and 'pmden'.

# A note on running the simulation study
Running the script 'estimate_risk_methods' is quite time consuming, but the calculation can be sped up somewhat by utilizing the fact that the implementation used for regular histogram supports multithreading. As such, setting the number of threads to a value greater than one can offer some speedups when running this script.

## References
Simensen, O. H., Christensen, D. & Hjort, N. L. (2025). Random Irregular Histograms. _arXiv preprint_. doi: [10.48550/ARXIV.2505.22034](https://doi.org/10.48550/ARXIV.2505.22034)