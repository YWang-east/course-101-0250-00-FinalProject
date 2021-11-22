# Final Project Repo

[![Build Status](https://github.com/eth-vaw-glaciology/FinalProjectRepo.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/eth-vaw-glaciology/FinalProjectRepo.jl/actions/workflows/CI.yml?query=branch%3Amaster)


[Add some short info here about the project, an abstract in a sense, and link to the documentation for **Part-1** and **Part-2**.]

## Meta-Info (delete this)

This project was generated with
```julia
using PkgTemplates
Template(;dir=".",
          plugins=[
                   Git(; ssh=true),
                   GitHubActions(; x86=true)],
        )("FinalProjectRepo")
```
Additionally, to the files generated by `PkgTemplates`, the following files and folders were added
- `scripts-part1/` and `scripts-part2/` which should contain the scripts for part 1 (solving the diffusion equation) and part 2 (solving your own equation)
- `docs/` the documentation (aka your final report), one for each part
- `test/part*.jl` testing scripts

Adapting this to your needs would entail:
- copy this repository (or clone)
- adapt the files (don't forget the LICENSE)
