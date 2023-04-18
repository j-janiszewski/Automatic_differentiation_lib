# Automatic differentiation in reverse mode - Julia library

Project made for the course:  "Algorithms in data engineering" of the Warsaw University of Technology

An example of how the library can be used for creating Convolutional neural networks can be found in *scripts/cnn.jl*.


It is authored by Jan Janiszewski.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "ad_lib"
```
which auto-activate the project and enable local path handling from DrWatson.
