# CMAES

[![Build Status](https://travis-ci.org/jonathanBieler/CMAES.jl.svg?branch=master)](https://travis-ci.org/jonathanBieler/CMAES.jl)

[![Coverage Status](https://coveralls.io/repos/jonathanBieler/CMAES.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/jonathanBieler/CMAES.jl?branch=master)

[![codecov.io](http://codecov.io/github/jonathanBieler/CMAES.jl/coverage.svg?branch=master)](http://codecov.io/github/jonathanBieler/CMAES.jl?branch=master)


## Usage:

```julia
using CMAES, Optim

f(x) = sum(abs2.(x))
D = 5

mfit = optimize(
    f,randn(D),CMAES.CMA(5;dimension=D),
    Optim.Options(f_calls_limit=5000,store_trace=true,extended_trace=true),
)

Optim.x_trace(mfit)
```