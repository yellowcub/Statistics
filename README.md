# Statistics

Statistics for Swift &mdash; v0.1.4

A custom implementation of [GitHub site](http://r0fls.github.io/swiftstats/) with the purpose of streamlining access to built-in functions and classes.  This is starting out as a personal project to facilitate the authors needs and intersts.

## Features

### Base Functions

- `factorial`
- `choose` (nCk)
- `median`
- `mean`
- `variance`
- `standardDeviation`
- `erfinv` (`erf` part of `Foundation`)
- `lsr`, least squares regression

### Discrete Distributions

- Bernoulli
- Poisson
- Geometric
- Binomial
- CustomDiscrete (new)

### Continuous Distributions

- Normal
- Log-normal
- Laplace
- Weibull (new)
- Exponential
- CustomRanked (new, needs pdf implementation)

And each distribution has these methods:

- pmf or pdf
- cdf
- quantile
- random (takes an optional int and returns an array of that length, or otherwise a single value)

## Testing

 Run `swift test` from the base directory in a terminal window.

## Contributing

If you would like to contribute, please submit a pull request, or raise an issue.

### TO-DO

## License

All code that I created in this repository (which is everything that was not generated by Xcode from a template, including the main source and the Unit Tests) is licensed under [CC0](https://creativecommons.org/publicdomain/zero/1.0/), which means that it is part of the public domain and you can do anything with it, without asking permission.
