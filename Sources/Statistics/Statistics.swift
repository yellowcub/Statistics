import Foundation

/**
Calculate `n!` for values of `n` that conform to the BinaryInteger protocol.
*/
func factorial<T>(_ n: T) -> T where T: BinaryInteger {
    assert(n > 0, "Attempted to pass non-positive integer into StatisticalDistribution.factorial()")
    return T(tgamma(Double(n+1)))
}

/**
Calculate n-choose-k for values of `n` and `k` that conform to the BinaryInteger protocol.

- Parameters:
    - `n`: Number of items to choose from.
    - `k: number of items chosen.

- Returns: The number of possible unique selections of `k` items from `n`.
*/
func choose<T>(n: T, k: T) -> T where T: BinaryInteger {
    assert(k < n, "In StatisticalDistribution.choose(n: k: ) n must be larger than k")
    return factorial(n)/(factorial(k)*factorial(n-k))
}

/**
Calculates the mean of an array of values for types that satisfy the
BinaryFloatingPoint protocol (e.g Float, Double).

- Parameters:
    - data: Array of values

- Returns: The mean of the array of values or `nil` if the array was empty.
*/
func mean<T>(_ data: [T]) -> T where T: BinaryFloatingPoint {
    assert(!data.isEmpty, "Attempted to pass empty array into StatisticalDistribution.mean()")
    return data.reduce(T(0), +) / T(data.count)
}

/**
Calculates the unbiased sample variance for an array for types that satisfy
the BinaryFloatingPoint protocol (e.g Float, Double).

- Parameters:
    - data: Sample of values.  Note that this should contain at least two values.

- Returns: The unbiased sample variance or `nil` if `data` contains fewer than two
    values.
*/
func variance<T>(_ data: [T]) -> T where T: BinaryFloatingPoint {
    let n = data.count
    assert(n > 1, "In StatisticalDistribution.variance() size of Array passed must be greater than 1.")
    let mu = mean(data)
    
    let total = data.reduce(T(0)) { t, x in
        let xbar = x - mu
        return t + xbar*xbar
    }
    return total/T(n - 1)
}

/**
Calculates the unbiased sample standard deviation for an array of values
for types that satisfy the BinaryFloatingPoint protocol (e.g Float, Double).

- Parameters:
    - data: Sample of values.  Note that this should contain at least two values.

- Returns: The sample unbiased standard deviation or `nil` if `data` contains fewer
    than two values.
*/
func standardDeviation<T>(_ data: [T]) -> T where T: BinaryFloatingPoint {
    assert(data.count > 1, "In StatisticalDistribution.standardDeviation() size of Array passed must be greater than 1.")
    let v = variance(data)
    return sqrt(v)
}

/**
Calculates the population variance for an array of values for types that
satisfy the BinaryFloatingPoint protocol (e.g Float, Double).

- Parameters:
    - data: Values of population.  Note that this should contain at least one value.

- Returns: The population variance or `nil` if `data` contains fewer than one value.
*/
func pvariance<T>(_ data: [T]) -> T where T: BinaryFloatingPoint {
    let n = data.count
    assert(n > 0, "In StatisticalDistribution.pvariance() Array passed must not be empty")
    let mu = mean(data)
    
    let total = data.reduce(T(0)) { t, x in
        let xbar = x - mu
        return t + xbar*xbar
    }
    return total/T(n)
}

/**
Calculates the median of an array of values for types that
satisfy the BinaryFloatingPoint protocol (e.g Float, Double).

- Parameters:
    - data: Values of population.  Note that this should contain at least one value.

- Returns: The population variance or `nil` if `data` contains fewer than one value.
*/
func median<T>(_ data: [T]) -> T where T: BinaryFloatingPoint {
    assert(!data.isEmpty, "Attempted to pass empty array into StatisticalDistribution.median()")
    
    let n = data.count
    let s = data.sorted()
    if n % 2 == 1 { return s[n/2] }
    return (s[n/2]+s[(n/2)-1])/2
}

/**
Calculates the inverse error function of values for types that
satisfy the BinaryFloatingPoint protocol (e.g Float, Double).

Adapted from https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions 

- Parameter
    - y: where y = erf(x)
    - withPrecision: controls the precision the function must converge to.
- Returns: The inverse error function.
*/
func erfinv<T>(_ y: T, withPrecision epsilon: Double = 1e-7) -> T where T: BinaryFloatingPoint {
    assert(y > 0 && y < 1, "Argument in call to StatisticalDistribution.erfinv() must be between 0 and 1 exclusive.")
    
    let pi = Double.pi
    
    let sgn: Double = y < 0.0 ? -1.0 : 1.0
    let a: Double = 0.5*8.0*(pi-3.0)/(3.0*pi*(4.0-pi))
    let b: Double = 0.5*log(1.0 - Double(y*y))
    let c: Double = 1/(pi*a) + b
    var x: Double = sgn*sqrt(sqrt(c*c-b/a) - c)
    
    /// A few newton iterations to improve approximation
    var delta = erf(x) - Double(y)
    repeat {
        x = x - delta * ( 2.0/sqrt(pi) ) * exp(-x*x)
        delta = erf(x) - Double(y)
    } while abs(delta) > epsilon
    return T(x) 
}

/**
Calculates a least squares regression for `x` & `y` data sets that
satisfy the BinaryFloatingPoint protocol (e.g Float, Double).

- Parameters:
    - 'x': Independent data.
    - `y`: Dependent data.

- Returns: The slope and intercept for the best lsr line fit.
*/
func lsr<T>(x: [T], y: [T]) -> (slope: T, intercept: T) where T: BinaryFloatingPoint {
    assert(x.count == y.count && x.count > 1, "In StatisticalDistribution.lsr(x: y: ) the size of x and y must be the same and greater than 1.")
    
    let z = T(0.0)
    let (Tx, Ty, Txy, Txx) = x.enumerated().reduce((z,z,z,z)) { r, arg2 in
        let (i, xi) = arg2
        return (r.0 + xi, r.1 + y[i], r.2 + xi*y[i], r.3 + xi*xi)
    }

    let n = T(x.count)
    let b = (n*Txy - Tx*Ty)/(n*Txx - Tx*Tx)
    let a = (Ty - b*Tx)/n
    return (slope: b, intercept: a)
}