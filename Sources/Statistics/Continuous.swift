import Foundation

/**
    Protocol for continuous distributions.
    
    Defines the `quantile()` method that must be implemented.
    */
public protocol ContinuousDistribution {
        associatedtype FType where FType: BinaryFloatingPoint
        func quantile(_ p: Double) -> FType
}

extension ContinuousDistribution {
    /**
        Single discrete random value using a user-provided random number generator

        - Parameters:
            - using: A random number generator

        - Returns:
        A random number from the distribution represented by the instance
    */
    public func random<T: RandomNumberGenerator>(using generator: inout T) -> FType {
        let x = Double.random(in: 0.0...1.0, using: &generator)
        return quantile(x)
    }


    /**
        Single discrete random value using the system random number generator

        - Returns:
        A random number from the distribution represented by the instance
    */
    public func random() -> FType {
        var rng = SystemRandomNumberGenerator()
        return random(using: &rng)
    }


    /**
        Array of discrete random values
        - Parameter n: number of values to produce
        - Complexity: O(n)
    */
    public func random(_ n: Int) -> [FType] {
        var results: [FType] = []
        for _ in 0..<n { results.append(random()) }
        return results
    }
    
}

public class Laplace<T>: ContinuousDistribution where T: BinaryFloatingPoint {
    var mean: Double
    var b: Double

    public init (mean: T, b: T) {
        self.mean = Double(mean)
        self.b = Double(b)
    }

    public convenience init?(data: [T]) {
        let m = median(data)
        
        var b = T(0)
        for d in data {
            b = b + abs(d - m)
        }
        b = b/T(data.count)
        self.init(mean: m, b: b)
    }

    public func pdf(_ x: T) -> Double {
        return 0.5 * exp(-abs(Double(x) - mean) / b)
    }

    public func cdf(_ x: T) -> Double {
        let xd = Double(x)
        let z = (xd - mean) / b
        if xd < mean {
            return 0.5 * exp(z)
        }
        return 1.0 - 0.5 * exp(-z)
    }

    public func quantile(_ p: Double) -> T {
        assert((0...1).contains(p), "Cannot specify a quantile probability outside the range 0...1")
        if p < 0.5 {
            return T(mean + b * log(2*p))
        }
        return T(mean - b * log(2*(1-p)))
    }
}

public class Exponential<T>: ContinuousDistribution where T: BinaryFloatingPoint {
    var l: Double
    public init(l: T) {
        self.l = Double(l)
    }

    public convenience init?(data: [T]) {
        let m = mean(data)
        
        self.init(l: 1/m)
    }

    public func pdf(_ x: T) -> Double {
        return l*exp(-l*Double(x))
    }

    public func cdf(_ x: T) -> Double {
        return 1.0 - exp(-l*Double(x))
    }

    public func quantile(_ p: Double) -> T {
        assert((0...1).contains(p), "Cannot specify a quantile probability outside the range 0...1")
        return T( -1.0 * log(1 - p)/l )
    }
}

public class Weibull<T>: ContinuousDistribution where T: BinaryFloatingPoint {
    // mean and variance
    var scale: Double
    var shape: Double

    public init(scale: T, shape: T) {
        self.scale = Double(scale)
        self.shape = Double(shape)
    }

    public func pdf(_ x: T) -> Double {
        if x < 0 { return Double(0) }
        return shape/scale * pow(Double(x)/scale, shape-1.0) * exp(-pow(Double(x)/scale, shape))
    }

    public func cdf(_ x: T) -> Double {
        if x < 0 { return Double(0) }
        return 1.0 - exp(-pow(Double(x)/scale, shape))
    }

    public func quantile(_ p: Double) -> T {
        assert((0...1).contains(p), "Cannot specify a quantile probability outside the range 0...1")
        return T(scale * pow(-1.0*log(1.0 - p), 1.0/shape))
    }
}

public class Normal<T>: ContinuousDistribution where T: BinaryFloatingPoint {
    private let pi = Double.pi
    
    // mean and variance
    var m: Double
    var v: Double

    public init(m: T, v: T) {
        self.m = Double(m)
        self.v = Double(v)
    }
    
    public convenience init(mean: T, sd: T) {
        // This contructor takes the mean and standard deviation, which is the more
        // common parameterisation of a normal distribution.
        let variance = T(pow(Double(sd), 2.0))
        self.init(m: mean, v: variance)
    }

    public convenience init?(data: [T]) {
        // this calculates the mean twice, since variance()
        // uses the mean and calls mean()
        let v = variance(data)
        let m = mean(data)
        self.init(m: m, v: v)
    }

    public func pdf(_ x: T) -> Double {
        return (1/sqrt(v*2*pi)) * exp(-pow(Double(x)-m,2)/(2*v))
    }

    public func cdf(_ x: T) -> Double {
        return (1 + erf((Double(x)-m)/pow(2*v,0.5)))/2
    }

    public func quantile(_ p: Double) -> T {
        assert((0...1).contains(p), "Cannot specify a quantile probability outside the range 0...1")
        return T( m + sqrt(v*2)*erfinv(2*p - 1) )
    }
}

/**
    The log-normal continuous distribution.
    
    Three constructors are provided.
    
    There are two parameter-based constructors; both take the mean of the
    distribution on the log scale.  One constructor takes the variance of
    the distribution on the log scale, and the other takes the standard
    deviation on the log scale.  See `LogNormal.init(meanLog:varianceLog:)` and
    `LogNormal.init(meanLog:sdLog:)`.
    
    One data-based constructor is provided.  Given an array of sample values,
    a log-normal distribution will be created parameterised by the mean and
    variance of the sample data.
*/
public class LogNormal<T>: ContinuousDistribution where T: BinaryFloatingPoint {
    private let pi = Double.pi
    
    // Mean and variance
    var m: Double
    var v: Double
    
    /**
        Constructor that takes the mean and the variance of the distribution
        under the log scale.
        */
    public init(meanLog: T, varianceLog: T) {
        self.m = Double(meanLog)
        self.v = Double(varianceLog)
    }
    
    /**
        Constructor that takes the mean and the standard deviation of the
        distribution under the log scale.
        */
    public convenience init(meanLog: T, sdLog: T) {
        // This contructor takes the mean and standard deviation, which is
        // the more common parameterisation of a log-normal distribution.
        let varianceLog = T(pow(Double(sdLog), 2.0))
        self.init(meanLog: meanLog, varianceLog: varianceLog)
    }
    
    /**
        Constructor that takes sample data and uses the the mean and the
        standard deviation of the sample data under the log scale.
        */
    public convenience init?(data: [T]) {
        // This calculates the mean twice, since variance()
        // uses the mean and calls mean()

        let logData = data.map(){ element in T(log(Double(element))) }
        
        let v = variance(logData)
        let m = mean(logData)
        self.init(meanLog: m, varianceLog: v)
    }
    
    public func pdf(_ x: T) -> Double {
        return 1/(Double(x)*sqrt(2*pi*v)) * exp(-pow(log(Double(x))-m,2)/(2*v))
    }
    
    public func cdf(_ x: T) -> Double {
        return 0.5 + 0.5 * erf((log(Double(x))-m)/sqrt(2*v))
    }
    
    public func quantile(_ p: Double) -> T {
        assert((0...1).contains(p), "Cannot specify a quantile probability outside the range 0...1")
        return T( exp(m + sqrt(2*v) * erfinv(2*p - 1)) )
    }

}

public class Uniform<T>: ContinuousDistribution where T: BinaryFloatingPoint {
    // a and b are endpoints, that is
    // values will be distributed uniformly between points a and b
    var a: Double
    var b: Double

    public init(a: T, b: T) {
        self.a = Double(a)
        self.b = Double(b)
    }

    public func pdf(_ x: T) -> Double {
        let xd = Double(x)
        if (a...b).contains(xd) {
            return 1/(b - a)
        }
        return 0
    }

    public func cdf(_ x: T) -> Double {
        let xd = Double(x)
        
        if xd < a { return 0 }
        if xd < b { return (xd - a)/(b - a) }
        return 1
    }

    public func quantile(_ p: Double) -> T {
        assert((0...1).contains(p), "Cannot specify a quantile probability outside the range 0...1")
        return T( p*(b-a)+a )
    }
}

public class CustomRanked<T>: ContinuousDistribution where T: BinaryFloatingPoint {
    /// The explicit distribution of probabilities in this distrbution,
    /// with probabilities associated with their zero-based index.
    public let values: [T]
    public let cdf: [Double]
    
    public init(_ list: [T], withRange range: ClosedRange<T>) {
		let v = list.filter({ element in range.contains(element) }).sorted(by: <)
		values = [range.lowerBound] + v + [range.upperBound]
		
		let N = Double(v.count)
		let c = v.enumerated().map(){ (i, _) in
            (Double(i + 1) - 0.3) / (N + 0.4)
        }
		cdf = [0.0] + c + [1.0]
    }
    
    public func quantile(_ p: Double) -> T {
        let index = cdf.reduce(0) { $0 + ($1 < p ? 1 : 0) }
		return values[index]
    }

}