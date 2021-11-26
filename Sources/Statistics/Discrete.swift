import Foundation

/**
Protocol for discrete distributions.

Defines the `quantile()` method that must be implemented.
*/
public protocol DiscreteDistribution {
    associatedtype Value
    func quantile(_ p: Double) -> Value
}

extension DiscreteDistribution {
    /**
    Single discrete random value using a user-provided random number generator

    - Parameters:
        - using: A random number generator

    - Returns:
    A random number from the distribution represented by the instance
    */
    public func random<T: RandomNumberGenerator>(using generator: inout T) -> Value {
        let x = Double.random(in: 0.0...1.0, using: &generator)
        return quantile(x)
    }

    /**
    Single discrete random value using the system random number generator

    - Returns: A random number from the distribution represented by the instance
    */
    public func random() -> Value {
        var rng = SystemRandomNumberGenerator()
        return random(using: &rng)
    }

    /**
    Array of discrete random values
    - Parameter n: number of values to produce
    
    Complexity: O(n)
    */
    public func random(_ n: Int) -> [Value] {
        var results: [Value] = []
        for _ in 0..<n {
            results.append(random())
        }
        return results
    }
    
}


//
public class Bernoulli: DiscreteDistribution {
    var p: Double
    
    public init(p: Double) {
        self.p = p
    }
        
    public convenience init(data: [Int]) {
        let m = mean(data.map({ Double($0) }))
        self.init(p: m)
    }
        
    public func pmf(_ k: Int) -> Double {
        if k == 1 { return self.p }
        if k == 0 { return 1 - self.p }
        return -1
    }
    
    public func cdf(_ k: Int) -> Double {
        if k < 0 { return 0 }
        if k < 1 { return 1 - self.p }
        if k >= 1 { return 1 }
        return -1
    }
    
    public func quantile(_ p: Double) -> Int {
        if p < 0 { return -1 }
        else if p < 1 - self.p { return 0 }
        else if p <= 1 { return 1 }
        return -1
    }
}

public class Poisson: DiscreteDistribution {
    var m: Double
    public init(m: Double) {
        self.m = m
    }

    public convenience init?(data: [Double]) {
        let m = mean(data)
        self.init(m: m)
    }

    public func pmf(_ k: Int) -> Double {
        return exp(Double(k) * log(m) - m - lgamma(Double(k+1)))
    }

    public func cdf(_ k: Int) -> Double {
        var total = Double(0)
        for i in 0..<k+1 {
            total += self.pmf(i)
        }
        return total
    }

    public func quantile(_ x: Double) -> Int {
        var total = Double(0)
        var j = 0
        total += self.pmf(j)
        while total < x {
            j += 1
            total += self.pmf(j)
        }
        return j
    }
}

public class Geometric: DiscreteDistribution {
    var p: Double
    public init(p: Double) {
        self.p = p
    }

    public convenience init?(data: [Double]) {
        let m = mean(data)
        self.init(p: 1/m)
    }

    public func pmf(_ k: Int) -> Double {
        return pow(1 - self.p, Double(k - 1))*self.p
    }

    public func cdf(_ k: Int) -> Double {
        return 1 - pow(1 - self.p, Double(k))
    }

    public func quantile(_ p: Double) -> Int {
        return Int(ceil(log(1 - p)/log(1 - self.p)))
    }
}

public class Binomial: DiscreteDistribution {
    var n: Int
    var p: Double
    public init(n: Int, p: Double) {
        self.n = n
        self.p = p
    }

    public func pmf(_ k: Int) -> Double {
        let r = Double(k)
        return Double(choose(n: self.n, k: k))*pow(self.p, r)*pow(1 - self.p, Double(self.n - k))
    }
    
    public func cdf(_ k: Int) -> Double {
        var total = Double(0)
        for i in 1..<k + 1 {
            total += self.pmf(i)
        }
        return total
    }
    
    public func quantile(_ x: Double) -> Int {
        var total = Double(0)
        var j = 0
        while total < x {
            j += 1
            total += self.pmf(j)
        }
        return j
    }
}

public class CustomDiscrete<ValueContainer>: DiscreteDistribution
    where ValueContainer: RandomAccessCollection, ValueContainer.Index == Int, ValueContainer.Element: Comparable {

    let min = 0
    var max: Int { return values.count - 1 }

    /// The explicit distribution of probabilities in this distrbution,
    /// with probabilities associated with their zero-based index.
    let values: ValueContainer
    
    public init(_ list: ValueContainer) {
		values = list
	}

    public var pmf: [Double] { 
        let c = Double(values.count)
        var pmf: [Double] = []
		for v in values {
			let p: Double = values.reduce(Double(0)){ base, nextElement in
                base + (nextElement == v ? 1.0 : 0.0)
            }
			pmf.append(p / c)
		}
        return pmf
    }

    public var cdf: [Double] {
        let c = pmf.reduce(into: []) { base, nextElement in
            base.append((base.last ?? 0) + nextElement)
        }
        return c.map(){ element in element / c.last! }
    }
    
    public func quantile(_ p: Double) -> ValueContainer.Element {
        let index = cdf.reduce(0) { $0 + ($1 < p ? 1 : 0) }
		return values[index]
    }

}