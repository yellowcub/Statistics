import XCTest
@testable import Statistics

class StatisticsTests: XCTestCase {
    
    let epsilon = 1e-7

    func testFunctions() throws {
        /// Test factorial()
        XCTAssertEqual(factorial(0), 1, "Failed factorial check.")
        XCTAssertEqual(factorial(1), 1, "Failed factorial check.")
        XCTAssertEqual(factorial(2), 2, "Failed factorial check.")
        XCTAssertEqual(factorial(3), 6, "Failed factorial check.")
        XCTAssertEqual(factorial(4), 24, "Failed factorial check.")
        XCTAssertEqual(factorial(5), 120, "Failed factorial check.")
        XCTAssertEqual(factorial(6), 720, "Failed factorial check.")
        XCTAssertEqual(factorial(7), 5_040, "Failed factorial check.")
        XCTAssertEqual(factorial(8), 40_320, "Failed factorial check.")
        XCTAssertEqual(factorial(9), 362_880, "Failed factorial check.")
        XCTAssertEqual(factorial(10), 3_628_800, "Failed factorial check.")
        
        /// Test choose()
        XCTAssertEqual(choose(n: 5, k: 3), 10, "Failed choose check.")

        /// Test mean()
        var numArr = [3.0, 3.0, 5.0, 9.0, 11.0]
        XCTAssertEqual(mean(numArr), 6.2, accuracy: epsilon, "Failed mean check.")

        /// Test variance()
        XCTAssertEqual(variance(numArr), 13.2, accuracy: epsilon, "Failed variance check.")

        /// Test standardDeviation()
        XCTAssertEqual(standardDeviation(numArr), 3.633180424917, accuracy: epsilon, "Failed standardDeviation check.")
        
        /// Test pvariance()
        XCTAssertEqual(pvariance(numArr), 10.56, accuracy: epsilon, "Failed variance check.")
        
        /// Test median()
        XCTAssertEqual(median(numArr), numArr[2], "Failed median odd array size check.")
        _ = numArr.popLast()
        XCTAssertEqual(median(numArr), 4.0, accuracy: epsilon, "Failed median even array size check.")

        /// Test erfinv()
        let x1 = erfinv(0.5)
        XCTAssertEqual(x1, 0.4769362762044698733814, accuracy: epsilon, "Failed erfinv() check.")

        /// Test lsr()
        let xdata = [0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
        let ydata = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        let (slope, intercept) = lsr(x: xdata, y: ydata)
        XCTAssertEqual(slope, 0.661016949, accuracy: epsilon, "Failed lsr() slope check.")
        XCTAssertEqual(intercept, 0.175423729, accuracy: epsilon, "Failed lsr() intercept check.")
    }
    
    func testDiscrete() throws {
        /// Test Bernoulli()
        let data = [1, 0, 1, 0, 0, 1]
        let bd = Bernoulli(data: data)
        XCTAssertEqual(bd.p, 0.5, "Failed Bernoulli init() check.")
        XCTAssertEqual(bd.quantile(0.75), 1, "Failed Bernoulli.quantile check.")

        /// Test GenericDiscreteDistribution
        let letters: [Character] = ["S", "O", "M", "E", "L", "E", "T", "T", "E", "R", "S"]
        let genD = CustomDiscrete(letters)
        XCTAssert(letters.contains(genD.random()), "Failed CustomDiscrete random check.")
    }

    func testContinuous() throws {
        /// Test GenericDiscreteDistribution
        let dataRange = 0.0...1.0
        let data = [0.3, 0.4, 0.2, 0.5, 0.3, 0.4, 0.8, 0.2, 0.9, 0.5, 0.6, 0.3]
        let genD = CustomRanked(data, withRange: dataRange)
        for _ in 1...10 {
            XCTAssert(dataRange.contains(genD.random()), "Failed CustomRanked random check.")
        }
    
    }

}
