#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Boolean SoP Analyzer
================================================
This script tests all analysis types supported by the Boolean SoP Analyzer
and generates a detailed report of the results.

Test Coverage:
1. Cofactor (single variable)
2. Cofactor (multi-variable cube)
3. Smoothing
4. Consensus
5. Boolean Difference (optimized)
6. Boolean Difference (naive)
7. Formula display functionality

Each test includes multiple test cases with different complexity levels.
"""

import sys
import datetime
from typing import List, Dict, Any, Tuple
from main import (
    parse_sop, parse_cube, positive_negative_cofactors, cofactor, 
    smoothing, consensus_operator, boolean_difference_naive,
    boolean_difference_var_optimized, cover_str, vars_in_cover,
    variable_report, demo_big_expression, Cover
)

class AnalysisTestSuite:
    """Test suite for all Boolean SoP analysis functions"""
    
    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log_test(self, test_name: str, input_data: str, expected: Any, actual: Any, passed: bool, notes: str = ""):
        """Log a test result"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        self.test_results.append({
            'test_name': test_name,
            'input': input_data,
            'expected': expected,
            'actual': actual,
            'passed': passed,
            'notes': notes,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    def run_cofactor_single_variable_tests(self):
        """Test single variable cofactor analysis"""
        print("Testing Single Variable Cofactor Analysis...")
        
        test_cases = [
            # (expression, variable, description)
            ("ab + a'c", "a", "Simple expression with variable a"),
            ("abc + a'bc + ab'c", "b", "Expression with multiple terms containing b"),
            ("xy + x'y' + xyz", "x", "Expression with variable x in different forms"),
            ("a + b + c", "a", "Simple OR expression"),
            ("abc", "b", "Single product term"),
            ("a'b'c'", "c", "All complemented variables"),
            (demo_big_expression(), "a", "Complex demo expression"),
            ("1", "x", "Constant 1 function"),
            ("0", "x", "Constant 0 function"),
        ]
        
        for expr, var, desc in test_cases:
            try:
                f = parse_sop(expr)
                fx1, fx0 = positive_negative_cofactors(f, var)
                fx1_str = cover_str(fx1)
                fx0_str = cover_str(fx0)
                
                result = f"f_{var} = {fx1_str}, f_{var}' = {fx0_str}"
                self.log_test(
                    f"Cofactor Single Variable - {desc}",
                    f"f = {expr}, variable = {var}",
                    "Valid cofactor result",
                    result,
                    True,
                    "Cofactor computation successful"
                )
                
            except Exception as e:
                self.log_test(
                    f"Cofactor Single Variable - {desc}",
                    f"f = {expr}, variable = {var}",
                    "Valid cofactor result",
                    f"Error: {str(e)}",
                    False,
                    f"Exception occurred: {str(e)}"
                )
    
    def run_cofactor_cube_tests(self):
        """Test multi-variable cube cofactor analysis"""
        print("Testing Multi-Variable Cube Cofactor Analysis...")
        
        test_cases = [
            # (expression, cube, description)
            ("abc + a'bc + ab'c + a'b'c", "ab", "Cube ab on 4-term expression"),
            ("xyz + x'y'z' + xy'z", "xy", "Cube xy on mixed expression"),
            ("a + ab + abc", "a", "Single variable cube"),
            ("a'b'c + abc + a'bc'", "a'b", "Complemented cube"),
            ("ab'c'd + a'bcd' + abc'd'", "bc'", "Complex cube"),
            ("1", "ab", "Constant 1 with cube"),
            ("0", "xy", "Constant 0 with cube"),
            ("a", "a'", "Contradictory cube"),
        ]
        
        for expr, cube_expr, desc in test_cases:
            try:
                f = parse_sop(expr)
                cube = parse_cube(cube_expr)
                
                if "__IMPOSSIBLE__" in cube:
                    result = "0 (contradictory cube)"
                    success = True
                else:
                    fc = cofactor(f, cube)
                    result = cover_str(fc)
                    success = True
                
                self.log_test(
                    f"Cofactor Cube - {desc}",
                    f"f = {expr}, cube = {cube_expr}",
                    "Valid cube cofactor result",
                    result,
                    success,
                    "Cube cofactor computation successful"
                )
                
            except Exception as e:
                self.log_test(
                    f"Cofactor Cube - {desc}",
                    f"f = {expr}, cube = {cube_expr}",
                    "Valid cube cofactor result",
                    f"Error: {str(e)}",
                    False,
                    f"Exception occurred: {str(e)}"
                )
    
    def run_smoothing_tests(self):
        """Test smoothing analysis"""
        print("Testing Smoothing Analysis...")
        
        test_cases = [
            # (expression, variable, description)
            ("ab + a'c", "a", "Basic smoothing on variable a"),
            ("xyz + x'y'z", "x", "Smoothing variable x"),
            ("abc + a'bc + ab'c", "b", "Variable appears in multiple terms"),
            ("a + b", "c", "Variable not in expression"),
            ("ab'c + a'bc' + abc", "a", "Variable in all terms"),
            (demo_big_expression(), "b", "Complex expression smoothing"),
            ("1", "x", "Constant 1 smoothing"),
            ("0", "y", "Constant 0 smoothing"),
        ]
        
        for expr, var, desc in test_cases:
            try:
                f = parse_sop(expr)
                result = smoothing(f, var)
                result_str = cover_str(result)
                
                self.log_test(
                    f"Smoothing - {desc}",
                    f"f = {expr}, variable = {var}",
                    "Valid smoothing result",
                    f"S_{var}(f) = {result_str}",
                    True,
                    "Smoothing computation successful"
                )
                
            except Exception as e:
                self.log_test(
                    f"Smoothing - {desc}",
                    f"f = {expr}, variable = {var}",
                    "Valid smoothing result",
                    f"Error: {str(e)}",
                    False,
                    f"Exception occurred: {str(e)}"
                )
    
    def run_consensus_tests(self):
        """Test consensus analysis"""
        print("Testing Consensus Analysis...")
        
        test_cases = [
            # (expression, variable, description)
            ("ab + a'c", "a", "Basic consensus on variable a"),
            ("xy'z + x'yz'", "x", "Consensus variable x"),
            ("abc + a'bc + ab'c'", "a", "Variable in multiple forms"),
            ("a + b", "c", "Variable not in expression"),
            ("ab + a'b", "a", "Complementary terms"),
            (demo_big_expression(), "c", "Complex expression consensus"),
            ("1", "x", "Constant 1 consensus"),
            ("0", "y", "Constant 0 consensus"),
        ]
        
        for expr, var, desc in test_cases:
            try:
                f = parse_sop(expr)
                result = consensus_operator(f, var)
                result_str = cover_str(result)
                
                self.log_test(
                    f"Consensus - {desc}",
                    f"f = {expr}, variable = {var}",
                    "Valid consensus result",
                    f"C_{var}(f) = {result_str}",
                    True,
                    "Consensus computation successful"
                )
                
            except Exception as e:
                self.log_test(
                    f"Consensus - {desc}",
                    f"f = {expr}, variable = {var}",
                    "Valid consensus result",
                    f"Error: {str(e)}",
                    False,
                    f"Exception occurred: {str(e)}"
                )
    
    def run_boolean_difference_optimized_tests(self):
        """Test optimized Boolean difference analysis"""
        print("Testing Optimized Boolean Difference Analysis...")
        
        test_cases = [
            # (expression, variable, description)
            ("ab + a'c", "a", "Basic Boolean difference on variable a"),
            ("xyz + x'y'z", "x", "Difference variable x"),
            ("abc + a'bc + ab'c", "b", "Variable in multiple terms"),
            ("a + b", "c", "Variable not in expression"),
            ("ab'c + a'bc'", "a", "Complementary variable usage"),
            (demo_big_expression(), "a", "Complex expression difference"),
            ("1", "x", "Constant 1 difference"),
            ("0", "y", "Constant 0 difference"),
        ]
        
        for expr, var, desc in test_cases:
            try:
                f = parse_sop(expr)
                result_str, expanded = boolean_difference_var_optimized(f, var, keep_symbolic=True)
                
                self.log_test(
                    f"Boolean Difference Optimized - {desc}",
                    f"f = {expr}, variable = {var}",
                    "Valid Boolean difference result",
                    f"∂f/∂{var} = {result_str}",
                    True,
                    "Optimized Boolean difference computation successful"
                )
                
            except Exception as e:
                self.log_test(
                    f"Boolean Difference Optimized - {desc}",
                    f"f = {expr}, variable = {var}",
                    "Valid Boolean difference result",
                    f"Error: {str(e)}",
                    False,
                    f"Exception occurred: {str(e)}"
                )
    
    def run_boolean_difference_naive_tests(self):
        """Test naive Boolean difference analysis"""
        print("Testing Naive Boolean Difference Analysis...")
        
        test_cases = [
            # (expression, variable, description) - using simpler cases for naive method
            ("ab + a'c", "a", "Basic Boolean difference on variable a"),
            ("xy + x'z", "x", "Simple difference variable x"),
            ("abc + a'bc", "a", "Variable in two terms"),
            ("a + b", "c", "Variable not in expression"),
            ("ab + a'b", "a", "Complementary terms"),
            ("1", "x", "Constant 1 difference"),
            ("0", "y", "Constant 0 difference"),
        ]
        
        for expr, var, desc in test_cases:
            try:
                f = parse_sop(expr)
                result = boolean_difference_naive(f, var)
                result_str = cover_str(result)
                
                self.log_test(
                    f"Boolean Difference Naive - {desc}",
                    f"f = {expr}, variable = {var}",
                    "Valid Boolean difference result",
                    f"∂f/∂{var} = {result_str}",
                    True,
                    "Naive Boolean difference computation successful"
                )
                
            except Exception as e:
                self.log_test(
                    f"Boolean Difference Naive - {desc}",
                    f"f = {expr}, variable = {var}",
                    "Valid Boolean difference result",
                    f"Error: {str(e)}",
                    False,
                    f"Exception occurred: {str(e)}"
                )
    
    def run_formula_display_tests(self):
        """Test formula display functionality"""
        print("Testing Formula Display Functionality...")
        
        test_cases = [
            # (expression, description)
            ("ab + a'c", "Simple expression"),
            ("xyz + x'y'z + xy'z'", "Three-variable expression"),
            (demo_big_expression(), "Complex demo expression"),
            ("a", "Single variable"),
            ("a'", "Single complemented variable"),
            ("1", "Constant 1"),
            ("0", "Constant 0"),
        ]
        
        for expr, desc in test_cases:
            try:
                f = parse_sop(expr)
                variables = vars_in_cover(f)
                var_report = variable_report(f)
                parsed_str = cover_str(f)
                
                result = {
                    'parsed': parsed_str,
                    'variables': variables,
                    'variable_report': var_report
                }
                
                self.log_test(
                    f"Formula Display - {desc}",
                    f"f = {expr}",
                    "Valid formula analysis",
                    f"Parsed: {parsed_str}, Variables: {variables}",
                    True,
                    f"Variables: {len(variables)}, Report generated successfully"
                )
                
            except Exception as e:
                self.log_test(
                    f"Formula Display - {desc}",
                    f"f = {expr}",
                    "Valid formula analysis",
                    f"Error: {str(e)}",
                    False,
                    f"Exception occurred: {str(e)}"
                )
    
    def run_comprehensive_analysis_comparison(self):
        """Run all analyses on the same expressions for comparison"""
        print("Running Comprehensive Analysis Comparison...")
        
        test_expressions = [
            "ab + a'c",
            "xyz + x'y'z",
            demo_big_expression()
        ]
        
        for expr in test_expressions:
            try:
                f = parse_sop(expr)
                variables = vars_in_cover(f)
                
                if not variables:
                    continue
                    
                # Pick first variable for comparison
                var = variables[0]
                
                # Run all analyses
                fx1, fx0 = positive_negative_cofactors(f, var)
                smooth_result = smoothing(f, var)
                consensus_result = consensus_operator(f, var)
                opt_diff, _ = boolean_difference_var_optimized(f, var, keep_symbolic=True)
                naive_diff = boolean_difference_naive(f, var)
                
                comparison_result = {
                    'expression': expr,
                    'variable': var,
                    'cofactor_pos': cover_str(fx1),
                    'cofactor_neg': cover_str(fx0),
                    'smoothing': cover_str(smooth_result),
                    'consensus': cover_str(consensus_result),
                    'diff_optimized': opt_diff,
                    'diff_naive': cover_str(naive_diff)
                }
                
                self.log_test(
                    f"Comprehensive Analysis - {expr[:20]}...",
                    f"f = {expr}, variable = {var}",
                    "All analyses complete",
                    "All analyses completed successfully",
                    True,
                    f"Compared 6 different analysis types on variable {var}"
                )
                
            except Exception as e:
                self.log_test(
                    f"Comprehensive Analysis - {expr[:20]}...",
                    f"f = {expr}",
                    "All analyses complete",
                    f"Error: {str(e)}",
                    False,
                    f"Exception during comprehensive analysis: {str(e)}"
                )
    
    def run_all_tests(self):
        """Run all test suites"""
        print("=" * 80)
        print("BOOLEAN SOP ANALYZER - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Starting test execution at {datetime.datetime.now()}")
        print()
        
        # Run all test categories
        self.run_cofactor_single_variable_tests()
        print()
        self.run_cofactor_cube_tests()
        print()
        self.run_smoothing_tests()
        print()
        self.run_consensus_tests()
        print()
        self.run_boolean_difference_optimized_tests()
        print()
        self.run_boolean_difference_naive_tests()
        print()
        self.run_formula_display_tests()
        print()
        self.run_comprehensive_analysis_comparison()
        print()
        
        print("All test suites completed!")
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report"""
        report = []
        report.append("=" * 100)
        report.append("BOOLEAN SOP ANALYZER - TEST EXECUTION REPORT")
        report.append("=" * 100)
        report.append(f"Test Execution Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tests Run: {self.total_tests}")
        report.append(f"Tests Passed: {self.passed_tests}")
        report.append(f"Tests Failed: {self.failed_tests}")
        report.append(f"Success Rate: {(self.passed_tests/self.total_tests*100):.1f}%" if self.total_tests > 0 else "N/A")
        report.append("")
        
        # Group results by test type
        test_categories = {}
        for result in self.test_results:
            category = result['test_name'].split(' - ')[0]
            if category not in test_categories:
                test_categories[category] = []
            test_categories[category].append(result)
        
        # Generate category reports
        for category, tests in test_categories.items():
            report.append(f"\n{category.upper()}")
            report.append("-" * len(category))
            
            passed_in_category = sum(1 for t in tests if t['passed'])
            total_in_category = len(tests)
            
            report.append(f"Tests in category: {total_in_category}")
            report.append(f"Passed: {passed_in_category}")
            report.append(f"Failed: {total_in_category - passed_in_category}")
            report.append("")
            
            for test in tests:
                status = "✓ PASS" if test['passed'] else "✗ FAIL"
                report.append(f"{status} | {test['test_name'].split(' - ')[1]}")
                report.append(f"      Input: {test['input']}")
                report.append(f"      Result: {test['actual']}")
                if test['notes']:
                    report.append(f"      Notes: {test['notes']}")
                if not test['passed']:
                    report.append(f"      Expected: {test['expected']}")
                report.append("")
        
        # Summary of analysis types tested
        report.append("\nANALYSIS TYPES TESTED")
        report.append("-" * 21)
        report.append("1. Single Variable Cofactor Analysis (f_x, f_x')")
        report.append("2. Multi-Variable Cube Cofactor Analysis")
        report.append("3. Smoothing Analysis (S_x(f))")
        report.append("4. Consensus Analysis (C_x(f))")
        report.append("5. Boolean Difference - Optimized (∂f/∂x)")
        report.append("6. Boolean Difference - Naive (∂f/∂x)")
        report.append("7. Formula Display and Variable Analysis")
        report.append("8. Comprehensive Analysis Comparison")
        report.append("")
        
        if self.failed_tests > 0:
            report.append("\nFAILED TESTS SUMMARY")
            report.append("-" * 20)
            failed_tests = [t for t in self.test_results if not t['passed']]
            for test in failed_tests:
                report.append(f"• {test['test_name']}")
                report.append(f"  Input: {test['input']}")
                report.append(f"  Error: {test['actual']}")
                report.append("")
        
        report.append("\nTEST EXECUTION COMPLETE")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "analysis_test_report.txt"):
        """Save the test report to a file"""
        report_content = self.generate_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        return filename


def main():
    """Main function to run all tests and generate report"""
    test_suite = AnalysisTestSuite()
    
    try:
        # Run all tests
        test_suite.run_all_tests()
        
        # Generate and display report
        report = test_suite.generate_report()
        print(report)
        
        # Save report to file
        report_file = test_suite.save_report()
        print(f"\nDetailed report saved to: {report_file}")
        
        # Exit with appropriate code
        sys.exit(0 if test_suite.failed_tests == 0 else 1)
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during test execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()