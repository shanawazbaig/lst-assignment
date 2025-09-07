#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis Runner - Run all analyses for Boolean SoP expressions
=============================================================
This script provides a non-interactive way to run all available analyses
on Boolean Sum-of-Products expressions and generate a comprehensive report.

Usage:
    python3 run_analyses.py [expression]
    
If no expression is provided, uses the demo expression.
"""

import sys
import datetime
from main import (
    parse_sop, positive_negative_cofactors, smoothing, consensus_operator,
    boolean_difference_naive, boolean_difference_var_optimized, 
    cover_str, vars_in_cover, variable_report, demo_big_expression,
    cofactor, parse_cube, DIFFERENCE_KEEP_SYMBOLIC_XOR
)

def run_all_analyses(expression: str) -> dict:
    """Run all available analyses on the given expression"""
    
    print(f"Analyzing Boolean SoP Expression: {expression}")
    print("=" * 80)
    
    # Parse the expression
    try:
        f = parse_sop(expression)
        parsed_expr = cover_str(f)
        variables = vars_in_cover(f)
        var_report = variable_report(f)
        
        print(f"Parsed Expression: {parsed_expr}")
        print(f"Variables: {', '.join(variables) if variables else '(none)'}")
        print(f"Number of variables: {len(variables)}")
        print(f"Number of terms: {len(f)}")
        print()
        
        print("Variable Analysis Report:")
        print(var_report)
        print()
        
    except Exception as e:
        print(f"Error parsing expression: {e}")
        return {}
    
    if not variables:
        print("No variables found in expression - cannot perform variable-based analyses")
        return {}
    
    results = {
        'expression': expression,
        'parsed': parsed_expr,
        'variables': variables,
        'analyses': {}
    }
    
    # Run analyses for each variable
    for var in variables:
        print(f"ANALYSES FOR VARIABLE '{var}'")
        print("-" * 40)
        
        var_results = {}
        
        try:
            # 1. Cofactor Analysis
            fx1, fx0 = positive_negative_cofactors(f, var)
            fx1_str = cover_str(fx1)
            fx0_str = cover_str(fx0)
            var_results['cofactor'] = {
                'positive': fx1_str,
                'negative': fx0_str
            }
            print(f"1. Cofactor Analysis:")
            print(f"   f_{var}  = {fx1_str}")
            print(f"   f_{var}' = {fx0_str}")
            
            # 2. Smoothing
            smooth_result = smoothing(f, var)
            smooth_str = cover_str(smooth_result)
            var_results['smoothing'] = smooth_str
            print(f"2. Smoothing S_{var}(f) = {smooth_str}")
            
            # 3. Consensus
            consensus_result = consensus_operator(f, var)
            consensus_str = cover_str(consensus_result)
            var_results['consensus'] = consensus_str
            print(f"3. Consensus C_{var}(f) = {consensus_str}")
            
            # 4. Boolean Difference (Optimized)
            diff_opt_str, _ = boolean_difference_var_optimized(f, var, keep_symbolic=DIFFERENCE_KEEP_SYMBOLIC_XOR)
            var_results['boolean_diff_optimized'] = diff_opt_str
            print(f"4. Boolean Difference (Optimized) ∂f/∂{var} = {diff_opt_str}")
            
            # 5. Boolean Difference (Naive)
            try:
                diff_naive_result = boolean_difference_naive(f, var)
                diff_naive_str = cover_str(diff_naive_result)
                var_results['boolean_diff_naive'] = diff_naive_str
                print(f"5. Boolean Difference (Naive) ∂f/∂{var} = {diff_naive_str}")
            except Exception as e:
                var_results['boolean_diff_naive'] = f"Error: {str(e)}"
                print(f"5. Boolean Difference (Naive) ∂f/∂{var} = Error: {str(e)}")
            
            results['analyses'][var] = var_results
            print()
            
        except Exception as e:
            print(f"Error analyzing variable {var}: {e}")
            results['analyses'][var] = {'error': str(e)}
            print()
    
    # Cube cofactor examples (if we have variables)
    if len(variables) >= 2:
        print("CUBE COFACTOR EXAMPLES")
        print("-" * 30)
        
        # Test some simple cube combinations
        cube_examples = []
        if len(variables) >= 2:
            cube_examples.append(variables[0] + variables[1])  # ab
            cube_examples.append(variables[0] + "'" + variables[1])  # a'b
        if len(variables) >= 3:
            cube_examples.append(variables[0] + variables[1] + "'" + variables[2])  # ab'c
        
        cube_results = {}
        for cube_expr in cube_examples:
            try:
                cube = parse_cube(cube_expr)
                if "__IMPOSSIBLE__" in cube:
                    result_str = "0 (contradictory cube)"
                else:
                    cube_result = cofactor(f, cube)
                    result_str = cover_str(cube_result)
                
                cube_results[cube_expr] = result_str
                print(f"f_{{{cube_expr}}} = {result_str}")
                
            except Exception as e:
                cube_results[cube_expr] = f"Error: {str(e)}"
                print(f"f_{{{cube_expr}}} = Error: {str(e)}")
        
        results['cube_cofactors'] = cube_results
        print()
    
    return results

def generate_summary_report(results: dict) -> str:
    """Generate a summary report of all analyses"""
    
    if not results:
        return "No results to report"
    
    report = []
    report.append("BOOLEAN SOP ANALYSIS SUMMARY REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append(f"Expression: {results['expression']}")
    report.append(f"Parsed: {results['parsed']}")
    report.append(f"Variables: {', '.join(results['variables']) if results['variables'] else '(none)'}")
    report.append("")
    
    if 'analyses' in results:
        for var, analysis in results['analyses'].items():
            report.append(f"VARIABLE '{var}' ANALYSIS")
            report.append("-" * 25)
            
            if 'error' in analysis:
                report.append(f"Error: {analysis['error']}")
            else:
                if 'cofactor' in analysis:
                    report.append(f"Cofactor +: {analysis['cofactor']['positive']}")
                    report.append(f"Cofactor -: {analysis['cofactor']['negative']}")
                if 'smoothing' in analysis:
                    report.append(f"Smoothing: {analysis['smoothing']}")
                if 'consensus' in analysis:
                    report.append(f"Consensus: {analysis['consensus']}")
                if 'boolean_diff_optimized' in analysis:
                    report.append(f"Bool Diff (Opt): {analysis['boolean_diff_optimized']}")
                if 'boolean_diff_naive' in analysis:
                    report.append(f"Bool Diff (Naive): {analysis['boolean_diff_naive']}")
            
            report.append("")
    
    if 'cube_cofactors' in results:
        report.append("CUBE COFACTOR EXAMPLES")
        report.append("-" * 25)
        for cube, result in results['cube_cofactors'].items():
            report.append(f"f_{{{cube}}} = {result}")
        report.append("")
    
    report.append("Analysis Complete")
    report.append("=" * 50)
    
    return "\n".join(report)

def main():
    """Main function"""
    
    # Get expression from command line or use demo
    if len(sys.argv) > 1:
        expression = sys.argv[1]
    else:
        expression = demo_big_expression()
        print("No expression provided, using demo expression:")
        print(f"Demo: {expression}")
        print()
    
    # Run all analyses
    try:
        results = run_all_analyses(expression)
        
        # Generate summary report
        summary = generate_summary_report(results)
        print("\n" + "="*80)
        print(summary)
        
        # Save summary to file
        report_filename = f"analysis_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\nSummary report saved to: {report_filename}")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()