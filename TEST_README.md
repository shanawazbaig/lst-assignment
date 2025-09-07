# Boolean SoP Analyzer - Testing and Analysis Documentation

This repository contains a Boolean Sum-of-Products (SoP) Analyzer with comprehensive testing and analysis capabilities.

## Available Analysis Types

The analyzer supports 7 different types of Boolean function analysis:

1. **Single Variable Cofactor** - Computes f_x and f_x' (positive and negative cofactors)
2. **Multi-Variable Cube Cofactor** - Computes f_cube for multi-variable cubes
3. **Smoothing** - Computes S_x(f) = f_x + f_x'
4. **Consensus** - Computes C_x(f) = f_x · f_x'
5. **Boolean Difference (Optimized)** - Computes ∂f/∂x using optimized algorithm with symbolic XOR
6. **Boolean Difference (Naive)** - Computes ∂f/∂x using standard XOR expansion
7. **Formula Display and Variable Analysis** - Shows parsed form and variable statistics

## Usage

### Interactive Mode
Run the main analyzer in interactive mode:
```bash
python3 main.py
```

### Automated Analysis Runner
Run all analyses on an expression automatically:
```bash
# Use demo expression
python3 run_analyses.py

# Analyze custom expression
python3 run_analyses.py "ab + a'c"
```

### Comprehensive Test Suite
Run the complete test suite that validates all analysis types:
```bash
python3 test_analysis.py
```

## Files

- **`main.py`** - Main Boolean SoP analyzer with interactive CLI
- **`test_analysis.py`** - Comprehensive test suite for all analysis types
- **`run_analyses.py`** - Non-interactive runner for all analyses
- **`analysis_test_report.txt`** - Detailed test execution report
- **`analysis_summary_*.txt`** - Generated analysis summary reports

## Test Coverage

The test suite includes:

- **58 total test cases** covering all analysis types
- **9 single variable cofactor tests** - various expressions and edge cases
- **8 multi-variable cube tests** - including contradictory cubes
- **8 smoothing tests** - different variable scenarios
- **8 consensus tests** - complementary terms and edge cases
- **8 optimized Boolean difference tests** - symbolic XOR handling
- **7 naive Boolean difference tests** - full expansion method
- **7 formula display tests** - parsing and variable analysis
- **3 comprehensive comparison tests** - all analyses on same expressions

## Example Results

For expression `ab + a'c` with variable `a`:

- **Cofactor**: f_a = b, f_a' = c
- **Smoothing**: S_a(f) = b + c
- **Consensus**: C_a(f) = bc
- **Boolean Difference**: ∂f/∂a = (b) ⊕ (c) = b'c + bc'

## Test Results Summary

Latest test execution:
- **Total Tests**: 58
- **Passed**: 58 (100%)
- **Failed**: 0
- **Success Rate**: 100.0%

All analysis types are working correctly and validated with comprehensive test cases covering various scenarios including:
- Simple and complex expressions
- Edge cases (constants 0 and 1)
- Contradictory inputs
- Multi-variable scenarios
- Demo expression testing

## Features

- **Comprehensive Coverage**: Tests every available analysis type
- **Edge Case Handling**: Validates behavior with constants, contradictions, and missing variables
- **Automated Reporting**: Generates detailed reports with pass/fail status
- **Multiple Input Methods**: Interactive CLI, automated runner, and test framework
- **Error Handling**: Graceful handling of invalid inputs and edge cases
- **Performance**: Optimized Boolean difference algorithm avoids combinational explosion