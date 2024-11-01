# Mean-Variance Optimization in Java

This project demonstrates a Java implementation of Mean-Variance Optimization (MVO), a fundamental approach to modern portfolio theory. The Mean-Variance Optimization model seeks to minimize the risk for a given return by adjusting asset weights.

## Overview

Mean-Variance Optimization (MVO), based on Harry Markowitz's modern portfolio theory, is widely used in finance for balancing risk and return in investment portfolios. By adjusting the weights of each asset, MVO allows us to calculate the optimal portfolio allocation for a given risk level.

This code performs the following tasks:
1. **Reads multiple CSV files** containing data of asset prices over time.
2. **Performs linear regression** to calculate expected returns for each asset.
3. **Calculates variances and covariances** for each asset.
4. **Solves the optimization problem** to find asset weights that minimize risk for a targeted return level.

> **Note**: The Monte Carlo simulation is a separate project from the core Mean-Variance Optimization code.

## Contributors:

* Arnav Malhotra
* Emily Wang
