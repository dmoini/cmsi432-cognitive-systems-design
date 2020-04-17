# Serena Zafiris, Donovan Moini, Lucille Njoo

import pandas as pd


def linear_scm():
    df = pd.read_csv("../data/reg_data.csv")

    mean = df.mean()
    variance = df.var()
    covariance = df.cov()

    # Results for 6.1
    print("PART 1")
    print(f'Expected Values:\n{mean}\n')
    print(f'Variance:\n{variance}\n')
    print(f'Covariance:\n{covariance}\n')

    x_var, y_var, z_var = df.X.var(), df.Y.var(), df.Z.var()
    xy_cov, xz_cov, yz_cov = df.X.cov(df.Y), df.X.cov(df.Z), df.Y.cov(df.Z)
    x_mean, y_mean, z_mean = df.X.mean(), df.Y.mean(), df.Z.mean(0)

    xy_regression_beta = xy_cov / x_var
    xy_regression_alpha = x_mean - (y_mean * xy_regression_beta)
    
    xz_regression_beta = xz_cov / x_var
    xz_regression_alpha = x_mean - (z_mean * xz_regression_beta)

    # Results for 6.2
    print("PART 2")
    print("Linear regression formula for X ---> Y")
    print(f'Y = {xy_regression_alpha} + ({xy_regression_beta} * X)\n')

    print("Linear regression formula for X ---> Z")
    print(f'Z = {xz_regression_alpha} + ({xz_regression_beta} * X)\n')

    df['Ux'] = df.X
    df["Uy"] = df.Y - xy_regression_alpha - (xy_regression_beta * df.X)
    df["Uz"] = df.Z - xz_regression_alpha - (xz_regression_beta * df.X)

    new_df = df.drop(["X", "Y", "Z"], axis=1)

    # Result for 6.3
    print("PART 3")
    print(new_df)

    new_covariance = new_df.cov()

    # Result for 6.4: Unobserved confounder between Y and Z
    print("PART 4")
    print(f'New Covariance:\n{new_covariance}\n')

    # Givens: X = 1, Z = 3
    computed_Uz = 3 - xz_regression_alpha - (xz_regression_beta * 1)
    # Counterfactual in world where X = 2
    counterfactual_z = xz_regression_alpha + (xz_regression_beta * 2) + computed_Uz

    # Result for 6.5
    print("PART 5")
    print(f'E[Z_X=2|X=1,Z=3]:\n{counterfactual_z}\n')

linear_scm()
