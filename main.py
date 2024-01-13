import numpy as np

domain_left = 0
domain_right = 2
domain_length = domain_right - domain_left
precision = 10
h = domain_length / precision
h_inv = 1 / h


def integrate_trapezoidal(function, lower_bound, upper_bound, iterations):
    """
Function performs numerical integration using trapezoidal rule.
    :param function: mathematical function to integrate
    :param lower_bound: lower bound of integration
    :param upper_bound: upper bound of integration
    :param iterations: number of iterations to perform over specified interval
    :return: numerical value of integration
    """
    step = (upper_bound - lower_bound) / iterations
    integral_value = 0
    for i in np.arange(lower_bound, upper_bound, step):
        a = function(i)
        b = function(i + step)
        integral_value += (a + b) * step / 2
    return integral_value


def E(x):
    return 3 if 0 <= x <= 1 else 5 if 1 < x <= 2 else None


def basis_function_value(i, x):
    """
    :param i: number of finite element
    :param x: point on x-axis
    :return: value of the basis function associated with i-th finite element at point x
    """
    center = domain_length * i / precision + domain_left
    left = center - h
    right = center + h

    if x < left or x > right:
        return 0.0
    if x <= center:
        return (x - left) * h_inv
    return (right - x) * h_inv


def basis_function_derivative(i, x):
    """
    :param i: number of finite element
    :param x: point on x-axis
    :return: derivative value of the basis function associated with i-th finite element at point x
    """
    center = domain_length * i / precision + domain_left
    left = center - h
    right = center + h

    if x < left or x > right:
        return 0.0
    if x <= center:
        return h_inv
    return -h_inv


if __name__ == "__main__":
    for v in np.linspace(domain_left, domain_right, 100):
        print(basis_function_value(1, v), basis_function_derivative(1, v))
