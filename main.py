import matplotlib.pyplot as plt
import numpy as np

domain_left = 0
domain_right = 2
domain_length = domain_right - domain_left
# number of finite elements, must be greater than 2
fe_number = 8
integration_step = 1/1000
h = domain_length / fe_number
h_inv = 1 / h


def integrate_trapezoidal(function, lower_bound, upper_bound, step):
    """
Function performs numerical integration using trapezoidal rule.
    :param function: mathematical function to integrate
    :param lower_bound: lower bound of integration
    :param upper_bound: upper bound of integration
    :param step: step size of iteration
    :return: numerical value of integration
    """
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
    center = domain_length * i / fe_number + domain_left
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
    center = domain_length * i / fe_number + domain_left
    left = center - h
    right = center + h

    if x < left or x > right:
        return 0.0
    if x <= center:
        return h_inv
    return -h_inv


def plot(result):
    x = np.linspace(domain_left, domain_right, fe_number+1)
    plt.plot(x, result)
    plt.show()


def solve():
    # Build B matrix
    b_matrix = np.zeros((fe_number, fe_number))

    for n in range(fe_number):
        for m in range(fe_number):
            integral = 0

            if abs(m - n) <= 1:
                integrate_from = domain_length * max(max(n, m) - 1, 0) / fe_number
                integrate_to = domain_length * min(min(n, m) + 1, fe_number) / fe_number

                integrand = lambda x: E(x) * basis_function_derivative(n, x) * basis_function_derivative(m, x)
                integral = integrate_trapezoidal(integrand, integrate_from, integrate_to, integration_step)

            b_matrix[n, m] = -E(0) * basis_function_value(n, 0) * basis_function_value(m, 0) + integral

    # Build L vector
    l_vector = np.zeros(fe_number)
    l_vector[0] = -10 * E(0) * basis_function_value(0, 0)

    # Calculate coefficients
    coefficients = np.linalg.solve(b_matrix, l_vector)

    result = np.concatenate((coefficients, [0]))
    return result


if __name__ == "__main__":
    plot(solve())
