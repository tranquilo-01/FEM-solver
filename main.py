import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

domain_left = 0
domain_right = 2
domain_length = domain_right - domain_left
# FIXME some combinations of fe_number and integration_step give weird results
# mostly occurs when fe_number is big and integration step is also big
# number of finite elements, must be greater than 2
fe_number = 32
integration_step = 1 / 1000
asc_slope_basis = domain_length / fe_number
desc_slope_basis = 1 / asc_slope_basis


def integrate_trapezoidal(f, lower_bound, upper_bound):
    """
Function performs numerical integration using trapezoidal rule.
    :param f: mathematical function to integrate
    :param lower_bound: lower bound of integration
    :param upper_bound: upper bound of integration
    :return: numerical value of integration
    """
    integral_value = 0
    i = lower_bound
    print(lower_bound, upper_bound, integration_step)
    while i < upper_bound:
        a = f(i)
        if i + integration_step > upper_bound:
            b = f(upper_bound)
            integral_value += (a + b) * (upper_bound - i) / 2
        else:
            b = f(i + integration_step)
            integral_value += (a + b) * integration_step / 2
        i += integration_step

    # scipy_value = integrate.quad(f, lower_bound, upper_bound)

    # print("my value: ", integral_value, "scipy value: ", scipy_value, "diff: ", abs(integral_value - scipy_value[0]))
    return integral_value


def E(x):
    if 0 <= x <= 1:
        return 3
    elif 1 < x <= 2:
        return 5
    else:
        pass


def basis_function_value(i, x):
    """
    :param i: number of finite element
    :param x: point on x-axis
    :return: value of the basis function associated with i-th finite element at point x
    """
    center = domain_length * i / fe_number + domain_left
    left = center - asc_slope_basis
    right = center + asc_slope_basis

    if x < left or x > right:
        return 0.0
    if x <= center:
        return (x - left) * desc_slope_basis
    return (right - x) * desc_slope_basis


def basis_function_derivative(i, x):
    """
    :param i: number of finite element
    :param x: point on x-axis
    :return: derivative value of the basis function associated with i-th finite element at point x
    """
    center = domain_length * i / fe_number + domain_left
    left = center - asc_slope_basis
    right = center + asc_slope_basis

    if x < left or x > right:
        return 0.0
    if x <= center:
        return desc_slope_basis
    return -desc_slope_basis


def plot(result):
    x = np.linspace(domain_left, domain_right, fe_number + 1)
    plt.plot(x, result)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Odkształcenie sprężyste")
    info_box = f"Liczba elementów skończonych: {fe_number}\nKrok całkowania: {integration_step}"
    plt.annotate(info_box, xy=(0.58, 0.91), xycoords='axes fraction', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    plt.grid(True)
    plt.yticks(np.arange(result.min(), result.max()+1, 2))
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

                def integrand(x): return E(x) * basis_function_derivative(n, x) * basis_function_derivative(m, x)

                integral = integrate_trapezoidal(integrand, integrate_from, integrate_to)
                # integral, _ = integrate.quad(integrand, integrate_from, integrate_to)

            b_matrix[n, m] = -E(0) * basis_function_value(n, 0) * basis_function_value(m, 0) + integral

    # Build L vector
    l_vector = np.zeros(fe_number)
    l_vector[0] = -10 * E(0) * basis_function_value(0, 0)

    # print(b_matrix)
    # print(l_vector)

    # Calculate coefficients
    # coefficients = np.linalg.solve(b_matrix, l_vector)
    coefficients = gaussian_elimination(b_matrix, l_vector)

    result = np.concatenate((coefficients, [0]))
    # print(result)
    return result


def gaussian_elimination(matrix, vector):
    """
    Functions solves a linear matrix equation, optimized for FEM.
    :param matrix: coefficient matrix
    :param vector: dependent variable vector
    :return: solution to matrix*x=vector
    """
    n = matrix.shape[0]
    for row in range(n - 1):
        col = row
        ratio = matrix[row + 1, col] / matrix[row, row]
        for k in range(n):
            matrix[row + 1][k] = matrix[row + 1][k] - ratio * matrix[row][k]
        vector[row + 1] = vector[row + 1] - ratio * vector[row]

    res = np.zeros(n)
    res[n - 1] = vector[n - 1] / matrix[n - 1, n - 1]
    for m in range(n - 2, -1, -1):
        ratio = matrix[m, m + 1] / matrix[m + 1, m + 1]
        matrix[m, m + 1] = matrix[m, m + 1] - ratio * matrix[m + 1, m + 1]
        vector[m] = vector[m] - ratio * vector[m + 1]
        res[m] = vector[m] / matrix[m, m]
    return res


def test_integration():
    print("x^2")
    integrate_trapezoidal(lambda x: x * x, -1.2137, 2)
    print("3x^3 + 5x^2 + 2x - 8")
    integrate_trapezoidal(lambda x: 3 * x ** 3 + 5 * x ** 2 + 2 * x - 8, -1.2137, 2)
    print("50x")
    integrate_trapezoidal(lambda x: 50 * x, -1.2137, 2)
    print("500")
    integrate_trapezoidal(lambda x: 500, -1.2137, 2)


if __name__ == "__main__":
    plot(solve())
