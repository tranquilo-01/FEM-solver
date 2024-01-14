import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

domain_left = 0
domain_right = 2
domain_length = domain_right - domain_left
# some combinations of fe_number and integration_step give weird results
# mostly occurs when fe_number is big and integration step is also big
# number of finite elements, must be greater than 2
fe_number = 32
integration_step = 1 / 1000
fe_step = domain_length / fe_number
fe_step_inv = 1 / fe_step


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
    return integral_value


def E(x):
    """
    elasticity function given in the assignment
    :param x: x value
    :return:  E(x)
    """
    if 0 <= x <= 1:
        return 3
    elif 1 < x <= 2:
        return 5


def basis_function_value(i, x):
    """
    :param i: number of finite element
    :param x: point on x-axis
    :return: value of the basis function associated with i-th finite element at point x
    """
    center = domain_length * i / fe_number + domain_left
    left = center - fe_step
    right = center + fe_step

    if x < left or x > right:
        return 0.0
    if x <= center:
        return (x - left) * fe_step_inv
    return (right - x) * fe_step_inv


def basis_function_derivative(i, x):
    """
    :param i: number of finite element
    :param x: point on x-axis
    :return: derivative value of the basis function associated with i-th finite element at point x
    """
    center = domain_length * i / fe_number + domain_left
    left = center - fe_step
    right = center + fe_step

    if x < left or x > right:
        return 0.0
    if x <= center:
        return fe_step_inv
    return -fe_step_inv


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
    """
    function builds a linear equation system connected to elastic deformation FEM problem
    given in the assignment and solves it
    :return: vector of finite elements coefficients
    """
    b_matrix = np.zeros((fe_number, fe_number))

    for i in range(fe_number):
        for j in range(fe_number):
            if abs(j - i) <= 1:
                integrate_from = domain_length * max(max(i, j) - 1, 0) / fe_number
                integrate_to = domain_length * min(min(i, j) + 1, fe_number) / fe_number

                def integrand(x): return E(x) * basis_function_derivative(i, x) * basis_function_derivative(j, x)

                # integrate E(x)du/dx*dv/dx dx -3u(0)v(0)
                b_matrix[i, j] = integrate.quad(integrand, integrate_from, integrate_to)[0] - 3 * basis_function_value(i, 0) * basis_function_value(j, 0)
                # b_matrix[i, j] = integrate_trapezoidal(integrand, integrate_from, integrate_to) - 3 * basis_function_value(i, 0) * basis_function_value(j, 0)

    l_vector = np.zeros(fe_number)
    # -30v(0)
    l_vector[0] = -30 * basis_function_value(0, 0)

    # coefficients = np.linalg.solve(b_matrix, l_vector)
    coefficients = gaussian_elimination(b_matrix, l_vector)

    # u(2) = 0
    result = np.concatenate((coefficients, [0]))
    return result


if __name__ == "__main__":
    plot(solve())
