import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

omega0 = [0.00696628, -0.03152652, -0.01437158, 0.02673555, 0.04050866, 0.01683239,
          -0.03949126, -0.01910853, 0.00495693, 0.02092892, -0.00496835, 0.00556729,
          -0.02365522, 0.01294543, 0.02428625, -0.00026533, -0.0335113, 0.03062016,
          -0.01458504, -0.00886471]

def main():
    data_x = np.loadtxt('./HW2-Data/X.csv', delimiter=",")    # (1000, 20)
    data_y = np.loadtxt('./HW2-Data/y.csv', delimiter=",")    # (1000,)
    np.random.seed(42)


    test_time = 100
    results = []
    lambs = []
    for i in range(test_time):
        lam = 0.0002 * i
        # Construct the problem.
        x = cp.Variable(data_x.shape[1])
        objective = cp.Minimize(cp.norm(data_x * x - data_y, 2) ** 2 + lam * cp.norm(x, 1))
        constraints = None
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        omega = x.value
        # res = np.linalg.norm(x=np.array(omega) - np.array(omega0), ord=2)
        res = len([ele for ele in omega if ele > (1 / 10**5)])
        results.append(res)
        lambs.append(lam)
    plt.plot(lambs, results)
    plt.xlabel('lambda')
    plt.ylabel('number of non-zero elements in the w∗')
    plt.show()
    print(lambs[results.index(min(results))])


if __name__ == '__main__':
    x1 = [-1.4061302594060445e-10, -4.65884647883577e-10, -1.656308226207918e-11, 1.0783624798375546e-09,
          0.010999979587224356, 5.443627077995256e-10, -2.4079179205561544e-10, -1.8217885680129265e-10,
          1.020317134648606e-10, 3.373589208991034e-10, 1.3363294577100737e-10, 1.6664435256944309e-10,
          -9.81998777663763e-11, 1.970396336241177e-10, 3.259584591576223e-10, -4.037271903035527e-11,
          -6.594393656999212e-10, -1.077694957500913e-10, -1.519109563741851e-09, -2.060329595971895e-09]
    x2 = [-2.491275254088687e-11, -1.1045703755170562e-10, -1.3075413915069995e-11, 1.7732233012580192e-10, 0.010955734757143745, 7.291304871759226e-11, -7.303131999374966e-11, -4.08660928720432e-11, 1.8747491645179315e-11, 7.3210736507542e-11, 2.7289142002056215e-11, 3.408178649590049e-11, -1.3934929176360982e-11, 4.4162058079090685e-11, 8.306344564321131e-11, -5.809687082843339e-13, -1.1984975941868357e-10, -4.3405526269429055e-12, -2.728128725700397e-10, -3.8536934588518413e-10]

    data_x = np.loadtxt('./HW2-Data/X.csv', delimiter=",")  # (1000, 20)
    data_y = np.loadtxt('./HW2-Data/y.csv', delimiter=",")  # (1000,)
    np.random.seed(42)

    lam = 0.1
    # Construct the problem.
    x = cp.Variable(data_x.shape[1])
    objective = cp.Minimize(cp.norm(data_x * x - data_y, 2) ** 2 + lam * cp.norm(x, 'inf'))
    constraints = None
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    omega = x.value
    print(list(omega))
