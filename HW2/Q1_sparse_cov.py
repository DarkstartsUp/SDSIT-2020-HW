import sklearn
import numpy as np
from scipy import linalg
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf, GraphicalLasso
import pandas as pd
from sklearn import preprocessing
import os.path as pt
import matplotlib.pyplot as plt


DATA_ROOT = '/home/luvision/Documents/GitHub/SDSIT-2020-HW/HW1/Data/INTLFXD_csv/data'

alpha = 0.00015783675646107936
alphas = [0.0024710117712324333, 0.0005323633479420285, 0.0003916289734717563,
          0.0002880988209564779, 0.00021193766615558776, 0.0001993156870229207,
          0.0001874454117290067, 0.00017628207244027057, 0.00016578356747810435,
          0.0001637601198889173, 0.00016176136919948, 0.00015978701397531995,
          0.00015783675646107936, 0.0001559103025356114, 0.00011469420645078192, 2.4710117712324306e-05, 0]


"""
The search for the optimal penalization parameter (alpha) is done on an iteratively refined grid: first the cross-validated scores on a grid are computed, then a new refined grid is centered around the maximum, and so on.
One of the challenges which is faced here is that the solvers can fail to converge to a well-conditioned estimate. The corresponding values of alpha then come out as missing values, but the optimum may be close to these missing values.
The GraphicalLasso estimator uses an l1 penalty to enforce sparsity on the precision matrix: the higher its alpha parameter, the more sparse the precision matrix. The corresponding GraphicalLassoCV object uses cross-validation to automatically set the alpha parameter.
"""


def main():
    # 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
    df = pd.read_csv(pt.join(DATA_ROOT, '1999_2018_complete.csv'))

    # drop malaysia and venezuela
    df.drop(['malaysia', 'venezuela'], axis=1, inplace=True)
    # print(df.columns[2:])

    data = df.values[:, 2:]
    data = preprocessing.scale(data, axis=1)
    print(data.shape)

    # plt.plot(data[:, 6], c='b', label='Japan')
    # plt.plot(data[:, 13], c='g', label='Sri Lanka')
    # plt.legend()
    # plt.show()

    # Estimate the covariance
    emp_cov = np.dot(data.T, data) / data.shape[0]

    for count in range(20):
        temp_data = data[count*242:(count+1)*242]

        # GraphicalLasso
        model = GraphicalLassoCV()
        model.fit(temp_data)
        cov_ = model.covariance_
        prec_ = model.precision_

        # print(model.alpha_)
        # print(model.cv_alphas_)
        # print(model.grid_scores_)
        # print(model.n_iter_)

        # Ledoit-Wolf
        lw_cov_, _ = ledoit_wolf(data)
        lw_prec_ = linalg.inv(lw_cov_)

        # #############################################################################
        # Plot the results
        # plt.figure(figsize=(8, 6))
        # plt.subplots_adjust(left=0.02, right=0.98)
        #
        # # plot the covariances
        # covs = [('Empirical', emp_cov), ('Ledoit-Wolf',
        #                                  lw_cov_), ('GraphicalLassoCV', cov_)]
        # vmax = cov_.max()
        # for i, (name, this_cov) in enumerate(covs):
        #     plt.subplot(2, 3, i + 1)
        #     plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
        #                cmap=plt.cm.RdBu_r)
        #     plt.xticks(())
        #     plt.yticks(())
        #     plt.title('%s covariance' % name)
        #
        # # plot the precisions
        # precs = [('Empirical', linalg.inv(emp_cov)), ('Ledoit-Wolf', lw_prec_),
        #          ('GraphicalLasso', prec_)]
        # vmax = .9 * prec_.max()
        # for i, (name, this_prec) in enumerate(precs):
        #     ax = plt.subplot(2, 3, i + 4)
        #     plt.imshow(np.ma.masked_equal(this_prec, 0),
        #                interpolation='nearest', vmin=-vmax, vmax=vmax,
        #                cmap=plt.cm.RdBu_r)
        #     plt.xticks(())
        #     plt.yticks(())
        #     plt.title('%s precision' % name)
        #     if hasattr(ax, 'set_facecolor'):
        #         ax.set_facecolor('.7')
        #     else:
        #         ax.set_axis_bgcolor('.7')
        # plt.show()
        # print(prec_)
        name = 'GraphicalLasso'
        this_prec = prec_
        vmax = .9 * prec_.max()
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        plt.imshow(np.ma.masked_equal(this_prec, 0),
                   interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title('year: %d' % (1999+count))
        if hasattr(ax, 'set_facecolor'):
            ax.set_facecolor('.7')
        else:
            ax.set_axis_bgcolor('.7')
        plt.show()


if __name__ == '__main__':
    # plt.bar([i+1 for i in range(len(alphas) - 1)], alphas[:-1])
    # plt.bar([12], [alpha], color='r')
    # plt.text(12, alpha, str(round(alpha, 7)), ha='center', va='bottom', fontsize=10, rotation=0)
    # plt.xlabel('Number of iterations')
    # plt.ylabel('Penalization parameter alpha')
    # # for x1, yy in zip([i+1 for i in range(len(alphas) - 1)], alphas[:-1]):
    # #     plt.text(x1, yy, str(round(yy, 5)), ha='center', va='bottom', fontsize=5, rotation=0)
    # # plt.axhline(y=alpha, c="red")
    # plt.show()
    main()
