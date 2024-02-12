# requires pandas, numpy, scipy, and matplotlib
import os
import argparse
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


ID_LBL = 'ID'
CLASS_LBL = 'CLASS'

A_COLOR = (1., 0., 0.)
B_COLOR = (0., 0., 1.)
F_COLOR = (1., 1., 1.)
T_COLOR = (0., 0., 0.)
TEST_COLOR = (0., 1., 0.)


def load_csv(filename):
    df = pd.read_csv(filename, sep=',')
    labels, classes = None, None
    if CLASS_LBL in df.columns:
        labels = (df[CLASS_LBL] == df[CLASS_LBL][0]).to_numpy()
        class_0 = [df[CLASS_LBL][0]]
        class_1 = pd.unique(df[CLASS_LBL][df[CLASS_LBL] != df[CLASS_LBL][0]])
        classes = {True: class_0, False: class_1}
        del df[CLASS_LBL]
    if ID_LBL in df.columns:
        ids = df[ID_LBL].to_numpy()
        del df[ID_LBL]
    else:
        ids = np.arange(len(df)) + 1
    return ids, df, labels, classes


def match_columns(train_df, test_df):
    shared_columns = train_df.columns.intersection(test_df.columns)
    return train_df[shared_columns], test_df[shared_columns]


def method_of_moments(data):
    n = data.shape[0]  # samples
    k = data.shape[1]  # dimensions
    mu = np.mean(data, axis=0)
    sigma = np.zeros([k, k])
    for i in range(k):  # covariances
        for j in range(k):
            sigma[i, j] = np.mean(data[:, i] * data[:, j], axis=0) - mu[i] * mu[j]
    return n, mu, sigma


def average_sigma(n_a, sigma_a, n_b, sigma_b):
    return (sigma_a * n_a + sigma_b * n_b) / (n_a + n_b)


def plot_pts(id, data2d, inside_color, outside_color=None):
    for i, pt, in zip(id, data2d):
        plt.text(pt[0] + 0.5, pt[1] + 0.5, str(i), size=6., c=(1,1,1))
    if outside_color is not None:
        plt.plot(*data2d.T, 'o', c=outside_color)
    plt.plot(*data2d.T, '.', c=inside_color)


def get_extent(data2d, margin_frac=0.2):
    x = data2d[:, 0]
    y = data2d[:, 1]
    margin_x = (x.max() - x.min()) * margin_frac
    margin_y = (y.max() - y.min()) * margin_frac
    xmin = max(x.min() - margin_x, 0)
    ymin = max(y.min() - margin_y, 0)
    return (xmin, x.max() + margin_x, ymin, y.max() + margin_y)


def plot_density(mvn_a, mvn_b, extent, resolution):
    xi, yi = np.mgrid[extent[0]:extent[1]:resolution*1j, extent[2]:extent[3]:resolution*1j]
    coords = np.stack([xi.flatten(), yi.flatten()], axis=1)
    density_a = mvn_a.pdf(coords).reshape(xi.shape).T
    density_b = mvn_b.pdf(coords).reshape(xi.shape).T
    img = np.stack([density_a / density_a.max(), np.zeros_like(xi),
                    density_b / density_b.max()], axis=2)
    plt.imshow(img, aspect='auto', extent=extent, origin='lower', alpha=0.5)
    plt.plot(*mvn_a.mean, 'x', c=A_COLOR)
    plt.plot(*mvn_b.mean, 'x', c=B_COLOR)


def within_extent(pt, extent):
    return extent[0] < pt[0] and pt[0] < extent[1] and extent[2] < pt[1] and pt[1] < extent[3]


def decision_boundary(mu_a, sigma_a, mu_b, sigma_b, extent, args, do_plot):
    if do_plot:
        brute_force_decision_boundary(mu_a, sigma_a, mu_b, sigma_b, extent, args['resolution'])
    print('')
    print('Decision boundary: ')
    if not args['covariance']:
        sigma = sigma_a
        inv_sigma = np.linalg.inv(sigma)
        r = inv_sigma @ (mu_a - mu_b)
        c = r @ (mu_a + mu_b) / 2
        print('Linear rx - c = 0\n    r=\n{}\n    c=\n{}'.format(r, c))
        if do_plot:
            print(extent)
            xlim = np.array(extent[:2])
            ylim = np.array(extent[2:])
            t_x = np.concatenate([xlim, (c - r[1] * ylim) / r[0]])
            t_y = np.concatenate([(c - r[0] * xlim) / r[1], ylim])
            dominant_idx = (np.argmin(t_x), np.argmax(t_x))  # remove the dominating points
            t_x = np.delete(t_x, dominant_idx)
            t_y = np.delete(t_y, dominant_idx)
            plt.plot(t_x, t_y, c=(0, 1, 0))
    else:
        inv_sigma_a = np.linalg.inv(sigma_a)
        inv_sigma_b = np.linalg.inv(sigma_b)
        A = inv_sigma_a - inv_sigma_b
        b = inv_sigma_a @ mu_a - inv_sigma_b @ mu_b
        c = mu_a.T @ inv_sigma_a @ mu_a - mu_b.T @ inv_sigma_b @ mu_b   \
            + np.log(np.linalg.det(sigma_a)) - np.log(np.linalg.det(sigma_b))
        print('Quadratic xAx - 2bx + c = 0\n    A=\n{}\n    b=\n{}\n    c={}'.format(A, b, c))
        if do_plot:
            xi, yi = np.mgrid[extent[0]:extent[1]:args['resolution'] * 1j,
                                extent[2]:extent[3]:args['resolution']*1j]
            coords = np.stack([xi.flatten(), yi.flatten()], axis=1)
            step_dist = np.linalg.norm(coords[1] - coords[0]) * 8
            curves = []
            for pt in coords:
                xpt = newton_raphson(1, pt, A, b, c)
                if xpt is not None:  # and within_extent(xpt, extent):
                    nearest_dist = step_dist
                    nearest = -1
                    for i, curve in enumerate(curves):  # find nearest curve by last point added
                        dist = np.linalg.norm(xpt - curve[-1])
                        if dist < nearest_dist:
                            nearest = i
                            nearest_dist = dist
                    if nearest == -1:  # create new curve
                        curves.append([xpt])
                    else:
                        curves[nearest].append(xpt)
            for curve in curves:
                plt.plot(*np.stack(curve, axis=0).T, c=TEST_COLOR)
    print('')


def brute_force_decision_boundary(mu_a, sigma_a, mu_b, sigma_b, extent, resolution):
    xi, yi = np.mgrid[extent[0]:extent[1]:resolution*1j, extent[2]:extent[3]:resolution*1j]
    coords = np.stack([xi.flatten(), yi.flatten()], axis=1)
    mvn_a = multivariate_normal(mean=mu_a, cov=sigma_a)
    mvn_b = multivariate_normal(mean=mu_b, cov=sigma_b)
    epsilon = 1e-1
    xs, ys = [], []
    for pt in coords:
        if abs(mvn_a.pdf(pt) / mvn_b.pdf(pt) - 1) < epsilon:
            xs.append(pt[0])
            ys.append(pt[1])
    plt.plot(xs, ys, '.', c=TEST_COLOR)


def newton_raphson(var_idx, x0, A, b, c, epsilon=1e-5, max_iter=20):
    k = var_idx
    x = copy.deepcopy(x0)
    for i in range(max_iter):
        f = x.T @ A @ x - 2 * x.T @ b + c
        df = 2 * (A @ x - b)[k]
        prev_x = copy.deepcopy(x[k])
        x[k] -= f / df
        if abs(x[k] - prev_x) < epsilon:
            return x
    return None  # fail to converge


def density_predict(x, mvn_a, mvn_b):
    return mvn_a.pdf(x) / mvn_b.pdf(x) > 1


def test(mvn_a, mvn_b, ids, data, labels, do_plot=True):
    predictions = density_predict(data, mvn_a, mvn_b)
    correct = None
    if labels is not None:  # calculate accuracy
        correct = np.logical_not(np.logical_xor(labels, predictions))
        n = len(data)
        n_correct = np.sum(correct)
        accuracy = n_correct / n
        print('Test: {} / {} correct predictions, accuracy {}'.format(n_correct, n, accuracy))
        if do_plot:  # plot
            at = np.logical_and(correct, labels)
            af = np.logical_and(np.logical_not(correct), labels)
            bt = np.logical_and(correct, np.logical_not(labels))
            bf = np.logical_and(np.logical_not(correct), np.logical_not(labels))
            plot_pts(ids[at], data[at,:], T_COLOR, A_COLOR)
            plot_pts(ids[af], data[af,:], F_COLOR, A_COLOR)
            plot_pts(ids[bt], data[bt,:], T_COLOR, B_COLOR)
            plot_pts(ids[bf], data[bf,:], F_COLOR, B_COLOR)
    else:
        if do_plot:  # plot
            a = predictions
            b = np.logical_not(predictions)
            plot_pts(ids[a], data[a,:], A_COLOR, TEST_COLOR)
            plot_pts(ids[b], data[b,:], B_COLOR, TEST_COLOR)
    return predictions, correct


def print_dist(mu, sigma, colnames, clsnames):
    print('')
    if mu is not None:
        print('Class "{}" distribution:'.format('", "'.join(clsnames)))
        print('Means:')
        for c, m in zip(colnames, mu):
            print('    {:>10}: {:.4f}'.format(c, m))
    if sigma is not None:
        if mu is None:
            print('Pooled covariance matrix:')
        else:
            print('Covariance matrix:')
        for i in sigma:
            for j in i:
                print('{:12.4f}'.format(j), end=' ')
            print('')
    print('')


def print_and_save(string, file_handle):
    print(string)
    if file_handle is not None:
        f.write(string + '\n')


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", required=True,
        help="path to training data (.csv)")
    ap.add_argument("-T", "--test", required=False,
        help="path to data to classify")
    ap.add_argument("-o", "--output", required=False,
        help="file to save classification results to")
    ap.add_argument("-c", "--covariance", type=bool, default=False,
        help="use individual class covariances instead of mean covariances")
    ap.add_argument("-d", "--display", type=bool, default=False,
        help="show plots")
    ap.add_argument("-m", "--margin", type=float, default=0.2,
        help="margin around plotted points as proportion of extent")
    ap.add_argument("-r", "--resolution", type=int, default=100,
        help="resolution of density plot")
    args = vars(ap.parse_args())

    train_id, train_df, train_lbl, train_cls = load_csv(args['train'])
    if 'test' in args:
        test_id, test_df, test_lbl, _ = load_csv(args['test'])
        train_df, test_df = match_columns(train_df, test_df)

    data = train_df.to_numpy()
    data_a = data[train_lbl,:]
    data_b = data[np.logical_not(train_lbl),:]
    id_a = train_id[train_lbl]
    id_b = train_id[np.logical_not(train_lbl)]
    n_a, mu_a, sigma_a = method_of_moments(data_a)
    n_b, mu_b, sigma_b = method_of_moments(data_b)
    if not args['covariance']:
        sigma = average_sigma(n_a, sigma_a, n_b, sigma_b)
        sigma_a = sigma
        sigma_b = sigma
        print_dist(mu_a, None, train_df.columns, train_cls[True])
        print_dist(mu_b, None, train_df.columns, train_cls[False])
        print_dist(None, sigma, None, None)
    else:
        print_dist(mu_a, sigma_a, train_df.columns, train_cls[True])
        print_dist(mu_b, sigma_b, train_df.columns, train_cls[False])
    mvn_a = multivariate_normal(mean=mu_a, cov=sigma_a)
    mvn_b = multivariate_normal(mean=mu_b, cov=sigma_b)

    do_plot = args['display'] and (data.shape[1] == 2)  # dimensions
    extent = None
    if do_plot:
        extent = get_extent(data, args['margin'])
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
        plot_density(mvn_a, mvn_b, extent, args['resolution'])
        plot_pts(id_a, data_a, A_COLOR)
        plot_pts(id_b, data_b, B_COLOR)

    decision_boundary(mu_a, sigma_a, mu_b, sigma_b, extent, args, do_plot)

    if 'test' in args:
        predictions, correct = test(mvn_a, mvn_b, test_id, test_df.to_numpy(),
                                    test_lbl, do_plot)
        print('')
        f = None
        if 'output' in args and args['output'] is not None:
            f = open(os.path.splitext(args['output'])[0] + '-prediction.csv', 'w')
        if correct is not None:
            print_and_save('ID,PREDICTION,CORRECT', f)
            for i, p, c in zip(test_id, predictions, correct):
                print_and_save('{},{},{}'.format(i, train_cls[p][0], c), f)
        else:
            print_and_save('ID,PREDICTION', f)
            for i, p in zip(test_id, predictions):
                print_and_save('{},{}'.format(i, train_cls[p][0]), f)
        if f is not None:
            f.close()
        print('')

    if do_plot:
        plt.xlabel(train_df.columns[0])
        plt.ylabel(train_df.columns[1])
        plt.legend([train_cls[True][0], train_cls[False][0]])
        if 'output' in args and args['output'] is not None:
            plt.savefig(os.path.splitext(args['output'])[0] + '-plot.png')
        plt.show()
