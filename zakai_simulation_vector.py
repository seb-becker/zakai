import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
import time
import pickle


def xyz_path():
    w = np.random.normal(0., np.sqrt(T / N), (d, N))
    v = np.random.normal(0., np.sqrt(T / N), (d, N))
    y = [np.random.normal(0., np.sqrt(1. / alpha), (d, ))]
    z = [np.zeros((d, ))]
    for i in range(N):
        y.append(y[-1] + beta * T / N * y[-1] / (1. + np.sum(y[-1] ** 2)) + np.sum(w[:, i]) / np.sqrt(d))
        z.append(z[-1] + gamma * T / N * (y[-2] + y[-1]) / 2. + v[:, i])

    return v, y, z


def solution(v, x_0, T, N, d, exp):
    def outer(j, mean):
        def inner(i, v, yy, BB, fact):
            w = tf.compat.v1.random_normal((batch_size, d, 1), stddev=np.sqrt(T / N))
            vv = tf.expand_dims(v[:, :, N - i - 1], axis=2)
            vv_sum = tf.reduce_sum(vv, axis=1, keepdims=True)
            yy_norm = tf.reduce_sum(yy ** 2, axis=1, keepdims=True)
            BB += fact * T / N * 0.5 * gamma ** 2 * tf.reduce_sum(
                (vv_sum / np.sqrt(d) + yy) * (vv_sum / np.sqrt(d) - yy), axis=1, keepdims=True)
            BB -= fact * T / N * gamma * beta / (1. + yy_norm) * tf.reduce_sum(vv * yy, axis=1, keepdims=True)
            BB -= fact * T / N * d * beta / (1. + yy_norm)
            BB += fact * T / N * 2 * beta * yy_norm / ((1. + yy_norm) ** 2)
            yy = yy + T/N * (gamma * vv_sum * tf.ones((1, d, 1)) - beta * yy / (1. + yy_norm)) \
                 + tf.reduce_sum(w, axis=1, keepdims=True) / np.sqrt(d)
            return i+1, v, yy, BB, tf.constant(1.)

        _i, _v, _yy, _BB, _fact = tf.while_loop(lambda i, v, yy, BB, fact: i < N, inner,
                                                (tf.constant(0), tf.cumsum(tf.expand_dims(tf.cast(v, tf.float32), axis=0), axis=2),
                                                 x_0 * tf.ones((batch_size, 1, 1)), tf.zeros((batch_size, 1, M)), tf.constant(0.5)))

        yy_norm = tf.reduce_sum(_yy ** 2, axis=1, keepdims=True)
        _BB -= 0.5 * T / N * 0.5 * gamma ** 2 * yy_norm
        _BB -= 0.5 * T / N * d * beta / (1. + yy_norm)
        _BB += 0.5 * T / N * 2 * beta * yy_norm / ((1. + yy_norm) ** 2)
        val = ((alpha / 2. / np.pi) ** (d / 2)) * tf.exp(_BB - alpha / 2. * yy_norm + exp)
        return j+1, mean + tf.reshape(tf.reduce_mean(val, axis=0, keepdims=True), (M, 1))

    _j, _mean = tf.while_loop(lambda j, mean: j < mc_runs, outer, (tf.constant(0), tf.zeros((M, 1))))

    _mean = _mean / mc_runs

    return _mean

disable_eager_execution()

alpha = 2. * np.pi
gamma = 1.
beta = 0.25
N = 20
M = 1024
batch_size = 1024
mc_runs = 100
num_solutions = 100


for d in [1, 2, 5, 10, 20, 25]:

    t_0 = time.time()

    tf.compat.v1.reset_default_graph()
    T = N / 40.
    v, y, z = xyz_path()

    z = np.stack(z, axis=1)
    z1 = np.diff(z, axis=1)

    x_in = tf.compat.v1.placeholder(tf.float32, (M, d))
    p_mean = solution(z1, tf.expand_dims(tf.transpose(x_in, (1, 0)), axis=0), T, N, d, np.sum(gamma * y[-1] * z[:, -1]))

    solutions = []
    x_0s = []
    with tf.compat.v1.Session() as sess:

        if d == 2:

            t_0 = time.time()
            xx = np.linspace(start=-5 + y[-1][0], stop=5 + y[-1][0], num=128)
            yy = np.linspace(start=-5 + y[-1][0], stop=5 + y[-1][0], num=128)

            xv, yv = np.meshgrid(xx, yy)
            xy = np.stack([xv.flatten(), yv.flatten()]).T

            solutions = []
            for l in range(16):
                p_m = sess.run(p_mean, feed_dict={x_in: xy[l * M: (l + 1) * M, :]})
                u = p_m
                solutions.append(u)

            t_1 = time.time()

            u = np.concatenate(solutions, axis=0)
            u = np.reshape(u, (128, 128))

            with open('result_' + str(d) + '_' + str(N) + '_2d.pickle', 'wb') as f:
                dic = {'x': xv, 'y': yv, 'z': u, 'xs': xx[0], 'xe': xx[-1], 'ys': yy[0], 'ye': yy[-1]}
                pickle.dump(dic, f)

        else:
            t_0 = time.time()

            xx = np.linspace(start=-5 + y[-1][0], stop=5 + y[-1][0], num=1024)
            x_1 = np.reshape(xx, (M, 1))
            xx = np.reshape(xx, (-1, 1))
            solutions = []
            p_m = sess.run(p_mean, feed_dict={x_in: xx})
            u = p_m

            t_1 = time.time()

            solutions.append(u)

            u = np.concatenate(solutions, axis=0)

            with open('result_' + str(d) + '_' + str(N) + '.pickle', 'wb') as f:
                dic = {'x': xx, 'y': u, 'xs': xx[0], 'xe': xx[-1]}
                pickle.dump(dic, f)

