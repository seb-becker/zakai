import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
import time
from scipy.stats import norm

disable_eager_execution()


def ref_solution(v, y):
    v = tf.cumsum(tf.expand_dims(tf.cast(v, tf.float32), axis=0), axis=2)
    y = tf.cast(tf.expand_dims(tf.stack(y, axis=1), axis=0), tf.float32)
    x_0 = tf.zeros((batch_size, d)) + 0.
    w = tf.compat.v1.random_normal((batch_size, d, N), stddev=np.sqrt(T / N))
    yy = [x_0]
    BB = 0.
    fact = 0.5
    for i in range(N):
        vv = v[:, :, N - i - 1]
        vv_sum = tf.reduce_sum(v[:, :, N - i - 1], axis=1, keepdims=True)
        yy_norm = tf.reduce_sum(yy[-1] ** 2, axis=1, keepdims=True)
        BB += fact * T/N * tf.reduce_sum(0.5 * vv_sum * tf.ones((1, d)) * vv * beta ** 2, axis=1, keepdims=True)
        BB += fact * T/N * tf.reduce_sum(yy[-1] * y[:, :, N - i - 1] * beta ** 2 - 0.5 * (beta * yy[-1]) ** 2
                                  - beta * gamma * vv * yy[-1] / (1. + yy_norm) - gamma * (1. + yy_norm - 2. * yy[-1] ** 2) / ((1. + yy_norm) ** 2), axis=1, keepdims=True)
        yy.append(yy[-1] + T/N * (beta * vv_sum * tf.ones((1, d)) - gamma * yy[-1] / (1. + yy_norm)) + tf.reduce_sum(w[:, :, i], axis=1, keepdims=True) / np.sqrt(d))
        fact = 1.
    vv = v[:, :, 0] * 0.
    vv_sum = tf.reduce_sum(v[:, :, 0] * 0., axis=1, keepdims=True)
    yy_norm = tf.reduce_sum(yy[-1] ** 2, axis=1, keepdims=True)
    BB += 0.5 * T / N * tf.reduce_sum(0.5 * vv_sum * tf.ones((1, d)) * vv * beta ** 2, axis=1, keepdims=True)
    BB += 0.5 * T / N * tf.reduce_sum(yy[-1] * y[:, :, 0] * beta ** 2 - 0.5 * (beta * yy[-1]) ** 2
                                      - beta * gamma * vv * yy[-1] / (1. + yy_norm) - gamma * (1. + yy_norm - 2. * yy[-1] ** 2) / ((1. + yy_norm) ** 2), axis=1, keepdims=True)
    val = ((alpha / 2. / np.pi) ** (d / 2)) * tf.exp(BB - alpha / 2. * yy_norm)
    p_mean, p_var = tf.nn.moments(val, axes=[0])
    return p_mean, p_var


alpha = 2. * np.pi
beta = 0.25
gamma = 0.25
T = 1.
N = 100

batch_size = 1024 * 8
mc_runs = 500

_file = open('Zakai_results.csv', 'w')
_file.write('d, T, N, run, value, cil, cir, time\n')

for d in [1, 2, 5, 10, 20, 25, 50, 75, 100]:

    for run in range(1):

        t_0 = time.time()

        tf.compat.v1.reset_default_graph()
        z = np.random.normal(0., np.sqrt(T / N), (d, N))
        y = [np.zeros((d,))] * N

        p_mean, p_var = ref_solution(z, y)

        p_m_l, p_v_l = [], []
        with tf.compat.v1.Session() as sess:
            for _ in range(mc_runs):
                p_m, p_v = sess.run([p_mean, p_var])
                p_m_l.append(p_m)
                p_v_l.append(p_v)

        p_m_l = np.array(p_m_l)
        p_m = np.mean(p_m_l)
        p_v = np.sum(np.array(p_v_l)) / mc_runs + np.var(p_m_l)
        tmp = norm.ppf(0.975) * np.sqrt(p_v / (mc_runs * batch_size - 1.))

        b1 = np.cumsum(z, 1)
        u_reference = p_m * np.exp(np.sum(beta * 0. * b1[:, -1]))

        ci1 = p_m - tmp
        ci2 = p_m + tmp

        t_1 = time.time()

        _file.write('%i, %f, %i, %i, %f, %f, %f, %f\n' % (d, T, N, run, p_m, ci1, ci2, t_1 - t_0))
        _file.flush()


_file.close()
