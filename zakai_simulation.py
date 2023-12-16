import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
import time
from scipy.stats import norm


def vyz_path():
    w = np.random.normal(0., np.sqrt(T / N), (d, N))
    v = np.random.normal(0., np.sqrt(T / N), (d, N))
    y = [np.random.normal(0., np.sqrt(1. / alpha), (d, ))]
    z = [np.zeros((d, ))]
    for i in range(N):
        y.append(y[-1] + beta * T / N * y[-1] / (1. + np.sum(y[-1] ** 2)) + np.sum(w[:, i]) / np.sqrt(d))
        z.append(z[-1] + gamma * T / N * (y[-2] + y[-1]) / 2. + v[:, i])

    return v, y, z


def solution(v, d, x_0, exp):
    v = tf.cumsum(tf.expand_dims(tf.cast(v, tf.float32), axis=0), axis=2)
    x_0 = tf.constant(x_0, dtype=tf.float32)
    w = tf.compat.v1.random_normal((batch_size, d, N), stddev=np.sqrt(T / N))
    yy = [x_0]
    BB = 0.
    fact = 0.5
    for i in range(N):
        vv = v[:, :, N - i - 1]
        vv_sum = tf.reduce_sum(v[:, :, N - i - 1], axis=1, keepdims=True)
        yy_norm = tf.reduce_sum(yy[-1] ** 2, axis=1, keepdims=True)
        BB += fact * T / N * 0.5 * gamma ** 2 * tf.reduce_sum((vv_sum / np.sqrt(d) + yy[-1])
                                                              * (vv_sum / np.sqrt(d) - yy[-1]), axis=1, keepdims=True)
        BB -= fact * T / N * gamma * beta / (1. + yy_norm) * tf.reduce_sum(vv * yy[-1], axis=1, keepdims=True)
        BB -= fact * T / N * d * beta / (1. + yy_norm)
        BB += fact * T / N * 2 * beta * yy_norm ** 2 / ((1. + yy_norm) ** 2)
        yy.append(yy[-1] + T/N * (gamma * vv_sum * tf.ones((1, d)) - beta * yy[-1] / (1. + yy_norm))
                  + tf.reduce_sum(w[:, :, i], axis=1, keepdims=True) / np.sqrt(d))
        fact = 1.
    yy_norm = tf.reduce_sum(yy[-1] ** 2, axis=1, keepdims=True)
    BB -= 0.5 * T / N * 0.5 * gamma ** 2 * yy_norm
    BB -= 0.5 * T / N * d * beta / (1. + yy_norm)
    BB += 0.5 * T / N * 2 * beta * yy_norm ** 2 / ((1. + yy_norm) ** 2)
    val = tf.exp(BB - alpha / 2. * yy_norm + exp)
    p_mean, p_var = tf.nn.moments(val, axes=[0])
    return p_mean, p_var


disable_eager_execution()

alpha = 2. * np.pi
gamma = 1.
beta = 0.25
T = 0.5
N = 100

batch_size = 1024 * 8
mc_runs = 500
steps = 12000


_file = open('results_scalar.csv', 'w')
_file.write('d, T, N, run, value_0, cil, cir, value_y, cil, cir, y, time\n')

for d in [1, 2, 5, 10, 20, 25]:

    for run in range(5):

        t_0 = time.time()
        tf.compat.v1.reset_default_graph()
        v, y, z = vyz_path()
        z = np.stack(z, axis=1)
        z1 = np.diff(z, axis=1)
        x_0 = np.ones((batch_size, d)) * np.reshape(z[:, -1] / gamma / T, (1, d))
        p_mean, p_var = solution(z1, d, x_0, np.sum(z[:, -1] ** 2))

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

        b1 = np.cumsum(v, 1)
        u_reference = p_m
        ci1 = p_m - tmp
        ci2 = p_m + tmp

        t_1 = time.time()

        tf.compat.v1.reset_default_graph()
        x_0 = np.ones((batch_size, d)) * np.reshape(y[-1], (1, d))
        p_mean, p_var = solution(z1, d, x_0, np.sum(gamma * y[-1] * z[:, -1]))

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

        u_reference1 = p_m
        ci11 = p_m - tmp
        ci21 = p_m + tmp

        _file.write('%i, %f, %i, %i, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %f, %f\n'
                    % (d, T, N, run, u_reference, ci1, ci2, u_reference1, ci11, ci21, y[-1][0], t_1 - t_0))
        _file.flush()


_file.close()
