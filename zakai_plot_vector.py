import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pickle
from mpl_toolkits.mplot3d import Axes3D

T = 0.5
N = 20

for d in [1, 2, 5, 10, 20, 25]:

    plt.rcParams['text.usetex'] = True
    fig = plt.figure()
    fig.set_tight_layout(True)
    fig.set_size_inches(12, 12)

    title_font = {'size': '40'}  # Bottom vertical alignment for more space
    axis_font = {'size': '40'}

    if d == 2:
        with open('result_' + str(d) + '_' + str(N) + '_2d.pickle', 'rb') as f:
            dic = pickle.load(f)

        xv = dic['x']
        yv = dic['y']
        u = dic['z']
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(xv, yv, u, cmap=cm.coolwarm)
        ax.set_xlabel('$x_1$', **axis_font)
        ax.set_ylabel('$x_2$', **axis_font)
        ax.set_xlim(dic['xs'], dic['xe'])
        ax.set_ylim(dic['ys'], dic['ye'])
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel('unnormalized conditional density', **axis_font, rotation=90)
        ax.zaxis.labelpad = 25
        ax.yaxis.labelpad = 15
        ax.xaxis.labelpad = 15
        ax.tick_params(axis='z', pad=10)

        ax.set_zlim(0., 0.3)
        ax.set_title('$d = %s$' % str(d), x=0.45, y=0.98, **title_font)
        ax.view_init(elev=20., azim=35, roll=-1.5)
        for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            label.set_fontsize(32)
    else:
        with open('result_' + str(d) + '_' + str(N) + '.pickle', 'rb') as f:
            dic = pickle.load(f)

        x = dic['x']
        u = dic['y']
        ax = fig.gca()
        ax.set_xlabel('$x$', **axis_font)
        ax.set_ylabel('unnormalized conditional density', **axis_font)
        ax.set_title('$d = %s$' % str(d), **title_font)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(32)

        ax.set_xlim(dic['xs'], dic['xe'])
        plt.plot(x, u)
        plt.vlines((dic['xs'] + 5.), 0., u[511], colors='k', linestyles='dotted', label='y')

    fig.tight_layout()
    plt.savefig('plot_' + str(d) + '_' + '.pdf')
    plt.show()