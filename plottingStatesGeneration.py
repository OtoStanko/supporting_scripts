import matplotlib.pyplot as plt
import numpy as np
from math import log, e

GAMMA = 0.5772156649


def average_days(n, n_unique):
    if n == n_unique:
        return n * (GAMMA + log(n))
    return sum(n/(n-n_new if n-n_new != 0 else 1) for n_new in range(0, n_unique))


def average_days2(n, n_unique):
    return sum(n/(n-n_new if n-n_new != 0 else 1) for n_new in range(0, n_unique))


def foo():
    # 100 linearly spaced numbers
    x = np.linspace(-np.pi,np.pi,100)
    x = np.linspace(0,1024, 500)

    # the function, which is y = sin(x) here
    #y = np.sin(x)
    y1 = [average_days(1024, int(i)) for i in x]
    y2 = [1024 + average_days(1024, 1024-int(i)) for i in x]


    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    """ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')"""
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    plt.plot(x,y1, 'b-')
    plt.plot(x,y2, 'm-')
    plt.plot(x,x)
    plt.plot(x, 1.008**(x))

    # show the plot
    plt.show()

def foo2():
    for n in range(100):
        x1 = 2**n
        for m in range(int(1.46212*(2**(n-1))), x1):
            if average_days(x1, x1-m) + x1 < average_days(x1, m):
                print(n, x1, m)
                break


x = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]            
y = [7,12,24,48,94,188,375,749,1498,2995,5990,11978,23956,47911,95822,
     191643,383286,766571,1533143,3066287,6132575,12265151,24530303,
     49060606,98121212,196242424,392484848,784969697]
p = [2**i for i in range(2, len(y)+2)]
#foo2()
"""for i in range(len(y)):
    print(y[i]/p[i])"""
"""for i in range(1, 2000, 50):
    print(i, average_days(1024, i))
print(1024, average_days(1024, 1024))"""
