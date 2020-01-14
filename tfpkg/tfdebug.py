import time as pytime
import matplotlib.pyplot as plt


class tfDida(object):
    tList = []

    @staticmethod
    def di():
        tfDida.tList.clear()

    @staticmethod
    def da():
        t = tfDida.tList
        t.append(pytime.time())
        if len(t) == 1:
            return 0.0, 0.0
        return t[-1] - t[-2], t[-1] - t[0]

    @staticmethod
    def daa(info=""):
        pre, ever = tfDida.da()
        print('\t{0}\n\t-   PRE: {1},\tEVER: {2}.   -\n'.format(info, pre, ever))
        return pre, ever


def adjust_ticks_fontsize(sz):
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(sz)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(sz)


def series_smoother(l, ws):
    l = [0] * int(ws / 2) + l + [0] * (ws - int(ws / 2))
    i_start = int(ws / 2)

    sum1, pt1 = 0, 0
    sum2, pt2 = sum(l[:ws]), ws
    ret = []
    for i in range(len(l) - ws):
        # [i-ws/2 : i+ws-ws/2]
        tmpN = min(len(l) - ws, i + ws - int(ws / 2)) - max(0, i - int(ws / 2))
        ret.append((sum2 - sum1) / tmpN)
        sum1 += l[pt1]
        sum2 += l[pt2]
        pt1 += 1
        pt2 += 1
    return ret
