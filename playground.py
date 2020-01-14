import geohash
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from jdpkg.jdutil import cpath
from tfpkg.temporal.dayfu import f_hourstep
from cleanground.metahelper import play, visualize_region, extract
from cleanground.metahelper import visualize_sale_cov

def test_singular():
    def is_singular(l):
        def f(l):
            tadj_step=3
            tadj_step_l=int((tadj_step-1)/2)
            tadj_step_r=tadj_step-1-tadj_step_l
            tadj=np.zeros((l, l))
            for i in range(l):
                for j in range(-tadj_step_l,tadj_step_r+1):
                    if i+j in range(l):
                        tadj[i,i+j]=1
            return tadj
        return l-np.linalg.matrix_rank(f(l))
    ret=[]
    for i in range(3, 1000):
        if i%10==1:
            print('{:04.2f}%'.format(i*100.0/1000), end='\r')
        dr=is_singular(i)
        if dr > 0:
            ret.append((i,dr))
    return ret


def test(R,C):
    def get_adj(R,C):
        num=R*C
        ret=[[0]*num for i in range(num)]
        for g in range(num):
            for i in range(-1,2):
                for j in range(-1,2):
                    row, col=int(g/C), g%C
                    _row, _col=row+i, col+j
                    if (_row not in range(R)) or (_col not in range(C)) or abs(i)+abs(j)>1:
                        continue
                    _g=_row*C+_col
                    ret[g][_g]=1
        return ret
    adj=np.array(get_adj(R,C))+np.eye(R*C)
    return adj, adj.shape[0] - np.linalg.matrix_rank(adj)

# for i in range(3,101):
#     for j in range(3,101):
#         adj,dr=test(i,j)
#         if dr==0:
#             print(i,j)


if __name__ == "__main__":
    # visualize_sale_cov()
    play()
