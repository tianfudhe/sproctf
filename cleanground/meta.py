import geohash
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from jdpkg.jdutil import cpath
from tfpkg.temporal.dayfu import f_hourstep
from tfpkg.geo.gridutil import get_gridutil, SquareGridUtil
import os

def dtstr2dt(dtstr):
    sd,sh=dtstr.split(' ')
    eles=list(map(int,sd.split('-')))
    return datetime.datetime(eles[0],eles[1], eles[2], int(sh))

def tidx(dt, basedt, granual):
    r"""Parameters

    granual: granual in `hours`
    """
    t=int((dt-basedt).total_seconds()/3600.0/granual)
    return t

def normalize_sale(data, norm_flag=True):
    r"""data: [r,s,t]"""
    if norm_flag:
        mean=np.mean(data, axis=-1)[..., None] # avoid div0
        std = np.std(data, axis=-1)[..., None] 
        bias, scale = mean, std
        scale[scale==0]=1.0
    else:
        bias, scale=np.zeros([*data.shape[:2], 1]), np.ones([*data.shape[:2], 1])
    return (data-bias)/scale, np.concatenate((bias, scale), axis=-1)

def normalize_cov(data):
    r"""data: [r,s,t,d]"""
    mmax,mmin=np.max(data, axis=2)[...,None,:], np.min(data, axis=2)[...,None,:]
    scale=np.float32(mmax-mmin)
    return (data-mmin)/scale

def sale_tensor(fdir, full_region_list, gr: SquareGridUtil,
                full_sku_list, t1, t2, granual=24, smoothing=None):
    r"""Parameters

    full_region_list: [0, ..., N-1]
    full_sku_list: [SKU1, SKU2, ...]
    """
    sdict={j:i for i,j in enumerate(full_sku_list)}

    T=tidx(t2,t1,granual)

    data=np.float32(np.zeros((len(full_region_list), len(full_sku_list), T)))
    fpath=os.path.join(fdir, 'data.csv')

    _line_cnt=0
    with open(fpath, encoding='utf8') as fr:
        while True:
            iline=fr.readline()
            if iline=='':
                break
            _line_cnt+=1
            if _line_cnt % 100000 ==0 :
                print('{:04.2f}%'.format(_line_cnt*100.0/12220000), end='\r')
            # parse line
            dt,lat,lng,s_ori=iline.split('|')[1:5]
            lat,lng=float(lat), float(lng)
            s_ori=int(s_ori)

            r=gr.latlng2gid([lat,lng])
            s = None if s_ori not in sdict else sdict[s_ori]
            t=tidx(dtstr2dt(dt), t1, granual)
            if (r is None) or (s is None) or t<0 or t>T:
                continue
            data[r,s,t]+=1
    print()
    tar_path=os.path.join(fdir, 'sale-ori.npy')
    np.save(tar_path, (t1, granual, data))

    print('Original sale density: {:04.2f}%'.format(np.sum(data>1e-15)*100.0/data.size))
    if smoothing is not None:
        # region smoothing
        adj, tadj_step=smoothing
        data=np.matmul(adj, data.transpose(2,0,1)).transpose(1,2,0)
        tadj_step_l=int((tadj_step-1)/2)
        tadj_step_r=tadj_step-1-tadj_step_l
        tadj=np.zeros((data.shape[-1], data.shape[-1]))
        for i in range(data.shape[-1]):
            for j in range(-tadj_step_l,tadj_step_r+1):
                if i+j in range(data.shape[-1]):
                    tadj[i,i+j]=1
        data=np.matmul(tadj, data.transpose(0,2,1)).transpose(0,2,1)
    print('Normalized sale density: {:04.2f}%'.format(np.sum(data>1e-15)*100.0/data.size))

    data, ampl=normalize_sale(data)
    ########## do normalization and then save the amplifier

    tar_path=os.path.join(fdir, 'sale.npy')
    np.save(tar_path, (t1, granual, data[...,None]))
    ampl_tar_path=os.path.join(fdir, 'ampl_full.npy') # amplifier path
    np.save(ampl_tar_path, ampl)

def cov_tensor(fdir, full_region_list, full_sku_list, t1,t2,granual=24):
    r"""Standarization strat

    by series
    """

    tfea=f_hourstep(t1,t2)
    tfea=np.array(tfea)
    # [t,d]
    tfea=tfea.transpose(1,0)
    
    # here simply apply global temporal-covariates to all region-sku's
    tfea=np.tile(tfea, (len(full_region_list), len(full_sku_list),1,1))
    # [r,s,t,d]
    tfea=normalize_cov(tfea)
    tar_path=os.path.join(fdir, 'cov.npy')
    np.save(tar_path, (t1, granual, tfea))

def sku_st(fdir, data_name, emb_r, emb_s, region_list,
           sku_list, t1, t2, win_x, win_y, purpose='train'):
    sale_path=os.path.join(fdir, 'sale.npy')
    cov_path=os.path.join(fdir, 'cov.npy')
    ampl_path=os.path.join(fdir, 'ampl_full.npy')

    basedt, granual, sale=np.load(sale_path, allow_pickle=True)
    pos1=tidx(t1, basedt, granual)
    pos2=tidx(t2, basedt, granual)
    basedt_cov, granual_cov, covData=np.load(cov_path, allow_pickle=True)
    ampl_full=np.load(ampl_path)
    assert basedt==basedt_cov and granual == granual_cov
    assert ampl_full.shape[:2] == sale.shape[:2]

    # data: [r,s,t,d]
    data=np.concatenate((sale, covData), axis=-1)[:,:,pos1:pos2,:]
    data=data[region_list]
    data=data[:, sku_list].reshape(-1,*data.shape[2:]) # [r*s, t, d]
    ampl=ampl_full[region_list][:, sku_list].reshape(-1, 2) # [r*s, 2(b&w)]

    _X, _y=[], []
    for i in range(data.shape[1]-win_x-win_y+1):
        _X.append(data[:,i:i+win_x]) # [r*s,win_x,d]
        _y.append(data[:,i+win_x:i+win_x+win_y]) # [r*s,win_y,d]
    _X=np.stack(_X) # [b,r*s,win_x,d]
    _y=np.stack(_y) # [b,r*s,win_y,d]

    ########### step: get X,y
    X,y=_X.transpose(2,1,0,3),_y.transpose(2,1,0,3) # [t, n, b, d]

    ########### step: get meta
    ey_r=np.eye(emb_r.shape[0])
    regions_onehot=ey_r[region_list]
    rfea=regions_onehot.repeat(len(sku_list), axis=0)

    sfea=emb_s[sku_list]
    sfea=np.tile(sfea, (len(region_list), 1))

    ########### step: dump data
    tar_dir=os.path.join(fdir, data_name)
    tmpl=os.path.join(tar_dir, purpose+'_{}.npy')
    np.save(tmpl.format('x'), X)
    np.save(tmpl.format('y'), y) # [t,n,b,d]
    np.save(tmpl.format('meta_it'), sfea) # sku features, named meta item
    np.save(tmpl.format('meta_r'), rfea)
    np.save(tmpl.format('ampl'), ampl) # [n, 2]
