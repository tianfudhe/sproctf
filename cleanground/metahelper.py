import geohash
import numpy as np
import matplotlib.pyplot as plt
import datetime
from jdpkg.jdutil import cpath
from tfpkg.geo.gridutil import get_gridutil
import os
from cleanground.meta import sale_tensor, cov_tensor, sku_st


city_name='yizhuang'
data_path=cpath('sale-forecast/'+city_name)
full_t1, full_t2=datetime.datetime(2016,8,1), datetime.datetime(2019,8,1)
GRANUAL=24

def get_embr():
    tmpg=get_gridutil(city_name)
    gcnt=[0] * tmpg.num_grids
    gcnt_c1=[[] for i in range(tmpg.num_grids)]

    sale_path=os.path.join(data_path, 'data.csv')
    slist=np.load(os.path.join(data_path, 'slist.npy'))
    sset=set(slist)
    c1dict=dict()

    _cnt=0
    with open(sale_path,encoding='utf8') as fr:
        while True:
            iline=fr.readline()
            if iline=='':
                break
            if _cnt%100000==0:
                print("{:04.2f}%".format(_cnt*100/12220000), end='\r')
            _cnt+=1
            eles=iline.split('|')
            lat,lng=float(eles[2]),float(eles[3])
            gid=tmpg.latlng2gid([lat,lng])
            if gid is not None:
                gcnt[gid]+=1
                s=int(float(eles[4]))
                if s in sset:
                    c1id=int(float(eles[5]))
                    if c1id not in c1dict:
                        c1dict[c1id]=len(c1dict)
                        for _ in gcnt_c1:
                            _.append(0)
                    gcnt_c1[gid][c1dict[c1id]]+=1
    print()
    gcnt=np.array(gcnt)[:,None]
    gcnt_c1=np.array(gcnt_c1)

    loc=[]
    for gid in range(tmpg.num_grids):
        lat, lng = tmpg.gid2latlng_c(gid)
        loc.append([lat,lng])
    # embr: [N_r, d_r]
    embr=np.concatenate((loc,gcnt,gcnt_c1), axis=-1)

    # normalization
    mmin,mmax=embr.min(axis=0), embr.max(axis=0)
    scale=mmax-mmin
    embr=(embr-mmin)/scale

    fpath=os.path.join(data_path, 'emb_r.npy')
    np.save(fpath, embr)
    return embr


def get_adj(straight_only=False):
    tmpg = get_gridutil(city_name)
    adj = tmpg.get_adj(straight_only=straight_only)
    adj = np.array(adj)

    fpath = os.path.join(data_path, 'adj.npy')
    np.save(fpath, adj)
    return adj

def get_sale(smoothing=False):
    # dump `sale.npy` and `sale-ori.npy`
    gr=get_gridutil(city_name)
    full_region_list=list(range(gr.num_grids))
    full_sku_list=np.load(os.path.join(data_path, 'slist.npy'))
    print('\ndumping sales data...')
    if smoothing:
        adj=np.load(os.path.join(data_path, 'adj.npy'))
        smoothing=(adj, 4) # smoothing time by 4 steps
    else:
        smoothing=None
    sale_tensor(data_path,
                full_region_list,
                gr,
                full_sku_list,
                full_t1,full_t2,
                granual=GRANUAL,
                smoothing=smoothing)
    print('ok sales data dumped!\n')

def get_cov():
    # dump `cov.npy`
    gr=get_gridutil(city_name)
    full_region_list=list(range(gr.num_grids))
    full_sku_list=np.load(os.path.join(data_path, 'slist.npy'))
    full_t1, full_t2=datetime.datetime(2016,8,1), datetime.datetime(2019,8,1)
    emb_r=np.load(os.path.join(data_path, 'emb_r.npy'))
    emb_s=np.load(os.path.join(data_path, 'emb_s.npy'))
    print('\ndumping covariates...')
    cov_tensor(data_path,
               full_region_list,
               full_sku_list,
               full_t1,full_t2,
               granual=GRANUAL)
    print('ok covariates dumped!\n')

def visualize_sale_cov():
    import os
    import numpy as np
    from jdpkg.jdutil import cpath
    import matplotlib.pyplot as plt
    
    sale_ori_path=os.path.join(data_path, 'sale-ori.npy')
    sale_path=os.path.join(data_path, 'sale.npy')
    cov_path=os.path.join(data_path, 'cov.npy')

    basedt, granual, sale_ori=np.load(sale_ori_path, allow_pickle=True)
    basedt, granual, sale=np.load(sale_path, allow_pickle=True)
    basedt_cov, granual_cov, covData=np.load(cov_path, allow_pickle=True)
    assert basedt==basedt_cov and granual == granual_cov

    for i in range(covData.shape[-1]):
        plt.plot(1 - covData[0,0,:,i])
    plt.plot(sale[4,0].squeeze(), label='sale')
    # plt.plot(sale_ori[4,0].squeeze(), label='sale')
    plt.legend()
    plt.show()

def visualize_region():
    import os
    import numpy as np
    from jdpkg.jdutil import cpath
    import matplotlib.pyplot as plt

    gr=get_gridutil(city_name)
    sale_ori_path=os.path.join(data_path, 'sale.npy')
    basedt, granual, sale_ori=np.load(sale_ori_path, allow_pickle=True)

    thre=50000
    rcnt=sale_ori.sum(axis=(1,2))
    rcnt=np.int32(rcnt)
    
    rcnt_show=np.flip(rcnt.reshape((gr.num_rows,gr.num_cols)), axis=0)

    print(rcnt_show)
    print('---------\n #. over thre: {}\n---------'.format(np.argwhere(rcnt>thre).shape[0]))
    print(np.int32(rcnt_show>thre))

    print(','.join(map(str, list(np.argwhere(rcnt>thre).squeeze()))))
    
def extract():
    gr=get_gridutil(city_name)
    full_region_list=list(range(gr.num_grids))
    full_sku_list=np.load(os.path.join(data_path, 'slist.npy'))
    emb_r=np.load(os.path.join(data_path, 'emb_r.npy'))
    emb_s=np.load(os.path.join(data_path, 'emb_s.npy'))

    # all regions with over 10000 orders
    _rlist=full_region_list
    _slist=list(range(len(full_sku_list)))[:30]

    rl_unseen=[6,8]
    rl=[i for i in _rlist if i not in rl_unseen]

    sl=_slist[::2]
    sl_unseen=_slist[1::2]

    # t0,t1=datetime.datetime(2016,8,1), datetime.datetime(2018,10,1) # 26-1
    # t2,t3=datetime.datetime(2018,9,1), datetime.datetime(2019,2,1) # 5-1
    # t4,t5=datetime.datetime(2019,1,1), datetime.datetime(2019,8,1) # 7-1
    t0,t1=datetime.datetime(2017,8,1), datetime.datetime(2019,1,1) # 17-1
    t2,t3=datetime.datetime(2018,12,1), datetime.datetime(2019,3,1) # 3-1
    t4,t5=datetime.datetime(2019,2,1), datetime.datetime(2019,8,1) # 6-1

    sku_st(data_path, 'day30_3', emb_r, emb_s, rl, sl, t0,t1,30,3,purpose='train')
    sku_st(data_path, 'day30_3', emb_r, emb_s, rl, sl, t2,t3,30,3,purpose='val')
    sku_st(data_path, 'day30_3', emb_r, emb_s, rl, sl, t4,t5,30,3,purpose='test')


def play():
    # given sale data

    #################### ONCE ENOUGH
    ## step[ext]: get SKU_list and `emb_s`
    ### save them as `slist.npy` and `emb_s.npy`
    
    ## step[ext]: get region_list and region bound
    ### hard-code them into `gridutil.bbox_table`

    # ## step: get `adj` and `emb_r` for regions
    # get_embr()
    # get_adj(straight_only=False)
    # ## now we have `adj.npy` and `emb_r.npy`

    # ## step: dump `sale` and `cov` in shape [r,s,t]
    # get_sale(smoothing=True)
    # get_cov()

    #################### EXP DATA CONSTRUCTION
    ## step: get exp data
    extract()
