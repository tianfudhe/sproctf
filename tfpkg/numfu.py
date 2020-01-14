import numpy as np
def tseq2xy(data, xlen, ylen):
    assert isinstance(data, np.ndarray)
    retx,rety=[],[]
    for i in range(len(data)-xlen-ylen+1):
        ix,iy=data[i:i+xlen], data[i+xlen: i+xlen+ylen]
        retx.append(ix)
        rety.append(iy)
    return np.stack(retx, axis=0), np.stack(rety, axis=0)