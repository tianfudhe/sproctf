bbox_table={'beijing':[[39.770013, 116.198948], [40.027082, 116.57729]],
            'yizhuang':[[39.76342, 116.48091], [39.80955, 116.53801]]}

gsize_table={
    'beijing': [0.035, 0.035], # ?
    'yizhuang': [0.016, 0.0191], # r3c3
    # 'yizhuang': [0.016, 0.015], # r3c4
}

class SquareGridUtil(object):
    def __init__(self, bound, gsr, gsc):
        """`gsr`: grid size -  row \\
            `gsc`: grid size - col \\
            `8`   `9`  `10`  `11` \\
            `4`   `5`  `6`   `7` \\
            `0`   `1`  `2`   `3` 
        """
        LB, RU = bound  # latlng
        self.olatlng = LB
        self.gsr = gsr
        self.gsc = gsc
        self.num_rows = int((RU[0]-self.olatlng[0])/gsr)+1
        self.num_cols = int((RU[1]-self.olatlng[1])/gsc)+1
        self.gidList = [i for i in range(self.num_rows*self.num_cols)]
        self.num_grids=len(self.gidList)

    def grid_in_city(self, row, col):
        if (0 <= row < self.num_rows) and (0 <= col < self.num_cols):
            return True
        return False

    def latlng2gid(self, latlng):
        row = int((latlng[0]-self.olatlng[0]) / self.gsr)
        col = int((latlng[1]-self.olatlng[1]) / self.gsc)
        return self._rowcol2gid(row, col)

    def gid2latlng(self, gid):
        # swne
        row, col=self._gid2rowcol(gid)
        swne=[self.olatlng[0]+row*self.gsr,
              self.olatlng[1]+col*self.gsc,
              self.olatlng[0]+(1+row)*self.gsr,
              self.olatlng[1]+(1+col)*self.gsc,]
        return swne
    
    def gid2latlng_c(self, gid):
        swne=self.gid2latlng(gid)
        return [(swne[0]+swne[2])*0.5, (swne[1]+swne[3])*0.5]

    def neighbors(self, gid):
        ret = []
        row, col = self._gid2rowcol(gid)
        for i in range(-1, 2):
            for j in range(-1, 2):
                tmp_row, tmp_col = row+i, col+j
                ret.append(self._rowcol2gid(tmp_row, tmp_col))
        return ret

    def _gid2rowcol(self, gid):
        row, col = int(gid/self.num_cols), gid % self.num_cols
        return row, col

    def _rowcol2gid(self, row, col):
        if not self.grid_in_city(row, col):
            return None
        return row*self.num_cols+col

    def neighbor(self, gid, neighbor_offset):
        """neighbor offset \\
            `6` `7` `8`    \\
            `3` `4` `5`    \\
            `0` `1` `2`    \\
            (`4` refers to the current grid `gid`)"""
        offset_row = int(neighbor_offset / 3) - 1
        offset_col = neighbor_offset % 3 - 1
        row, col = self._gid2rowcol(gid)
        row += offset_row
        col += offset_col
        return self._rowcol2gid(row+offset_row, col+offset_col)
    
    def neighbors(self, gid):
        ret=[]
        row,col=self._gid2rowcol(gid)
        for i in range(-1,2):
            for j in range(-1,2):
                if i==0 and j==0:
                    ret.append(gid)
                    continue
                _gid=self._rowcol2gid(row+i, col+j)
                ret.append(_gid)
        return ret
    
    def get_adj(self, straight_only=False):
        if straight_only:
            neiIdx=[1,3,4,5,7]
        else:
            neiIdx=list(range(9))
        adj=[[0]*self.num_grids for i in range(self.num_grids)]
        for gid in range(self.num_grids):
            neis= self.neighbors(gid)
            neis=[neis[i] for i in neiIdx]
            for inei in neis:
                if inei is None:
                    continue
                adj[gid][inei]=1
        return adj

def get_gridutil(city_name) -> SquareGridUtil:
    grid_size_lat, grid_size_lng=gsize_table[city_name]
    return SquareGridUtil(bbox_table[city_name], grid_size_lat, grid_size_lng)

def play():
    import numpy as np
    def printadj(gr, n):
        adj=np.array(gr.get_adj(straight_only=False))
        print(adj[n].reshape(gr.num_rows, -1))
    def printadj_s(gr, n):
        adj=np.array(gr.get_adj(straight_only=True))
        print(adj[n].reshape(gr.num_rows, -1)) 
    gr=get_gridutil('yizhuang', 0.016, 0.015)
    print('rows: {}, cols: {}'.format(gr.num_rows, gr.num_cols))
    
    adj_straight=np.array(gr.get_adj(straight_only=True))
    return gr.get_adj(), gr.get_adj(straight_only=True)
