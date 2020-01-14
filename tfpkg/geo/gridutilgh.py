import geohash
import random

bbox_table = {'beijing': [116.069093, 39.685882, 116.738295, 40.215102],
              'yizhuang':[116.48090976668708, 39.763425439462935, 116.53800984878623, 39.80955021101395]}
# 'yizhuang': [[39.764676, 116.486948], [39.810646, 116.543854]
# 6-precision grid length (lat, lng): 0.0054931640625 0.010986328125


class GidMapper(object):
    def __init__(self, bbox, precision=6):
        """bbox as list: [lng-LB, lat-LB, lng-RU, lat-RU]"""
        self.bbox = bbox
        self.precision = precision
        self.d_gid2hash, self.d_hash2gid = dict(), dict()
        minLng, minLat, maxLng, maxLat = self.bbox
        curG = geohash.encode(minLat, minLng, precision=precision)
        # bbox of grid
        tmpBbox = geohash.bbox(curG)
        bs, bw = tmpBbox['s'], tmpBbox['w']
        # neighbors of grid `x`
        # 2 3 4
        # 0 x 1
        # 5 6 7
        neis = geohash.neighbors(curG)
        
        self.num_col=0
        gid = 0
        while bs <= maxLat:
            # record new row
            northG = neis[3]
            while bw <= maxLng:
                self.d_gid2hash[gid] = curG
                self.d_hash2gid[curG] = gid
                gid += 1
                eastG = neis[1]
                curG = eastG
                neis = geohash.neighbors(curG)
                tmpBbox = geohash.bbox(curG)
                bs, bw = tmpBbox['s'], tmpBbox['w']
            if self.num_col==0:
                self.num_col=gid
            curG = northG
            neis = geohash.neighbors(curG)
            tmpBbox = geohash.bbox(curG)
            bs, bw = tmpBbox['s'], tmpBbox['w']
        self.num_grids=len(self.d_gid2hash.keys())
        self.num_row=self.num_grids//self.num_col
        if self.num_grids % self.num_col!=0:
            raise ValueError('not a rectangle!')

    def lnglat2gid(self, lng, lat, return_none=False):
        gh = geohash.encode(lat, lng, self.precision)
        return self.hash2gid(gh,return_none=return_none)
    
    def latlng2gid(self, lat, lng, return_none=False):
        return self.lnglat2gid(lng, lat,return_none=return_none)

    def gid2latlng(self, gid):
        retBbox=geohash.bbox(self.gid2hash(gid))
        # swne
        return [retBbox[i] for i in 'swne']
    
    def gid2latlng_c(self,gid):
        _swne=self.gid2latlng(gid)
        return [(_swne[0]+_swne[2])*0.5, (_swne[1]+_swne[3])*0.5]

    def gid2hash(self, gid):
        if gid not in self.d_gid2hash:
            raise ValueError("No such gid: {}".format(gid))
        return self.d_gid2hash[gid]

    def hash2gid(self, gh, return_none=False):
        if gh not in self.d_hash2gid:
            if return_none:
                return None
            else:
                raise ValueError("Query {} not in city.".format(gh))
        return self.d_hash2gid[gh]
    
    def get_adj(self, straight_only=False):
        if straight_only:
            neiIdx=[0,1,3,6]
        else:
            neiIdx=list(range(8))
        adj=[[0]*self.num_grids for i in range(self.num_grids)]
        for gid in range(self.num_grids):
            latlng=self.gid2latlng_c(gid)
            _gh=self.gid2hash(gid)
            neis= geohash.neighbors(_gh)
            neis=[neis[i] for i in neiIdx]
            for inei in neis:
                _nei_gid=self.hash2gid(inei, return_none=True) 
                if _nei_gid is not None:
                    adj[gid][_nei_gid]=1
            adj[gid][gid]=1
        return adj

def get_gidmapper(cityName, precision):
    return GidMapper(bbox_table[cityName], precision=precision)

def geohash_decode_noise(gh):
    tmpBbox = geohash.bbox(gh)
    r1, r2 = random.random(), random.random()
    return (r1*tmpBbox['w']+(1-r1)*tmpBbox['e'], r2*tmpBbox['s']+(1-r2)*tmpBbox['n'])
