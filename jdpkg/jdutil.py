import datetime
import os
import numpy as np
from vincenty import vincenty
from jdpkg.jdconf import glconf
import tfpkg.tfcoord as tfcoord
from dateutil.parser import parse as parseDtStr
import geohash

baseDt = datetime.datetime(2017, 8, 1)
endDt = datetime.datetime(2017, 11, 1)
path_figure = os.path.join(glconf['dir_data'], "figures")


def cpath(s):
    return os.path.join(glconf['dir_data'], s)

def mmAnglePass(ofst, err):
    if len(ofst) == 1:
        return True
    centPos = int(len(ofst) / 2)
    delta_ofst = np.mean(ofst[centPos:]) - np.mean(ofst[:centPos])
    if delta_ofst == 0:
        return False
    delta_err = np.mean(err[centPos:]) - np.mean(err[:centPos])
    return abs(np.arctan(delta_err / delta_ofst)) < np.pi / 3


def mmDisPass(err):
    mmDisThre = 34.5
    return np.mean(np.abs(err)) < mmDisThre


def smooth_INTP_MA(x, y, ws):
    def avgwindow(xy):
        x, y = zip(*xy)
        if abs(x[-1] - x[0]) < 0.1:
            return (x[0] + x[-1]) / 2, np.mean(y)
        avg = 0
        for i in range(len(x) - 1):
            dx = x[i + 1] - x[i]
            yValue = (y[i + 1] + y[i]) / 2
            avg += dx * yValue
        avg /= x[-1] - x[0]
        return (x[0] + x[-1]) / 2, avg

    pt = int((x[0] + ws) / ws)
    pos = []
    l = []
    for i in range(len(x) - 1):
        l.append([x[i], y[i]])
        while x[i] <= pt * ws <= x[i + 1]:
            pos.append(len(l))
            vlinePos = pt * ws
            vlineHeight = y[i] + (y[i + 1] - y[i]) * \
                (pt * ws - x[i]) / (x[i + 1] - x[i])
            l.append((vlinePos, vlineHeight))
            pt += 1
    l.append((x[-1], y[-1]))

    pos = [0] + pos + [len(l) - 1]
    ret = []
    for i in range(len(pos) - 1):
        avgx, avgy = avgwindow(l[pos[i]:pos[i + 1] + 1])
        ret.append(avgy)
    tmpRet = []
    for i in range(0, int(x[0] / ws)):
        tmpRet.append(0)
    ret = tmpRet + ret
    retOfst = [ws * i + ws / 2 for i in range(len(ret))]
    return retOfst[int(x[0] / ws):], ret[int(x[0] / ws):]


def cpth(s):
    return os.path.join(glconf['basedir'], s)


def tstmp2dtStr(t):
    # if len(t) != 13:
    #     raise ValueError("Timestamp length is not 13.")
    tmpDT = datetime.datetime.fromtimestamp(int(t[:-3]))
    tmpMS = t[-3:]
    return tmpDT.strftime('%Y/%m/%d %H:%M:%S') + '.' + tmpMS


def tstmp2dt(t):
    return datetime.datetime.fromtimestamp(t)


def dtStr2dt(dtStr):
    return parseDtStr(dtStr)

def dtFloor(dt):
    return datetime.datetime(dt.year, dt.month, dt.day)

def tStr2t(tStr):
    if '.' in tStr:
        return datetime.datetime.strptime(tStr, "%H:%M:%S.%f").time()
    return datetime.datetime.strptime(tStr, "%H:%M:%S").time()


def speed(p1, p2):
    tmpdis = geodis(p1[:2], p2[:2])
    tmpt = (p2[2] - p1[2]) / 1000
    return tmpdis / tmpt * 3.6


def stringifyTrack(t):
    return ','.join([' '.join(map(str, i)) for i in t])


def stringifyTrack_wkt(t):
    return 'LINESTRING({})'.format(stringifyTrack(t))


def stringifyTrackList(tList):
    return ';'.join(map(stringifyTrack, tList))


def parseLatLng(s, valSep=' '):
    eles = s.split(valSep)
    return [float(eles[0]), float(eles[1])]


def parseTrack(s, pointSep=',', valSep=' '):
    if s == '':
        return []
    return list(map(lambda o: parseLatLng(o, valSep=valSep), s.split(pointSep)))


def parseTrackList(s, trackSep=';', pointSep=',', valSep=' '):
    return list(map(lambda o: parseTrack(o, pointSep=pointSep, valSep=valSep), s.split(trackSep)))


def parseHiveTrack(s):
    def parseHivePoint(sPoint):
        locStr, tStr = sPoint.split(';')
        lngStr, latStr = locStr.split(',')
        return [float(latStr), float(lngStr), int(float(tStr))]

    return list(map(parseHivePoint, s[1:].split('#')))


def parseMyTrack(s):
    def parseHivePoint(sPoint):
        latStr, lngStr, tStr = sPoint.split(' ')
        return [float(latStr), float(lngStr), int(tStr)]

    return list(map(parseHivePoint, s.split(',')))


def stTr2sTr(s):
    ret = parseMyTrack(s)
    ret = [str(i[0]) + ' ' + str(i[1]) for i in ret]
    return ','.join(ret)


def geodis(latlng1, latlng2):
    return vincenty(latlng1, latlng2) * 1000.0


def dis2EWSN(olatlng, dis_meters):
    # E, W, S, N
    if dis_meters > 25000:
        raise ValueError("`dis_meters` too large!")

    eps = 0.01
    E, W, S, N = np.array([0, 1]), np.array(
        [0, -1]), np.array([-1, 0]), np.array([1, 0])

    def _dis2latlng(o, tdir):
        l, r = 0, 1
        while True:
            m = (l+r)*0.5
            mDis = geodis(o, o+tdir*m)
            if abs(mDis-dis_meters) < eps:
                return m
            if mDis < dis_meters:
                l = m
            else:
                r = m
    oriPoint = np.array(olatlng)
    return [_dis2latlng(oriPoint, iDir) for iDir in [E, W, S, N]]


def printproc(value):
    print(value, end='\r', flush=True)


def toPlain(l, f=lambda o: o):
    ret = []
    for i in l:
        ret.extend(list(map(f, i)))
    return ret


def sourceStr(sCities):
    sCities.sort()
    return '-'.join(sCities)


def noiseLatlng(lat, lng):
    LB = [39.26, 115.25]
    RU = [41.03, 117.30]
    return not (LB[0] < lat < RU[0] and LB[1] < lng < RU[1])


cityRangeDict = {
    # bbox in GCJ
    'msra': [[39.9705, 116.2949], [39.9904, 116.3227]],
    'mobike': [[39.9401, 116.4577], [39.9562, 116.4808]],
    'chaoyang': [[39.919547, 116.433453], [39.962322, 116.476626]],
    'chaoyangb': [[39.922932, 116.43232], [39.944115, 116.479178]],
    'chaoyangu': [[39.944115, 116.43232], [39.965299, 116.479178]],
    'haidian': [[39.950437, 116.282719], [40.001609, 116.355074]],
    'haidianl': [[39.964317, 116.303107], [40.00448, 116.32418]],
    'haidianr': [[39.964317, 116.32418], [40.00448, 116.345253]],
    'baoding': [[39.042081, 115.846261], [39.063556, 115.890271]],
    'chengdu': [[30.641904, 104.043243], [30.670983, 104.090194]],
    'hefei': [[31.849315, 117.260277], [31.877417, 117.312215]],
    'yizhuang': [[39.764676, 116.486948], [39.810646, 116.543854]]
}


def InCity(cityName, latlng, wgs=False):
    lat, lng = latlng
    if wgs:
        lat, lng = wgs2gcj(lat, lng)
    LB, RU = cityRangeDict[cityName]
    return LB[0] < lat < RU[0] and LB[1] < lng < RU[1]

def InCityDict(cityName):
    import functools
    return functools.partial(InCity, cityName)


def CityBound(cityName, wgs=False):
    LB, RU = cityRangeDict[cityName]
    retLB, retRU=[0,0],[0,0]
    if wgs:
        retLB[0], retLB[1] = gcj2wgs(LB[0], LB[1])
        retRU[0], retRU[1] = gcj2wgs(RU[0], RU[1])
    return retLB, retRU


def wgs2gcj(lat, lng):
    lng, lat = tfcoord.wgs84_to_gcj02(lng, lat)
    return lat, lng


def gcj2wgs(lat, lng):
    lng, lat = tfcoord.gcj02_to_wgs84(lng, lat)
    return lat, lng


def ecdf(x_):
    x = np.sort(x_)
    x = np.concatenate((x[:1], x))
    p = np.linspace(0, 1, len(x))
    return np.array(x), np.array(p)
