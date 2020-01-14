from tfpkg.temporal.chinaholiday import ChinaHoliday, PH_purch
import datetime

ch = ChinaHoliday()
chp = ChinaHoliday(PH_purch)


def f():
    t1 = datetime.datetime(2013, 1, 1)
    t2 = datetime.datetime(2020, 1, 1)
    num_days = (t2-t1).days
    DEFAULT_VALUE = 365
    holiday_ahead = [DEFAULT_VALUE]*num_days
    restday_ahead = [DEFAULT_VALUE]*num_days
    purchday_ahead = [DEFAULT_VALUE]*num_days
    purchday_past = [DEFAULT_VALUE]*num_days
    print(num_days)

    # public holiday
    found = False
    for i in range(num_days-1, -1, -1):
        t = t1+datetime.timedelta(days=i)
        t=t.date()
        if ch.is_public_holiday(t):
            holiday_ahead[i] = 0
            found = True
        elif found:
            holiday_ahead[i] = holiday_ahead[i+1]+1

    # non-workday
    found = False
    for i in range(num_days-1, -1, -1):
        t = t1+datetime.timedelta(days=i)
        t=t.date()
        if ch.is_holiday(t):
            restday_ahead[i] = 0
            found = True
        elif found:
            restday_ahead[i] = restday_ahead[i+1]+1

    # purchday
    found = False
    for i in range(num_days-1, -1, -1):
        t = t1+datetime.timedelta(days=i)
        t=t.date()
        if chp.is_public_holiday(t):  # do not use is_holiday
            purchday_ahead[i] = 0
            found = True
        elif found:
            purchday_ahead[i] = purchday_ahead[i+1]+1

    # days passed from purchday
    found = False
    for i in range(num_days):
        t = t1+datetime.timedelta(days=i)
        t=t.date()
        if chp.is_public_holiday(t):
            purchday_past[i] = 0
            found = True
        elif found:
            purchday_past[i] = purchday_past[i-1]+1
    daylist = [t1+datetime.timedelta(days=i) for i in range(num_days)]
    return [daylist, holiday_ahead, restday_ahead, purchday_ahead, purchday_past]

def f_hourstep(t1,t2,hours=24):
    tmpx=f() # from 20130101 to 20200101
    offset=(t1-datetime.datetime(2013,1,1)).days
    wsize=(t2-t1).days
    tmpx=tmpx[1:]
    for i in range(len(tmpx)):
        tmpx[i]=tmpx[i][offset:offset+wsize]
    if hours==24:
        return tmpx
    assert 24%hours == 0
    steps_perday=int(24/hours)
    x=[]
    for itmpx in tmpx:
        x.append([])
        for iv in itmpx:
            x[-1].extend([iv]*steps_perday)
    return x

def play():
    data=f()
    txt='date,holiday,restday,purchday,purchday_past\n'
    for i in range(len(data[0])):
        txt+='{},{},{},{},{}\n'.format(*[data[j][i] for j in range(5)])
    return txt


# paste to playground.py to run
def testdayfu():
    from jdpkg.jdutil import cpath
    import numpy as np
    import matplotlib.pyplot as plt
    from tfpkg.temporal.dayfu import f_hourstep

    def plot01(x, label):
        x = np.array(x)
        x -= x.min()
        x = x/x.max()
        plt.plot(x, label=label)
    # visual tset case 1:
    t1 = datetime.datetime(2019, 4, 1)
    t2 = datetime.datetime(2019, 8, 1)
    step_size = 24

    x = f_hourstep(t1, t2, hours=step_size)
    feacnt = 0
    for ix in x:
        plot01(ix, label='fea{}'.format(feacnt))
        feacnt += 1
    data = np.load(cpath('spatial-shopping/tensor1-day_grid_item.npy'))
    data = data.sum(axis=1).sum(axis=1)
    plot01(data, label='sell volume')
    plt.legend()
    plt.show()

    # visual test case 2:
    t1 = datetime.datetime(2016, 8, 1)
    t2 = datetime.datetime(2019, 8, 1)
    step_size = 6

    x = f_hourstep(t1, t2, hours=step_size)
    feacnt = 0
    for ix in x:
        plot01(ix, label='fea{}'.format(feacnt))
        feacnt += 1
    data = np.load(cpath('spatial-shopping/3years/tensor1-day_grid_item.npy'))
    data = data.sum(axis=1).sum(axis=1)
    plot01(data, label='sell volume')
    plt.legend()
    plt.show()
