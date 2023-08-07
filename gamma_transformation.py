import numpy as np
import math 
from scipy.special import gamma, psi, polygamma
import scipy.stats as stats
import matplotlib.pyplot as plt

def gammaDist(x,k,th):
    return 1/(gamma(k) * th ** k) * x ** (k-1) * math.e ** ( - x / th )

def gammaDist2(x,k,th,mu):
    return np.array(list(map(lambda xx: 0 if xx < mu else 1/(gamma(k) * th ** k) * (xx-mu) ** (k-1) * math.e ** ( - (xx-mu) / th ), x)), dtype='longdouble')
    
def convolve(t,dt,y1,y2):
    yc = []
    for i in range(len(t)):
        y0 = 0 
        for j in range(i):
            y0 += y1[j] * y2[i-j]
        yc.append(y0 * dt)
    return yc

def normalize(y,dt):
    total = sum(y)*dt
    print(total)
    y /= total
    return y

def getGammaPar(x,y,dx,n = 5):
    ex = 0
    exx = 0
    for i in range(len(x)):
        ex += x[i] * y[i] * dx
        exx += (x[i] ** 2) * y[i] * dx
    vx = exx - ex ** 2
    kh = ex ** 2 / vx
    thh = vx / ex
    return [kh, thh, ex, vx]

def getGammaPars(x,y,dx,n = 5):
    ex = 0
    exx = 0
    exxx = 0
    for i in range(len(x)):
        ex += x[i] * y[i] * dx
        exx += (x[i] ** 2) * y[i] * dx
        exxx += (x[i] ** 3) * y[i] * dx
    vx = exx - ex ** 2
    sig = math.sqrt(vx)
    skx = (exxx - 3 * ex * sig ** 2 - ex ** 3) / sig ** 3
    kh = 4 / skx ** 2
    thh =  math.sqrt(vx / kh)
    muh = ex - kh * thh
    return [kh, thh, muh, ex, exx, vx, sig, exxx, skx]

def getFitError(y,yh):
    err = 0 
    for i in range(len(y)):
        err += (yh[i] - y[i]) ** 2 / len(y)
    err = math.sqrt(err)
    return err

k =  6981.278075034904
th = 1031.4967007158257
offset = 0
mu = -offset
coeff = 1

xmin = stats.gamma.ppf(1e-6,k,loc = mu, scale = th) - 0.1
xmax = stats.gamma.ppf(1-1e-6,k,loc = mu, scale = th)
dx = (xmax-xmin)/100

t = np.arange(xmin,xmax,dx)

y = gammaDist2(t,k,th,mu)
print(t)
print(y)
param = getGammaPars(t,y,dx)
print(param)

ycfit = gammaDist2(t,param[0],param[1],param[2])

fig, ax = plt.subplots()

ax.plot(t, y, linewidth=2.0)
ax.plot(t, ycfit, linewidth=1.0)

plt.show()

xmin = stats.gamma.ppf(1e-6,k,loc = mu*coeff, scale = th*coeff) - 0.1
xmax = stats.gamma.ppf(1-1e-6,k,loc = mu*coeff, scale = th*coeff)
dx = (xmax-xmin)/100000

t = np.arange(xmin,xmax,dx)

y2 = gammaDist2(t,k,th*coeff,mu*coeff)
param = getGammaPars(t,y2,dx)
print(param)

ycfit2 = gammaDist2(t,param[0],param[1],param[2])

fig, ax = plt.subplots()

ax.plot(t, y2, linewidth=2.0)
ax.plot(t, ycfit2, linewidth=1.0)

plt.show()
