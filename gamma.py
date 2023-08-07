import numpy as np
import math 
from scipy.special import gamma, psi, polygamma
import scipy.stats as stats
import matplotlib.pyplot as plt

def gammaDist(x,k,th):
    y = []
    for i in x:
        y.append(stats.gamma.pdf(i,k,loc = 0, scale = th))
    return y

def gammaDist2(x,k,th,mu):
    y = []
    for i in x:
        y.append(stats.gamma.pdf(i,k,loc = mu, scale = th))
    return y
    
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

k1 = float(input('k1: '))
th1 = float(input('th1: '))
mu1 = float(input('mu1: '))
k2 = float(input('k2: '))
th2 = float(input('th2: '))
mu2 = float(input('mu2: '))

xmin1 = max(stats.gamma.ppf(1e-7,k1,loc = 0, scale = th1) - 0.1,0)
xmax1 = stats.gamma.ppf(1-1e-7,k1,loc = 0, scale = th1)

xmin2 = max(stats.gamma.ppf(1e-7,k2,loc = 0, scale = th2) - 0.1,0)
xmax2 = stats.gamma.ppf(1-1e-7,k2,loc = 0, scale = th2)

xmin = 0
xmax = xmax1 + xmax2
dx = (xmax - xmin) / 10000
    
t = np.arange(xmin,xmax,dx)

y = gammaDist(t,k1,th1)
print(getGammaPar(t,y,dx))

y2 = gammaDist(t,k2,th2)
print(getGammaPar(t,y2,dx))

# yc = convolve(t,dx,y2,y)
# param = getGammaPar(t,yc,dx)
# print(param)
# ycfit = gammaDist(t,param[0],param[1])

yc = convolve(t,dx,y2,y)
param = getGammaPars(t,yc,dx)
print(param)
ycfit = gammaDist2(t,param[0],param[1],param[2])

plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(figsize=(7,5))

ax.plot(t, y, "C0", linewidth=2.0, label="Input 1")
ax.plot(t, y2, "C1", linewidth=2.0, label="Input 2")
ax.plot(t, yc, "C2", linewidth=2.0, label="Convolution")
ax.plot(t, ycfit, "C3", linewidth=2.0, label="Fitted gamma")
plt.xlim((0,1200))
plt.xlabel("x")
plt.ylabel("P(x)")
#plt.title("The inputs, their convolution and the fitted gamma function")
plt.legend()

plt.savefig("conv21.png",pad_inches=0)
plt.show()

fig, ax = plt.subplots(figsize=(7,5))

ax.plot(t, yc, "C2", linewidth=2.0, label="Convolution")
ax.plot(t, ycfit, "C3", linewidth=2.0, label="Fitted gamma")
plt.xlim((0,1200))
plt.xlabel("x")
plt.ylabel("P(x)")
#plt.title("The convoluted function and the fitted gamma function")
plt.legend()

plt.savefig("conv22.png",pad_inches=0)
plt.show()

print(f'Error:{getFitError(yc,ycfit)}')