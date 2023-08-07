import numpy as np
import math 
import random
from scipy.special import gamma, psi, polygamma
import scipy.stats as stats
import matplotlib.pyplot as plt
from progress.bar import Bar
import pandas as pd
import random

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

k1_k = 1
k1_th = 3.51931
th1_k = 1
th1_th = 127.895

k2_k = 7.28
k2_th = 2.69
th2_k = 1.48
th2_th = 1.20

# k1min = 0.37
# th1min = 0.64
# mu1min = -5337.18
# k1max = 7628.94
# th1max = 11882.40
# mu1max = 22.26

# k2min = 9.64
# th2min = 0.42
# mu2min = -172.37
# k2max = 74.93
# th2max = 5.13
# mu2max = -9.55

df1 = pd.read_excel("2DFIT.xlsx",sheet_name='G1') 
df2 = pd.read_excel("2DFIT.xlsx",sheet_name='G2') 

file1 = open("exports30.txt", "a")  # append mode
file2 = open("logging3.txt", "a")  # append mode
n = 15000

with Bar('Processing:', max=n, width=50, fill='█', suffix='%(index)d/%(max)d - %(percent).1f%% - Elapsed: %(elapsed)ds - ETA: %(eta)ds' ) as bar:

    for i in range(n):
        
        # k1 = stats.gamma.rvs(k1_k, loc=0, scale=k1_th, size=1, random_state=None)[0]
        # th1 = stats.gamma.rvs(th1_k, loc=0, scale=th1_th, size=1, random_state=None)[0]
        # k2 = stats.gamma.rvs(k2_k, loc=0, scale=k2_th, size=1, random_state=None)[0]
        # th2 = stats.gamma.rvs(th2_k, loc=0, scale=th2_th, size=1, random_state=None)[0]
        i1 = random.randint(0, 100000)        
        i2 = random.randint(0, 100000)

        k1 = df1['k'][i1]
        th1 = df1['th'][i1]
        k2 = df2['k'][i2]
        th2 = df2['th'][i2]
        
        #print(f"Current element: {k1}, {th1}, {k2}, {th2}")
    
        xmin1 = stats.gamma.ppf(1e-7,k1,loc = 0, scale = th1)
        xmax1 = stats.gamma.ppf(1-1e-7,k1,loc = 0, scale = th1)
        
        xmin2 = stats.gamma.ppf(1e-7,k2,loc = 0, scale = th2)
        xmax2 = stats.gamma.ppf(1-1e-7,k2,loc = 0, scale = th2)
        
        xmin = 0
        xmax = xmax1 + xmax2
        
        dx = (xmax - xmin) / 5000
    
        t = np.arange(xmin,xmax,dx)
    
        try:
        
            y = gammaDist(t,k1,th1)
            y2 = gammaDist(t,k2,th2)
        
            yc = convolve(t,dx,y2,y)
            param = getGammaPars(t,yc,dx)
            #print(param)
            ycfit = gammaDist2(t,param[0],param[1],param[2])
        
            fiterror = getFitError(yc,ycfit)
        
            file1.write(f"{k1}\t{th1}\t{k2}\t{th2}\t{param[0]}\t{param[1]}\t{param[2]}\t{fiterror}\n")
            file1.flush()
        
            bar.next()
    
        except Exception as err:
            print(f"\nError\t{k1}\t{th1}\t{k2}\t{th2}\t{err=}\t{type(err)=}")
            file2.write(f"\nError\t{k1}\t{th1}\t{k2}\t{th2}\t{err=}\t{type(err)=}")
            file2.flush()

file1.close()
file2.close()
