import numpy as np                  # 导入模块 numpy 并简写成 np
import pandas as pd                 # 导入模块 pandas 并简写成 pd
import matplotlib.pyplot as plt     # 导入模块 matplotlib.pyplot 并简写成 plt
from math import *                  # 导入模块 math
import random                       # 导入模块 random

####参数
w=1.4005# 入射波浪频率 (s-1)
f=6250# 垂荡激励力振幅 (N)
dt=0.02# 时间步长
T=2*pi/w #周期
m3=1335.535# 垂荡附加质量 (kg)
k1=656.3616# 垂荡兴波阻尼系数 (N·s/m)
m1=4866# 浮子质量 (kg)
r1=1# 浮子底半径 (m)
h11=3# 浮子圆柱部分高度 (m)
h12=0.8# 浮子圆锥部分高度 (m)
m2=2433# 振子质量 (kg)
r2=0.5# 振子半径 (m)
h2=0.5# 振子高度 (m)
rho=1025# 海水的密度 (kg/m3)
g=9.8# 重力加速度 (m/s2)
F_gangdu1=80000# 弹簧刚度 (N/m)
L=0.5# 弹簧原长 (m)
S=pi*r1**2# 浮子横截面积
zeta=10000# 阻尼系数

t=np.arange(0,40*T,dt) #时间
n=len(t)
v1=np.zeros(n, dtype=float)
x1=np.zeros(n, dtype=float)
v2=np.zeros(n, dtype=float)
x2=np.zeros(n, dtype=float)
v1[0]=0# 浮子初始位移
x1[0]=0# 浮子初始速度
v2[0]=0# 振子初始位移
x2[0]=0# 振子初始速度

for i in range(1,n):
    F_1 = -rho * g * (S * x1[i-1])  # 静水恢复力
    F_2 = -F_gangdu1 * (x1[i-1] - x2[i-1])  # 弹力
    F_3 = -zeta * (v1[i-1] - v2[i-1])  # 阻尼器阻力
    F_4 = -k1 * v1[i-1]  # 兴波阻尼力
    F_5 = f * cos(w * i*dt)  # 波浪激励力
    F_h1 = F_1 + F_2 + F_3 + F_4 + F_5  # 浮子受力
    F_h2 = -F_2 - F_3  # 振子受力
    v1[i]=dt/(m1+m3)*F_h1+v1[i-1]#浮子速度
    v2[i] = dt / m2 * F_h2 + v2[i - 1]#振子速度
    x1[i]=0.5*(v1[i]+v1[i-1])*dt+x1[i-1]#浮子位移
    x2[i]=0.5*(v2[i]+v2[i-1])*dt+x2[i-1]#振子位移

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.figure()
plt.plot(t,v2,label='速度（m/s）')
plt.plot(t,x2,label='位移（m）')
plt.xlabel('时间（s）')
plt.title('振子速度和位移变化')
plt.legend()

plt.figure()
plt.plot(t,v1,label='速度（m/s）')
plt.plot(t,x1,label='位移（m）')
plt.xlabel('时间（s）')
plt.title('浮子速度和位移变化')
plt.legend()

plt.figure()
plt.subplot(2,2,1)
plt.plot(t,v1,color='blue',linewidth=1.0)
plt.xlabel('时间（s）')
plt.ylabel('速度（m/s）')
plt.title('浮子速度变化')

plt.subplot(2,2,2)
plt.plot(t,x1,color='blue',linewidth=1.0)
plt.xlabel('时间（s）')
plt.ylabel('浮子位移（m）')
plt.title('浮子位移变化')

plt.subplot(2,2,3)
plt.plot(t,v2,color='blue',linewidth=1.0)
plt.xlabel('时间（s）')
plt.ylabel('速度（m/s）')
plt.title('振子速度变化')

plt.subplot(2,2,4)
plt.plot(t,x2,color='blue',linewidth=1.0)
plt.xlabel('时间（s）')
plt.ylabel('振子位移变化（m）')
plt.title('振子位移变化')

plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.5,hspace=0.5)
plt.show()