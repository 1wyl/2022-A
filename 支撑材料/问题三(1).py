from math import *                  # 导入math模块
import random                       # 导入random模块
import pandas as pd                 # 导入pandas模块命名为pd
import numpy as np                  # 导入numpy模块命名为np
import matplotlib.pyplot as plt     # 导入matplotlib.pyplot模块命名为plt

####参数
#振子 浮子 弹簧的参数
rho=1025# 海水的密度 (kg/m3)
g=9.8# 重力加速度 (m/s2)
m1=4866# 浮子质量 (kg)
m2=2433# 振子质量 (kg)
m3=1028.876# 垂荡附加质量 (kg)
r1=1# 浮子底半径 (m)
r2=0.5# 振子半径 (m)
h11=3# 浮子圆柱部分高度 (m)
h12=0.8# 浮子圆锥部分高度 (m)
h2=0.5# 振子高度 (m)
k=80000# 弹簧刚度 (N/m)
l00=0.5# 弹簧原长 (m)
l0=l00-m2*g/k#弹簧初始长度
#参数
alpha1=10000#直线阻尼器的阻尼系数N·s/m
alpha2=1000 #旋转阻尼器的阻尼系数N·m·s
k1=683.4558# 垂荡兴波阻尼系数 (N·s/m)
k2=654.3383# 纵摇兴波阻尼系数 (N·m·s)
k3=8890.7# 静水恢复力矩系数 (N·m)
k4=250000# 扭转弹簧刚度 (N·m)
I1=31961#浮子的转动惯量(kg·m2)
I3=7001.914# 浮子的附加转动惯量(kg·m2)# 纵摇附加转动惯量

#f=3640# 垂荡激励力振幅 (N)
f=4640# 垂荡激励力振幅 (N)

omega=1.7152# 入射波浪频率 (s-1)
T=2*pi/omega #周期
dt=0.002# 时间步长
#L=1690# 纵摇激励力矩振幅 (N·m)
L=1990# 纵摇激励力矩振幅 (N·m)

V0=(m1+m2)/rho# 初始排水体积
x_1 = 2.177
t=np.arange(0,40*T,dt) #时间
n=len(t)
v1=np.zeros(n, dtype=float)
x1=np.zeros(n, dtype=float)
v2=np.zeros(n, dtype=float)
x2=np.zeros(n, dtype=float)
omega1=np.zeros(n, dtype=float)
theta1=np.zeros(n, dtype=float)
omega2=np.zeros(n, dtype=float)
theta2=np.zeros(n, dtype=float)
l=np.zeros(n, dtype=float)
v1[0]=0# 浮子初始速度
x1[0]=0# 浮子初始位移
v2[0]=0# 振子初始速度
x2[0]=0# 振子初始位移
l[0]=l0# 弹簧初始长度
omega1[0]=0# 浮子初始角速度
theta1[0]=0# 浮子初始角位移
omega2[0]=0# 振子初始角速度
theta2[0]=0# 振子初始角位移


for i in range(1,n):
    z = 2.8 - x1[i - 1]
    V=(49/15-3.8+z/cos(theta1[i-1]))*pi
    F_1 = rho * g * (V - V0)  # 静水恢复力
    F_2 = k * (l[i - 1] - l0)  # 弹力
    F_3 = -alpha1 * (v1[i - 1] * cos(theta2[i - 1]) - v2[i - 1])  # 阻尼器阻力
    F_4 = -k1 * v1[i - 1]  # 兴波阻尼力
    F_5 = f * cos(omega * i * dt)  # 波浪激励力
    F_6 = m2 * g * (1 - cos(theta2[i - 1]))#重力变化
    F_h1 = F_1 + (F_2 + F_3)*cos(theta1[i-1]) + F_4 + F_5  # 浮子受力
    F_h2 = -F_2 - F_3 + F_6  # 振子受力
    v1[i] = dt / (m1 + m3) * F_h1 + v1[i - 1]  # 浮子速度
    v2[i] = dt / m2 * F_h2 + v2[i - 1]  # 振子速度
    x1[i] = 0.5 * (v1[i] + v1[i - 1]) * dt + x1[i - 1]  # 浮子位移
    x2[i] = 0.5 * (v2[i] + v2[i - 1]) * dt + x2[i - 1]  # 振子位移
    x_2 = 0.25 + l[i-1]#x2
    M1 = -k3 * theta1[i - 1]  # 静水恢复力矩
    M2 = -k4 * (theta1[i - 1] - theta2[i - 1])  # 扭转弹簧力矩
    M3 = -alpha2 * (omega1[i - 1] - omega2[i - 1])  # 选择阻尼器力矩
    M4 = -k2 * omega1[i-1]  # 兴波阻尼力矩
    M5 = L * cos(omega * i * dt)  # 波浪激励力矩
    M6 = m1 * g * x_1 * sin(theta1[i - 1]) #* sgn(theta1[i - 1])  # 浮子重力矩
    M7 = m2 * g * x_2 * sin(theta2[i - 1]) #* sgn(theta2[i - 1])  # 振子重力矩
    I2 = m2 * (0.25 + l[i-1]) ** 2 + m2 / 12 * (3 * 0.25 + 0.25)  # 振子的转动惯量
    omega1[i]=dt/(I1+I3)*(M1+M2+M3+M4+M5+M6)#浮子角速度
    omega2[i]=dt/I2*(-M2-M3+M7)#振子角速度
    theta1[i]=(omega1[i]+omega1[i-1])/2*dt + theta1[i-1]#浮子角位移
    theta2[i] = (omega2[i]+omega2[i-1])/2 * dt + theta2[i - 1]#振子角位移
    l[i]=l[i-1]+(x2[i]-x2[i-1])-((x1[i]-x1[i-1])+0.8*(cos(theta1[i])-cos(theta1[i-1])))/cos\
            (theta2[i - 1])#弹簧长度
    if i==5000 or i==10000 or i==20000 or i==30000 or i==50000:
        print(x2[i],v2[i],theta2[i],omega2[i])

plt.figure()
plt.plot(t,omega2)
plt.plot(t,theta2)
plt.figure()
plt.plot(t,omega1)
plt.plot(t,theta1)
plt.figure()
plt.plot(t,v1)
plt.plot(t,x1)
plt.figure()
plt.plot(t,v2)
plt.plot(t,x2)

plt.show()


