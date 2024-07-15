from math import *                  # 导入math模块
import random                       # 导入random模块
import pandas as pd                 # 导入pandas模块命名为pd
import numpy as np                  # 导入numpy模块命名为np
import matplotlib.pyplot as plt     # 导入matplotlib.pyplot模块命名为plt

####参数
#振子 浮子 弹簧的参数
m1=4866# 浮子质量 (kg)
m2=2433# 振子质量 (kg)
m3=1091.099# 垂荡附加质量 (kg)
r1=1# 浮子底半径 (m)
r2=0.5# 振子半径 (m)
h11=3# 浮子圆柱部分高度 (m)
h12=0.8# 浮子圆锥部分高度 (m)
h2=0.5# 振子高度 (m)
F_gangdu1=80000# 弹簧刚度 (N/m)
l_=0.5# 弹簧原长 (m)
l0=l_-m2/F_gangdu1#弹簧初始长度
rho=1025# 海水的密度 (kg/m3)
g=9.8# 重力加速度 (m/s2)

#系数
k1=528.5018# 垂荡兴波阻尼系数 (N·s/m)
k2=1655.909# 纵摇兴波阻尼系数 (N·m·s)
k3=8890.7# 静水恢复力矩系数 (N·m)
k4=250000# 扭转弹簧刚度 (N·m)

#转动惯量
I1=31961#浮子的转动惯量(kg·m2)
I3=7142.493# 浮子的附加转动惯量 纵摇附加转动惯量 (kg·m2)

f=1760# 垂荡激励力振幅 (N)
omega=1.9806# 入射波浪频率 (s-1)
T=2*pi/omega #周期
dt=0.002# 时间步长
L=2140# 纵摇激励力矩振幅 (N·m)
V0=(m1+m2)/rho# 初始排水体积
x_1 = 2.177
t=np.arange(0,20*T,dt) #时间
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
P=np.zeros(n, dtype=float)
P1=np.zeros(n, dtype=float)

# 设置基本参数
def init_parameter():
    t_init = 100            # 初始退火温度
    t_final = 0             # 终止退火温度
    n_1 = 100               # 内循环运行次数
    return t_init,t_final,n_1

# Metropolis准则判断是否接受新的阻尼系数
def Metropolis(curr_power,prev_power,best_power,prev_x,curr_x,best_x,t_now):
    # dE 新解与原解的差值
    dE = curr_power - prev_power
    # 新的阻尼系数对应的平均功率大于当前解，接受新解
    if dE > 0:
        accept = True
        # 新的阻尼系数对应的平均功率值大于最大平均功率，将新的阻尼系数保存为最优解
        if curr_power > best_power:
            best_x[:] = curr_x[:]
            best_power = curr_power
    # 新的阻尼系数对应的平均功率小于当前平均功率，以一定概率接受新的阻尼系数
    else:
        # 依据Metropolis准则计算接受的概率
        p_accept = exp(-dE / t_now)
        if p_accept > random.random():
            accept = True
        else:
            accept = False
    # 接受新的阻尼系数，将新解保存为当前解
    if accept == True:
        prev_x[:] = curr_x[:]
        prev_power = curr_power
    return prev_power,prev_x,best_power,best_x

#生成新的决策变量
def random_new(k):#k为决策变量列表
    if k[0]<=4000:
        k1 = k[0] + random.uniform(0, 2000)
    elif k[0]>=96000:
        k1 = k[0] + random.uniform(-2000, 0)
    else:
        k1=k[0]+random.uniform(-2000, 2000)

    if k[1]<=4000:
        k2 = k[1] + random.uniform(0, 2000)
    elif k[1]>=96000:
        k2 = k[1] + random.uniform(-2000, 0)
    else:
        k2=k[1]+random.uniform(-2000, 2000)
    return [k1,k2]

# 设置基本参数
t_init,t_final,n_1= init_parameter()
# 初始化参数（当前、最优）
v1[0]=0# 浮子初始位移
x1[0]=0# 浮子初始速度
v2[0]=0# 振子初始位移
x2[0]=0# 振子初始速度
omega1[0]=0# 浮子初始角速度
theta1[0]=0# 浮子初始角位移
omega2[0]=0# 振子初始角速度
theta2[0]=0# 振子初始角位移
t_now  = t_init    #初始化 当前温度
#初始化当前决策参数(比例系数[0,100000] 幂指数 [0,1] )
prev_x=[50000,50000]
#初始化最优决策参数
best_x=prev_x.copy()
#初始化当前平均输出功率(目标函数)
P[0]=0
P1[0]=0
prev_power=P[0]
#初始化最大平均输出功率(目标函数)
best_power=prev_power
#不同温度下的最大平均功率记录
best_power_record = [0]
# 模拟退火
aa=0
# 终止条件
while t_now >= t_final:
    # 当前温度下，寻找最优阻尼系数
    for k in range(n_1):
        # 随机产生新的阻尼系数
        curr_x=random_new(prev_x)
        alpha1 = curr_x[0]  # 直线阻尼器的阻尼系数N·s/m
        alpha2 = curr_x[1]  # 旋转阻尼器的阻尼系数N·m·s
        #计算目标函数average power
        for i in range(1, n):
            z = 2.8 - x1[i - 1]
            V = (49 / 15 - 3.8 + z / cos(theta1[i - 1])) * pi
            F_1 = rho * g * (V - V0)  # 静水恢复力
            F_2 = k * (l[i - 1] - l0)  # 弹力
            F_3 = -alpha1 * (v1[i - 1] * cos(theta2[i - 1]) - v2[i - 1])  # 阻尼器阻力
            F_4 = -k1 * v1[i - 1]  # 兴波阻尼力
            F_5 = f * cos(omega * i * dt)  # 波浪激励力
            F_6 = m2 * g * (1 - cos(theta2[i - 1]))  # 重力变化
            F_h1 = F_1 + (F_2 + F_3) * cos(theta1[i - 1]) + F_4 + F_5  # 浮子受力
            F_h2 = -F_2 - F_3 + F_6  # 振子受力
            v1[i] = dt / (m1 + m3) * F_h1 + v1[i - 1]  # 浮子速度
            v2[i] = dt / m2 * F_h2 + v2[i - 1]  # 振子速度
            x1[i] = 0.5 * (v1[i] + v1[i - 1]) * dt + x1[i - 1]  # 浮子位移
            x2[i] = 0.5 * (v2[i] + v2[i - 1]) * dt + x2[i - 1]  # 振子位移

            x_2 = 0.25 + l[i - 1]  # x2
            M1 = -k3 * theta1[i - 1]  # 静水恢复力矩
            M2 = -k4 * (theta1[i - 1] - theta2[i - 1])  # 扭转弹簧力矩
            M3 = -alpha2 * (omega1[i - 1] - omega2[i - 1])  # 选择阻尼器力矩
            M4 = -k2 * omega1[i - 1]  # 兴波阻尼力矩
            M5 = L * cos(omega * i * dt)  # 波浪激励力矩
            M6 = m1 * g * x_1 * sin(theta1[i - 1])  # * sgn(theta1[i - 1])  # 浮子重力矩
            M7 = m2 * g * x_2 * sin(theta2[i - 1])  # * sgn(theta2[i - 1])  # 振子重力矩
            I2 = m2 * (0.25 + l[i - 1]) ** 2 + m2 / 12 * (3 * 0.25 + 0.25)  # 振子的转动惯量
            omega1[i] = dt / (I1 + I3) * (M1 + M2 + M3 + M4 + M5 + M6)  # 浮子角速度
            omega2[i] = dt / I2 * (-M2 - M3 + M7)  # 振子角速度
            theta1[i] = omega1[i] * dt + theta1[i - 1]  # 浮子角位移
            theta2[i] = omega2[i] * dt + theta2[i - 1]  # 振子角位移
            l[i] = l[i - 1] + (x2[i] - x2[i - 1]) - (
                        (x1[i] - x1[i - 1]) + 0.8 * (cos(theta1[i]) - cos(theta1[i - 1]))) / cos \
                       (theta2[i - 1])  # 弹簧长度
            P[i] = abs(F_3 * ((x1[i] - x1[i - 1]) - (x2[i] - x2[i - 1])))
            P1[i] = abs(M3 * ((theta1[i] - theta1[i - 1]) - (theta2[i] - theta2[i - 1])))

        P_aver=sum(P)/(20 * T)
        P1_aver = sum(P1)/(20 * T)
        P_P1_aver = sum(P + P1)/(20 * T)
        print('P_aver',P_aver,P1_aver,P_P1_aver,curr_x)

        #判断是否接受新的阻尼系数
        curr_power = P_aver
        prev_power,prev_x,best_power,best_x=Metropolis(curr_power, prev_power, best_power,prev_x, curr_x, best_x, t_now)
    # 结束在当前温度的搜索，更新最大平均功率列表
    best_power_record.append(best_power)         # 本次温度下的最大输出功率加到最大功率记录表
    t_now = t_now -1
print(max(best_power_record))
print(best_x)
