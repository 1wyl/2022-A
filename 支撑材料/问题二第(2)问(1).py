from math import *                  # 导入math模块
import random                       # 导入random模块
import pandas as pd                 # 导入pandas模块命名为pd
import numpy as np                  # 导入numpy模块命名为np
import matplotlib.pyplot as plt     # 导入matplotlib.pyplot模块命名为plt

#参数
w=2.2143# 入射波浪频率 (s-1)
f=4890# 垂荡激励力振幅 (N)
T=2*pi/w #周期
dt=0.02# 时间步长
m3=1165.992# 垂荡附加质量 (kg)
k1=167.8395# 垂荡兴波阻尼系数 (N·s/m)
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
F_gangdu2=250000# 扭转弹簧刚度 (N·m)
S=pi*r1**2# 浮子横截面积

t=np.arange(0,40*T,dt) #时间
n=len(t)
v1=np.zeros(n, dtype=float)
x1=np.zeros(n, dtype=float)
v2=np.zeros(n, dtype=float)
x2=np.zeros(n, dtype=float)
P=np.zeros(n, dtype=float)
zetazeta=np.zeros(n, dtype=float)

# 初始化控制参数
def init_parameter():
    t_init = 100            # 初始退火温度
    t_final = 0             # 终止退火温度
    n_1 = 100               # 内循环运行次数
    return t_init,t_final,n_1

# 按照 Metropolis 准则决定是否接受新的阻尼系数
def Metropolis(curr_power,prev_power,best_power,prev_x,curr_x,best_x,tNow):
    # dE 新解与原解的差值
    dE = curr_power - prev_power
    # 如果新的阻尼系数对应的平均功率好于当前解，则接受新的阻尼系数
    if dE > 0:
        accept = True
        # 如果新的阻尼系数的目标函数好于最优解，则将新解保存为最优解
        if curr_power > best_power:
            best_x[:] = curr_x[:]
            best_power = curr_power
    # 如果的阻尼系数的目标函数比当前解差，则以一定概率接受新的阻尼系数
    else:
        # 按照Metropolis 判断是否接受新的阻尼系数
        p_accept = exp(-dE / tNow)
        if p_accept > random.random():
            accept = True
        else:
            accept = False
    # 接受新的阻尼系数，并将新解保存为当前解
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

    if k[1]<=0.04:
        k2 = k[1] + random.uniform(0, 0.02)
    elif k[1]>=0.96:
        k2 = k[1] + random.uniform(-0.02, 0)
    else:
        k2=k[1]+random.uniform(-0.02, 0.02)
    return [k1,k2]

# 设置控制参数
t_init,t_final,n_1= init_parameter()
# 初始化参数（当前、最优）
v1[0]=0# 浮子初始位移
x1[0]=0# 浮子初始速度
v2[0]=0# 振子初始位移
x2[0]=0# 振子初始速度

t_now  = t_init    #初始化 当前温度
#初始化当前决策参数(比例系数[0,100000] 幂指数 [0,1] )
prev_x=[50000,0.5]
#初始化最优决策参数
best_x=prev_x.copy()
#初始化当前平均输出功率(目标函数)
P[0]=0
prev_power=P[0]
#初始化最大平均输出功率(目标函数)
best_power=prev_power
#不同温度下的最大平均功率记录
best_power_record = []
best_k=[0,0]
# 模拟退火
# 终止条件
while t_now >= t_final:
    # 当前温度下，求出最大输出功率
    for k in range(n_1):
        # 随机产生新的阻尼系数
        curr_x=random_new(prev_x)
        #计算目标函数average power
        for i in range(1, n):
            zeta1=curr_x[0]*abs(v1[i - 1] - v2[i - 1])**curr_x[1]
            zetazeta[i] = zeta1
            F_1 = -rho * g * (S * x1[i - 1])  # 静水恢复力
            F_2 = -F_gangdu1 * (x1[i - 1] - x2[i - 1])  # 弹力
            F_3 = -zeta1 * (v1[i - 1] - v2[i - 1])  # 阻尼器阻力
            F_4 = -k1 * v1[i - 1]  # 兴波阻尼力
            F_5 = f * cos(w * i * dt)  # 波浪激励力
            F_h1 = F_1 + F_2 + F_3 + F_4 + F_5  # 浮子受力
            F_h2 = -F_2 - F_3  # 振子受力
            v1[i] = dt / (m1 + m3) * F_h1 + v1[i - 1]  # 浮子速度
            v2[i] = dt / m2 * F_h2 + v2[i - 1]  # 振子速度
            x1[i] = 0.5 * (v1[i] + v1[i - 1]) * dt + x1[i - 1]  # 浮子位移
            x2[i] = 0.5 * (v2[i] + v2[i - 1]) * dt + x2[i - 1]  # 振子位移
            P[i] = abs(F_3 * ((x1[i] - x1[i - 1]) - (x2[i] - x2[i - 1])))
        P_aver=np.mean(P)
        print('P_aver',P_aver,curr_x)
        #判断是否接受新的阻尼系数
        curr_power = P_aver
        prev_power,prev_x,best_power,best_x=Metropolis(curr_power, prev_power, best_power,prev_x, curr_x, best_x, t_now)

    # 完成当前温度的搜索，更新最优解列表
    best_power_record.append(best_power)                # 将本次温度下的最大平均功率加入记录表
    t_now = t_now -5
print(max(best_power_record))
print(best_x)

itter=[]

for i in range(1,22):
    itter.append(i)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.figure()
plt.plot(itter,best_power_record)

plt.xlabel('迭代次数')
plt.ylabel('最优解')
plt.title('模拟退火求最优解过程')
plt.show()

