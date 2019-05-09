#this file is for scheduling by Round-Robin Scheduling
#N:the number of server
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
N = 30
M = 20
'''data form:
columns:id, arrival time,brust time, priority(scheduling class)
'''
#visualization
# server_name= [i for i in range(N)]
# server_task = [M for i in range(N)]
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# plt.ion()
# plt.show()


#generate data
sample_num = 10000
feature_num = 4
data = np.load('data.npy')
# def RR (data, total_server_number, capacity,graph):
    # current_time = 0
    # loss = 0
    # remind_resource = {}
    # for i in range(total_server_number):
    #     remind_resource[i]=[]
    #
    # for i in range(data.shape[0]):
    #     remind_resource[i%total_server_number].append([data[i][1],data[i][2]])
    # for i in range(total_server_number):
    #     print(len(remind_resource[i]))
    # while True:
    #     if graph:
    #         if current_time%2000==0:
    #             print("current_time",current_time)
    #             for i in range(total_server_number):
    #                 server_task[i] = len(remind_resource[i])
    #             try:
    #                 ax.lines.remove(lines[0])
    #             except Exception:
    #                 pass
    #             lines = ax.plot(server_name,server_task,color='red')
    #             plt.pause(0.1)
    #     for i in range(total_server_number):
    #         if len(remind_resource[i]) < capacity:
    #             j = 0
    #             while j < len(remind_resource[i]):
    #                 if remind_resource[i][j][0] + remind_resource[i][j][1] <= current_time:
    #                     remind_resource[i].pop(j)
    #                     j -= 1
    #                 j += 1
    #         else:
    #             del_num = 0
    #             k = 0
    #             try:
    #                 while k < capacity:
    #                     if remind_resource[i][k][0] + remind_resource[i][k][1] <=current_time:
    #                         del_num += 1
    #                         remind_resource[i].pop(k)
    #                         k -= 1
    #                     k += 1
    #             except:
    #                 continue
    #             for j in range(capacity-del_num,len(remind_resource[i])):
    #                 if remind_resource[i][j][0] < current_time:
    #                     remind_resource[i][j][1] += (current_time - remind_resource[i][j][0])
    #     flag = False
    #     for i in range(total_server_number):
    #         if len(remind_resource[i]) != 0:
    #             flag = True
    #             break
    #     if not flag:
    #         break
    #     current_time +=1
    # print(current_time)

def NewRR(data, total_server_number,capacity):
    res = np.array([])
    #capacity = np.full(total_server_number,capacity)
    capacity = np.load('m.npy')
    servering = np.full(total_server_number, 0)
    resouce = {}
    for i in range(total_server_number):
        resouce[i] =[]
    index = 0
    currenttime = 0
    while index < sample_num:
        while index < sample_num and data[index][1] <= currenttime:
            #release the resource for currenttime
            for i in range(total_server_number):
                servering[i] = len(resouce[i])
                mintask = min(capacity[i],servering[i])
                delNum =0
                j = 0
                while j < mintask - delNum:
                    if resouce[i][j][1] + resouce[i][j][2] <=currenttime:
                        resouce[i].pop(j)
                        servering[i] -= 1
                        delNum +=1
                        j -= 1
                    j += 1
            resouce[index % total_server_number].append(data[index])
            servering[index % total_server_number] += 1
            index += 1
        currenttime += 1
        print(currenttime)
        res = np.append(res,servering)
    res = res.reshape(-1,total_server_number)
    print(res)
    np.save('RR.npy',res)



NewRR(data,N,M)

#
# RR(data=data, total_server_number=N, capacity=M,graph=True)
# def Weighted_RR(data, total_server_number, capacity):
#     pass
#
