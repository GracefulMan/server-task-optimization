import matplotlib.pyplot as plt
import numpy as np
N = 30
M = 20
server_name = [i for i in range(N)]
q_learning_data = np.load('Q_learn.npy')
sarsa = np.load('sarsa.npy')
sarsalambda = np.load('saraslambda.npy')
rr = np.load('RR.npy')
dnn =np.load('dnn.npy')
fig = plt.figure()
index = 0
plt.ylim((0,100))
plt.title("N=30")
const = np.load('m.npy')
plt.plot(server_name,const,linestyle='-.',color='coral')
plt.xlabel("server")
plt.ylabel("current task")
ax = fig.add_subplot(1,1,1)
plt.ion()
plt.show()
total_line = q_learning_data.shape[0]
while index < total_line:
    try:
        l.remove()
        lw.remove()
        lw2.remove()
        lw3.remove()
        lw4.remove()

    except Exception:
        pass
    lines = ax.plot(server_name,q_learning_data[index],'r')
    sarsalines = ax.plot(server_name,sarsa[index],'b')
    saraslambdalines = ax.plot(server_name,sarsalambda[index],'g')
    rrlines = ax.plot(server_name,rr[index],'y')
    dnnlines = ax.plot(server_name,dnn[index],'black')
    plt.legend(['capacity','Q_learning','sarsa','sarsa lambda(0.5)','Round-Robin','DQN'])
    l = lines.pop(0)
    lw = sarsalines.pop(0)
    lw2 = saraslambdalines.pop(0)
    lw3 = rrlines.pop(0)
    lw4 = dnnlines.pop(0)
    plt.pause(0.05)
    index += 1
