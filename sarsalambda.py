
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
np.set_printoptions(suppress=True)
N = 30
M = 20
#generate data
sample_num = 10000
feature_num = 4
data = np.ones([sample_num, feature_num])
data[:,0] = np.arange(sample_num).reshape(sample_num, )
data[:,1] = np.sort(np.random.randint(0,int(sample_num/10),sample_num)).reshape(sample_num, )
data[:,2] = np.random.randint(1, 100, sample_num).reshape(sample_num, )
data[:,3] = np.random.randint(0, 4, sample_num).reshape(sample_num, )
class SarsaLambdaTable:
    def __init__(self, serverNum, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,trace_decay = 0.5):
        actions=[i for i in range(serverNum)]
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.eligibility_trace = self.q_table.copy()
        self.lambda_ = trace_decay

    def choose_action(self, observation):#传入servering数组
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation,:]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action #返回selected_server

    def learn(self,flag,s,a,r,s_,a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if not flag :
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict
        #method1
        #self.eligibility_trace.loc[s, a] +=1

        #method 2
        self.eligibility_trace.loc[s, :]*=0
        self.eligibility_trace.loc[s, a] = 1

        self.q_table += self.lr * error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )

            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)


class Env:
    #状态：各个服务器的资源分配状态，保持各个服务器尽可能无排队现象以及无极度空闲的状态为目标
    def __init__(self,init_server_number,capacity):
        self.number = init_server_number
        self.campacity = capacity
        self._servering = np.zeros(init_server_number)
        self.resouce_queue = {}
        for i in range(init_server_number):
            self.resouce_queue[i] = []
    def reward(self,data,selected_server,show_time):#selected_server indicting the index of server
        #first, clear the task which had finished.
        value = 0
        current_time = data[1]
        # if show_time%50==0:
        #     try:
        #         ax.lines.remove(lines[0])
        #     except Exception:
        #         pass
        #     server_name = [i for i in range(self.number)]
        #     lines = ax.plot(server_name, self._servering, color='red')
        #     plt.pause(0.5)
        for i in range(self.number):
            j = 0
            del_num = 0
            current_task_num = self._servering[i]
            index = min(current_task_num, self.campacity[i])
            while j < index - del_num:
                if self.resouce_queue[i][j][1] + self.resouce_queue[i][j][2] <=current_time:
                    #if len(self.resouce_queue[i]) > index:
                       # value += (self.resouce_queue[i][index][1] + self.resouce_queue[i][index][2]- current_time) * self.resouce_queue[i][index][3]
                    self.resouce_queue[i].pop(j)
                    self._servering[i] -= 1
                    del_num += 1
                    j -= 1
                j += 1

        if self._servering[selected_server] < self.campacity[selected_server]:
            value += data[3]
        else:
            value -= (self._servering[selected_server] - self.campacity[selected_server]) *data[3]
            for i in range(self.number):
                if i != selected_server and self._servering[i] < self.campacity[i]:
                    value -= (self.campacity[i] - self._servering[i])
        #create environment,transfer vector into a number.
        # print('---------------------------------------------------------ks')
        # print(selected_server,'\t',value)
        # print(self._servering)
        # print(self.campacity)
        # print('---------------------------------------------------------------------------js')
        tempEnv =np.zeros(self.number)
        for i in range(self.number):
            if self.campacity[i] - self._servering[i] > 0:
                tempEnv[i] = 0
            else:
                if self._servering[i] > 3*self.campacity[i]:
                    tempEnv[i] = 3
                elif self._servering[i] >2*self.campacity[i]:
                    tempEnv[i] = 2
                else:
                    tempEnv[i] = 1
        myString = str(int(data[3]))
        for i in tempEnv:
            myString+=str(int(i))
        self._servering[selected_server] += 1
        self.resouce_queue[selected_server].append(data)
        return myString, value #返回在当前环境s下采取a行动后的环境状态以及奖励值（环境自动更新）
    def reset(self):
        self._servering = np.zeros(self.number)
        self.resouce_queue = {}
        for i in range(self.number):
            self.resouce_queue[i] = []
        # try:
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass
    def getInfo(self):
        return self._servering
def main():
    #capacity = [M for i in range(N)]
    capacity = np.load('m.npy')
    print(capacity)
    environment = Env(init_server_number=N,capacity=capacity)
    Brain = SarsaLambdaTable(serverNum=N)
    MaxEpisode = 5
    res = np.array([])
    for episode in range(MaxEpisode):
        if episode == MaxEpisode -1:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            plt.ion()
            plt.show()
            pass
        environment.reset()
        current_time = 0
        current_server=0
        index = 0
        # initial observation
        observation = ""
        for i in range(N):
            observation += '0'
        action = Brain.choose_action(observation)
        print('episode:%s .....................'%episode)
        while index < sample_num:
            while index < sample_num and  data[index][1] <= current_time:
                # RL choose action based on observation
                observation_, reward = environment.reward(data[index],action,current_time)
                action_ = Brain.choose_action(observation)
                # RL take action and get next observation and reward
                if index!=0 and index%500==0:
                    print("current reward:%s \n"%reward)
                if episode == MaxEpisode - 1:
                    current_server = environment.getInfo()
                # RL learn from this transition
                Brain.learn(index == sample_num-1,observation, action, reward, observation_,action_)
                # swap observation
                observation = observation_
                action = action_
                # break while loop when end of this episode
                index += 1
            current_time += 1
            if episode == MaxEpisode - 1:
                res = np.append(res,environment.getInfo())
                # try:
                #     ax.lines.remove(lines[0])
                # except Exception:
                #     pass
                # server_name = [i for i in range(N)]
                # lines = ax.plot(server_name, current_server, color='red')
                # res.append(lines)
                # plt.pause(0.1)
    # ani = animation.ArtistAnimation(fig,res,interval=200,repeat=1000)
    # ani.save('test.gif',writer='pillow')
    #Brain.q_table.to_csv('./table.csv',sep=',',header=True)
    res = res.reshape(-1,N)
    np.save('saraslambda.npy',res)
if __name__=='__main__':
    main()