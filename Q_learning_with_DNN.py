"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
N = 30
M = 20
#generate data
sample_num = 5000
feature_num = 4
data = np.ones([sample_num, feature_num])
data[:,0] = np.arange(sample_num).reshape(sample_num, )
data[:,1] = np.sort(np.random.randint(0,int(sample_num/10),sample_num)).reshape(sample_num, )
data[:,2] = np.random.randint(1, 100, sample_num).reshape(sample_num, )
data[:,3] = np.random.randint(0, 4, sample_num).reshape(sample_num, )
class DeepQnetwork:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate = 0.01,
        reward_decay = 0.9,
        e_greedy = 0.9,
        replace_target_iter = 300,
        memory_size = 1000,
        batch_size = 32,
        e_greedy_increment = None,
        output_graph = False

    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0
        #initialize zero memory[s,a,r,s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))#need to be fixed
        # consist of [target_net, evaluate_net]
        self._build_net()


        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess = tf.Session()
        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        w_initializer, b_initializer = tf.random_normal_initializer(0, 0.3), tf.constant_initializer(0.1)
        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')


        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
    def store_transition(self,s, a, r, s_):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        #replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1


    def choose_action(self, observation):#传入servering数组
        #to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(
                self.q_eval,
                feed_dict={self.s:observation}
            )
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0,self.n_actions)
        return action

    def learn(self):
        #check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced:step_counter: %s\n'%self.learn_step_counter)
        #sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size = self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size = self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s : batch_memory[:, : self.n_features],
                self.a : batch_memory[:, self.n_features],
                self.r : batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:]
            })
        self.cost_his.append(cost)

        #increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig('1.png')

class Env:
    #状态：各个服务器的资源分配状态，保持各个服务器尽可能无排队现象以及无极度空闲的状态为目标
    def __init__(self,init_server_number,capacity):
        self.number = init_server_number
        self.campacity = capacity
        self._servering = np.zeros(init_server_number)
        self.resouce_queue = {}
        for i in range(init_server_number):
            self.resouce_queue[i] = []
    def reward(self,data,selected_server):#selected_server indicting the index of server
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
        self._servering[selected_server] += 1
        self.resouce_queue[selected_server].append(data)
        return self.campacity - self._servering, value #返回在当前环境s下采取a行动后的环境状态以及奖励值（环境自动更新）

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
    #训练集的feature：时间，优先级，运行时间长度
    capacity =np.array( np.full(N,M) )
    environment = Env(init_server_number=N,capacity=capacity)
    Brain = DeepQnetwork(
        n_actions = N,
        n_features = N,
        learning_rate = 0.01,
        reward_decay = 0.9,
        e_greedy = 0.9,
        replace_target_iter = 200,
        memory_size = 2000,
        output_graph=True
    )
    MaxEpisode = 20
    # res = []
    for episode in range(MaxEpisode):
        if episode == MaxEpisode -1:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            plt.ion()
            plt.show()
        environment.reset()
        current_time = 0
        index = 0
        # initial observation
        observation = capacity
        print(observation)
        print('episode:%s .....................'%episode)
        while index < sample_num:
            while index < sample_num and  data[index][1] <= current_time:
                # RL choose action based on observation
                action = Brain.choose_action(observation)
                # RL take action and get next observation and reward
                observation_, reward = environment.reward(data[index],np.round(action))
                Brain.store_transition(observation, action, reward, observation_)
                if index!=0 and index%500==0:
                    print("current reward:%s \n"%reward)
                # RL learn from this transition
                if index > 200 and index%50==0:
                    Brain.learn()
                # swap observation
                observation = observation_
                # break while loop when end of this episode
                index += 1
            current_time += 1
            if episode == MaxEpisode - 1 and current_time % 20 == 0:
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                server_name = [i for i in range(N)]
                lines = ax.plot(server_name, current_server, color='red')
                # res.append(lines)
                plt.pause(0.1)
    # ani = animation.ArtistAnimation(fig,res,interval=200,repeat=1000)
    # ani.save('test.gif',writer='pillow')
    Brain.plot_cost()
if __name__=='__main__':
    main()

