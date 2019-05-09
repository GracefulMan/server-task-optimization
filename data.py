import numpy as np
sample_num = 10000
feature_num = 4
def data(name):
    data = np.ones([sample_num, feature_num])
    data[:,0] = np.arange(sample_num).reshape(sample_num, )
    data[:,1] = np.sort(np.random.randint(0,int(sample_num/10),sample_num)).reshape(sample_num, )
    data[:,2] = np.random.randint(1, 100, sample_num).reshape(sample_num, )
    data[:,3] = np.random.randint(0, 4, sample_num).reshape(sample_num, )
    np.save('%s.npy'%name,data)
    print('ok')


M = np.random.randint(10,100,30)
np.save('m.npy',M)