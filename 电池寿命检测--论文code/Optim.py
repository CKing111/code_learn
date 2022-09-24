# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model import BiLSTM_Attention
import math
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

def HUATU(trace,result):
    plt.figure()
    plt.plot(trace)
    plt.title('fitness curve')
    plt.xlabel('iteration')
    plt.ylabel('fitness value')
    plt.savefig('figure/woa fitness curve.png')
    
    plt.figure()
    plt.plot(result[:,0])
    plt.xlabel('优化次数/代')
    plt.ylabel('学习率')
    plt.savefig('figure/学习率优化.png')
    
    plt.figure()
    plt.plot(result[:,1])
    plt.xlabel('优化次数/代')
    plt.ylabel('训练次数')
    plt.savefig('figure/训练次数优化.png')
    
    plt.figure()
    plt.plot(result[:,2])
    plt.xlabel('优化次数/代')
    plt.ylabel('Batchsize')
    plt.savefig('figure/Batchsize优化.png')
    
    plt.figure()
    plt.plot(result[:,3])
    plt.xlabel('优化次数/代')
    plt.ylabel('第一层BiLSTM节点数')
    plt.savefig('figure/第一层BiLSTM节点数优化.png')
    
    plt.figure()
    plt.plot(result[:,4])
    plt.xlabel('优化次数/代')
    plt.ylabel('第二层BiLSTM节点数')
    plt.savefig('figure/第二层BiLSTM节点数优化.png')
    
    plt.figure()
    plt.plot(result[:,5])
    plt.xlabel('优化次数/代')
    plt.ylabel('全连接层节点数')
    plt.savefig('figure/全连接层节点数优化.png')
    plt.show()

def fitness(pop,P,T,Pt,Tt):
    
    tf.random.set_seed(0)
    lr = pop[0]  # 学习率
    num_epochs = int(pop[1])  # 迭代次数
    batch_size = int(pop[2])  # batchsize
    hidden1 = int(pop[3])  # bilstm节点数1
    hidden2 = int(pop[4])  # bilstm节点数2
    fc = int(pop[5]) #全连接层节点数
    sequence, feature = P.shape[-2:]
    output_node = T.shape[1]
    
    model = BiLSTM_Attention(sequence, feature,hidden1,hidden2, fc, output_node).build_model()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    
    model.fit(P, T, 
              epochs=num_epochs,batch_size=batch_size, verbose=0)
    
    Tp = model.predict(Pt)
        
    F2=np.mean(np.square((Tp-Tt)))
    return F2



def WOA(p_train,t_trian,p_test,t_test):

    max_iterations=10
    noposs=5
    '''
    学习率，训练次数，batcisize，
    两个bilstm隐含层的节点数、全连接层节点数
    的寻优上下界
    '''
    lb=[0.001,10, 16, 1 ,1 ,1]
    ub=[0.01, 100,128,20,20,100]#
    noclus=len(lb)#寻优维度

    poss_sols = np.zeros((noposs, noclus)) # whale positions
    gbest = np.zeros((noclus,)) # globally best whale postitions
    b = 2.0
    
   # 种群初始化，学习率为小数，其他的都是整数
    for i in range(noposs):
        for j in range(noclus):
            if j==0:
                poss_sols[i][j] =(ub[j]-lb[j])*np.random.rand()+lb[j]
            else:
                poss_sols[i][j] =  np.random.randint(lb[j],ub[j])

    global_fitness = np.inf
    for i in range(noposs):
        cur_par_fitness = fitness(poss_sols[i,:],p_train,t_trian,p_test,t_test) 
        if cur_par_fitness < global_fitness:
            global_fitness = cur_par_fitness
            gbest = poss_sols[i].copy()
    # 开始迭代
    trace,trace_pop=[],[]
    for it in range(max_iterations):
        for i in range(noposs):
            a = 2.0 - (2.0*it)/(1.0 * max_iterations)
            r = np.random.random_sample()
            A = 2.0*a*r - a
            C = 2.0*r
            l = 2.0 * np.random.random_sample() - 1.0
            p = np.random.random_sample()
            
            for j in range(noclus):
                x = poss_sols[i][j]
                if p < 0.5:
                    if abs(A) < 1:
                        _x = gbest[j].copy()
                    else :
                        rand = np.random.randint(noposs)
                        _x = poss_sols[rand][j]
                    D = abs(C*_x - x)
                    updatedx = _x - A*D
                else :
                    _x = gbest[j].copy()
                    D = abs(_x - x)
                    updatedx = D * math.exp(b*l) * math.cos(2.0* math.acos(-1.0) * l) + _x
                #if updatedx < ground[0] or updatedx > ground[1]:
                #    updatedx = (ground[1]-ground[0])*np.random.rand()+ground[0]
                #   randomcount += 1
                
                poss_sols[i][j] = updatedx
            poss_sols[i,:]=boundary(poss_sols[i,:],lb,ub)#边界判断
            fitnessi = fitness(poss_sols[i],p_train,t_trian,p_test,t_test)
            if fitnessi < global_fitness :
                global_fitness = fitnessi
                gbest = poss_sols[i].copy()
        trace.append(global_fitness)
        print ("iteration",it+1,"=",global_fitness,[gbest[i] if i==0 else int(gbest[i]) for i in range(len(lb))])

        trace_pop.append(gbest)
    return gbest, trace,trace_pop

def boundary(pop,lb,ub):
    
    # 防止粒子跳出范围,除学习率之外 都应该是整数
    pop=[pop[i] if i==0 else int(pop[i]) for i in range(len(lb))]
    for j in range(len(lb)):
        if pop[j]>ub[j] or pop[j]<lb[j]:
            if j==0:
                pop[j] =(ub[j]-lb[j])*np.random.rand()+lb[j]
            else:
                pop[j] =  np.random.randint(lb[j],ub[j])
    
    
    return pop

    
    