#使用LSTM网络看是否可以成功分类sin函数和tan函数
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import random
import math
 
num_steps = 100
batch_size = 2
state_size = 10
num_features = 1
num_classes = 2
learning_rate = 1e-4
# 生成数据的函数
def get_data_set(data_num):
    label = np.array([random.randint(0,1) for _ in range(data_num)])
    data = np.zeros((data_num,num_steps),dtype=np.float32)
    # 数据随机产生并填充，原来是再末尾填充0，直接生成数据的话我就随机填充，更符合实际情况
    for data_index in range(data_num):
        if label[data_index]==1:
            data_list=[i for i in map(math.tan,np.linspace(random.random()*math.pi,random.randint(15,100),random.randint(50,100)))]
        else:
            data_list=[i for i in map(math.sin,np.linspace(random.random()*math.pi,random.randint(15,100),random.randint(50,100)))]
        for i in range(num_steps-len(data_list)):
            data_list.insert(random.randint(0,len(data_list)),0)
        data[data_index] = np.array(data_list)
    return data,label
#数据在读取时已经做好了填充，且已是打乱的数据
def getDataAndLabel(raw_data,raw_label,num_steps,batch_size):
    data = np.zeros([batch_size,num_steps,num_features],np.float32)
    labels = np.zeros([batch_size,1],np.int32)
    # 根据batch_size确定单epoch训练步数
    for step_index in range(len(raw_data)//batch_size):
        data = raw_data[step_index*batch_size:(step_index+1)*batch_size].reshape(batch_size,-1,num_features)
        labels = raw_label[step_index*batch_size:(step_index+1)*batch_size].reshape(batch_size,1)
        yield data,labels
 
def genEpochs(num_epochs,raw_data,raw_label,num_steps,batch_size):
    for i in range(num_epochs):
        yield getDataAndLabel(raw_data,raw_label,num_steps,batch_size)
 
def buildNewLSTMGraph(
    num_steps = num_steps,
    batch_size = batch_size,
    state_size = state_size,
    learning_rate = learning_rate):
    #初始化视图
    tf.reset_default_graph()
    #给定输入和标签管道
    x = tf.placeholder(shape=[batch_size,num_steps,num_features],dtype=tf.float32)
    y = tf.placeholder(shape=[batch_size,1],dtype=tf.int32)
    #使用静态list作为输入和标签,squeeze返回的是一个张量
    x_temp = tf.squeeze(tf.split(value=x,num_or_size_splits=num_steps,axis=1),axis=2)
    rnn_inputs = list()
    rnn_labels = list()
    for i in range(x_temp.shape[0]):
        rnn_inputs.append(x_temp[i])
    for i in range(y.shape[0]):
        rnn_labels.append(tf.one_hot(y[i],depth=num_classes))
    #定义网络节点
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=state_size,state_is_tuple=True)
    init_state = cell.zero_state(batch_size,tf.float32)
    rnn_outpus, final_state = tf.contrib.rnn.static_rnn(cell=cell,initial_state=init_state,inputs=rnn_inputs)
    #使用LSTM的最后状态对函数进行预测
    with tf.variable_scope('fullyConnLayer'):
        #第一层全连接层,输出状态为64种
        W1 = tf.get_variable('W1',shape=[state_size,64])
        b1 = tf.get_variable('b1',shape=[64],initializer=tf.constant_initializer(dtype=tf.float32,value=0))
        tempState1 = [tf.matmul(final_state[1],W1) + b1]
        tempState1 = tf.nn.relu(tempState1)
        #第二层全连接层，输出状态为32种
        W2 = tf.get_variable('W2',shape=[64,32])
        b2 = tf.get_variable('b2',shape=[32],initializer=tf.constant_initializer(dtype=tf.float32,value=0))
        tempState2 = [tf.matmul(tempState1[0],W2) + b2]
        tempState2 = tf.nn.relu(tempState2)
        #第三层全连接层，进行分类
        W3 = tf.get_variable('W3',shape=[32,num_classes])
        b3 = tf.get_variable('b3',shape=[num_classes],initializer=tf.constant_initializer(dtype=tf.float32,value=0))
    logits = [tf.matmul(tempState2[0],W3) + b3]
    #softmax的话则计算交叉熵
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=rnn_labels,logits=logits)
    total_loss = tf.reduce_mean(losses)
    #优化器的选择
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(losses)
    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        pred = tf.nn.softmax(logits),
        labels = rnn_labels,
    )
#这一部分是用来做最初的训练，主要是将模型数据保存下来以便后面继续训练
#我比较懒，没有保存交叉熵最小的几个模型，大家如有需要自己加个判断即可
def train_rnn(g,num_epochs,batch_size=batch_size,num_steps=num_steps):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_losses = []
        for idx,epoch in enumerate(genEpochs(batch_size=batch_size,num_epochs=num_epochs,num_steps=num_steps,
                                            raw_data=dataArray,raw_label=labelArray)):
            train_loss = 0
            step = 0
            training_state = None
            for X,Y in epoch:
                step +=1
                feed_dict = {g['x']:X,g['y']:Y}
                if(training_state != None):
                    feed_dict[g['init_state']] = training_state
                train_loss_, training_state,_ = sess.run([g['total_loss'],
                                                              g['final_state'],
                                                              g['train_step']],
                                                             feed_dict = feed_dict)
            train_loss += train_loss_
            print('epoch: {0}的平均损失值：{1}'.format(idx, train_loss/step))
            train_losses.append(train_loss/step)
        #保存模型
        saver=tf.train.Saver(max_to_keep=1)
        path = saver.save(sess,'Saved_model/funcClassify')
        print("保存至",path)
    return train_losses
# 生成训练数据
dataArray,labelArray=get_data_set(200)
g = buildNewLSTMGraph()
train_rnn(g,3)
#执行下面代码可以对保存的神经网络进行再次训练
g = buildNewLSTMGraph()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
num_epochs = 500
with sess:
    saver = tf.train.Saver()
    saver.restore(sess,'Saved_model/funcClassify')
    train_losses = []
    for idx,epoch in enumerate(genEpochs(batch_size=batch_size,num_epochs=num_epochs,
                                         num_steps=num_steps,raw_data=dataArray,raw_label=labelArray)):
        train_loss = 0
        step = 0
        prediction = list()
        labels = list()
        training_state = None
        for X,Y in epoch:
            step +=1
            feed_dict = {g['x']:X,g['y']:Y}
            if(training_state != None):
                feed_dict[g['init_state']] = training_state
            train_loss_, training_state,_ = sess.run([g['total_loss'],
                                                          g['final_state'],
                                                          g['train_step']],
                                                         feed_dict = feed_dict)
        train_loss += train_loss_
        print('epoch: {0}的平均损失值：{1}'.format(idx, train_loss/step))
    #保存模型    
    saver=tf.train.Saver(max_to_keep=1)
    path = saver.save(sess,'Saved_model/funcClassify')
    print('模型已保存到',path)