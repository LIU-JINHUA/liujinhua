#'''
#训练 DCGAN
#'''

#加上激活函数的非线性映射，多层全连接层理论上可以模拟任何非线性变换
import os
import glob #读取文件，返回所有匹配的文件路径列表，不包括子文件夹
import numpy as np
from scipy import misc #读取图片信息
import tensorflow as tf
from define import *
import matplotlib.pyplot as plt

def train():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略cpu和tensorflow对AVX等扩展的兼容警告。
    #获取训练数据
    data=[]
    for image in glob.glob("face2/*"):
        image_data = misc.imread(image)
        data.append(image_data)#append()方法是在列表中添加元素
                               #extend()方法是接受列表作为参数，并将参数的每个元素添加到原有列表。
    input_data=np.array(data)#转变为数组类型

    #将数据标准化成[-1,1]的范围中，这也是tanh激活函数的输出范围
    input_data=((input_data.astype(np.float32)-127.5)/127.5)

    #构造生成器和判别器
    g=generator_model()
    d=discriminator_model()

    #构建生成器和判别器组成的网络模型
    d_on_g=generator_containing_discriminator(g,d)

    #优化器Adam Optimizer
    g_optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE2,beta_1=BETA_1)
    d_optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE,beta_1=BETA_1)

    #配置生成器和判别器，Sequential.compile用来配置神经网络
    g.compile(loss="binary_crossentropy",optimizer=g_optimizer)
    d_on_g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d.trainable=True
    d.compile(loss="binary_crossentropy",optimizer=d_optimizer)



    #开始训练
    for epoch in range(EPOCHS):
        #每经过整数倍的BATCH_SIZE去训练input_data
        for index in range(int(input_data.shape[0]/BATCH_SIZE)):
            #构建一个在此batch下循环的input_data的集合
            input_batch=input_data[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            
            #构建连续型均匀分布的随机数据（噪声）,uniform:随机生成连续性[-1，1]之间的数据
            random_data=np.random.uniform(-1,1,size=(BATCH_SIZE,100))

            #生成器生成的图片数据
            generated_images=g.predict(random_data,verbose=0)

            input_batch=np.concatenate((input_batch,generated_images))

            output_batch=[1]*BATCH_SIZE+[0]*BATCH_SIZE


            #训练判别器，使其具备识别不合格生成图片的能力
            d_loss=d.train_on_batch(input_batch,output_batch)


            #训练生成器,固定判别器
            d.trainable=False
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

            #训练生成器，并通过不可被训练的判别器去判别
            g_loss=d_on_g.train_on_batch(random_data,[1]*BATCH_SIZE)

            #恢复判别器可被训练
            d.trainable=True

            #打印损失
            print("Epoch {}, 第 {} 步, 生成器的损失: {:.3f},判别器的损失: {:.3f}".format(epoch, index, g_loss,d_loss))

        #保存生成器和判别器的参数
        if epoch %10 == 9:
            g.save_weights("generator_weight",True)
            d.save_weights("discriminator_weight",True)


if __name__=="__main__":
    train()