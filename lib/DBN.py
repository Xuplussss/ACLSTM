import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# from RBM import RBM
import time, math


BATCH_SIZE = 20

class RBM(nn.Module):
    '''
    这个类定义了RBM需要的所有函数
    激活函数 : sigmoid
    '''

    def __init__(self,visible_units=26,
                hidden_units = 9,
                k=2,
                learning_rate=1e-3,
                momentum_coefficient=0.5,
                weight_decay = 1e-4,
                use_gpu = False,
                _activation='sigmoid'):
        '''
        定义模型
        W:权重矩阵 shape=(可视节点数,隐藏节点数)
        c:隐藏层偏置 shape=(隐藏节点数 , )
        b : 可视层偏置 shape=(可视节点数 ,)
        '''
        super(RBM,self).__init__()

        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_gpu = use_gpu
        self._activation = _activation

        self.weight = torch.randn(self.visible_units,self.hidden_units) / math.sqrt(self.visible_units)  #初始化
        self.c = torch.randn(self.hidden_units) / math.sqrt(self.hidden_units)
        self.b = torch.randn(self.visible_units) / math.sqrt(self.visible_units)

        self.W_momentum = torch.zeros(self.visible_units,self.hidden_units)
        self.b_momentum = torch.zeros(self.visible_units)
        self.c_momentum = torch.zeros(self.hidden_units)


    def activation(self,X):
        if self._activation=='sigmoid':
            return nn.functional.sigmoid(X)
        elif self._activation=='tanh':
            return nn.functional.tanh(X)
        elif self._activation=='relu':
            return nn.functional.relu(X)
        else:
            raise ValueError("Invalid Activation Function")


    def to_hidden(self ,X):
        '''
        根据可视层生成隐藏层
        通过采样进行
        X 为可视层概率分布
        :param X: torch tensor shape = (n_samples , n_features)
        :return -  hidden - 新的隐藏层 (概率)
                    sample_h - 吉布斯样本 (1 or 0)
        '''
        # print('hinput:',X)
        hidden = torch.matmul(X,self.weight)
        hidden = torch.add(hidden, self.c)  #W.x + c
        # print('mm:',hidden)
        hidden  = self.activation(hidden)

        sample_h = self.sampling(hidden)
        # print('h:',hidden,'sam_h:',sample_h)

        return hidden,sample_h

    def to_visible(self,X):
        '''
        根据隐藏层重构可视层
        也通过采样进行
        X 为隐藏层概率分布
        :returns - X_dash - 新的重构层(概率)
                    sample_X_dash - 新的样本(吉布斯采样)
        '''
        # 计算隐含层激活，然后转换为概率
        # print('vinput:',X)
        X_dash = torch.matmul(X ,self.weight.transpose( 0 , 1) )
        X_dash = torch.add(X_dash , self.b)     #W.T*x+b
        # print('mm:',X_dash)
        X_dash = self.activation(X_dash)

        sample_X_dash = self.sampling(X_dash)
        # print('v:',X_dash, 'sam_v:', sample_X_dash)

        return X_dash,sample_X_dash

    def sampling(self,s):
        '''
        通过Bernoulli函数进行吉布斯采样
        '''
        s = torch.distributions.Bernoulli(s)
        return s.sample()

    def reconstruction_error(self , data):
        '''
        通过损失函数计算重构误差
        '''
        return self.contrastive_divergence(data, False)


    def contrastive_divergence(self, input_data ,training = True):
        '''
        对比散列算法
        '''
        # positive phase
        positive_hidden_probabilities,positive_hidden_act  = self.to_hidden(input_data)

        # 计算 W
        positive_associations = torch.matmul(input_data.t() , positive_hidden_act)



        # negetive phase
        hidden_activations = positive_hidden_act
        for i in range(self.k):     #采样步数
            visible_p , _ = self.to_visible(hidden_activations)
            hidden_probabilities,hidden_activations = self.to_hidden(visible_p)

        negative_visible_probabilities = visible_p
        negative_hidden_probabilities = hidden_probabilities

        # 计算 W
        negative_associations = torch.matmul(negative_visible_probabilities.t() , negative_hidden_probabilities)


        # 更新参数
        if(training):
            self.W_momentum *= self.momentum_coefficient
            self.W_momentum += (positive_associations - negative_associations)

            self.b_momentum *= self.momentum_coefficient
            self.b_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

            self.c_momentum *= self.momentum_coefficient
            self.c_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

            batch_size = input_data.size(0)

            self.weight += self.W_momentum * self.learning_rate / BATCH_SIZE
            self.b += self.b_momentum * self.learning_rate / BATCH_SIZE
            self.c += self.c_momentum * self.learning_rate / BATCH_SIZE

            self.weight -= self.weight * self.weight_decay  # L2 weight decay

        # 计算重构误差
        error = torch.mean(torch.sum((input_data - negative_visible_probabilities)**2 , 1))
        # print('i:',input_data,'o:',negative_hidden_probabilities)

        return error


    def forward(self,input_data):
        return  self.to_hidden(input_data)

    def step(self,input_data):
        '''
            包括前馈和梯度下降，用作训练
        '''
        # print('w:',self.weight);print('b:',self.b);print('c:',self.c)
        return self.contrastive_divergence(input_data , True)


    def trains(self,train_data,num_epochs = 50,batch_size= 20):

        BATCH_SIZE = batch_size

        if(isinstance(train_data ,torch.utils.data.DataLoader)):
            train_loader = train_data
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)


        for epochs in range(num_epochs):
            epoch_err = 0.0

            for batch,_ in train_loader:
            #     batch = batch.view(len(batch) , self.visible_units)

                if(self.use_gpu):
                    batch = batch.cuda()
                batch_err = self.step(batch)

                epoch_err += batch_err


            print("Epoch Error(epoch:%d) : %.4f" % (epochs , epoch_err))
        return


    def extract_features(self,test_dataset):
        if(isinstance(test_dataset ,torch.utils.data.DataLoader)):
            test_loader = test_dataset
        else:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # print(len(test_dataset));print(self.hidden_units)
        test_features = []
        test_labels = []

        for i, (batch, labels) in enumerate(test_loader):
            batch = batch.view(len(batch), self.visible_units)

            if self.use_gpu:
                batch = batch.cuda()

            print(self.to_hidden(batch));print(type(self.to_hidden(batch)));print(type(self.to_hidden(batch)[0]))
            print(labels);print(type(labels))
            # print(BATCH_SIZE,len(batch))
            test_labels.append(labels.numpy())
            test_features.append((self.to_hidden(batch)[0].numpy(),self.to_hidden(batch)[1].numpy()))
            # test_features.append([(m,n) for m,n in zip([x[i] for x in self.to_hidden(batch)[0].numpy()],[x[i] for x in self.to_hidden(batch)[1].numpy()])])
        # print(test_labels[0])
        # print(test_features[0])

        return np.array(test_features),np.array(test_labels)

class DBN(nn.Module):
    def __init__(self,
                visible_units = 4,             
                hidden_units = [8192,4096,2048],            
                k = 2,                           
                learning_rate = 1e-3,            
                momentum_coefficient = 0.9,      
                weight_decay = 1e-4,             
                use_gpu = False,
                _activation = 'sigmoid',):        
        super(DBN,self).__init__()

        self.n_layers = len(hidden_units)        
        self.rbm_layers =[]                      
        self.rbm_nodes = []

        # 构建不同的RBM层
        for i in range(self.n_layers ):

            if i==0:
                input_size = visible_units
            else:
                input_size = hidden_units[i-1]
            rbm = RBM(visible_units = input_size,
                    hidden_units = hidden_units[i],
                    k= k,
                    learning_rate = learning_rate,
                    momentum_coefficient = momentum_coefficient,
                    weight_decay = weight_decay,
                    use_gpu=use_gpu,
                    _activation = _activation)

            self.rbm_layers.append(rbm)

        # rbm_layers = [RBM(rbn_nodes[i-1] , rbm_nodes[i],use_gpu=use_cuda) for i in range(1,len(rbm_nodes))]
        self.W_rec = [nn.Parameter(self.rbm_layers[i].weight.data.clone()) for i in range(self.n_layers-1)]
        self.W_gen = [nn.Parameter(self.rbm_layers[i].weight.data) for i in range(self.n_layers-1)]
        self.bias_rec = [nn.Parameter(self.rbm_layers[i].c.data.clone()) for i in range(self.n_layers-1)]
        self.bias_gen = [nn.Parameter(self.rbm_layers[i].b.data) for i in range(self.n_layers-1)]
        self.W_mem = nn.Parameter(self.rbm_layers[-1].weight.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].b.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].c.data)

        for i in range(self.n_layers-1):
            self.register_parameter('W_rec%i'%i, self.W_rec[i])
            self.register_parameter('W_gen%i'%i, self.W_gen[i])
            self.register_parameter('bias_rec%i'%i, self.bias_rec[i])
            self.register_parameter('bias_gen%i'%i, self.bias_gen[i])

        self.BPNN=nn.Sequential(            #用作分类和反向微调参数
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048,7),
        )
        self.softmax = nn.Softmax()

    def forward(self , input_data):
        '''
            前馈
        '''
        v = input_data
        for i in range(len(self.rbm_layers)):
            v = v.view((v.shape[0] , -1)).type(torch.FloatTensor)#flatten
            p_v,v = self.rbm_layers[i].forward(v)
        # print('p_v:', p_v.shape,p_v)
        # print('v:',v.shape,v)
        out=self.BPNN(p_v)
        # print('out',out.shape,out)
        # print(self.BPNN(p_v))
        probs = self.softmax(out)
        return probs

    def train_static(self, train_data,train_labels,num_epochs,batch_size):
        '''
        逐层贪婪训练RBM,固定上一层
        '''

        tmp = train_data

        for i in range(len(self.rbm_layers)):
            print("-"*20)
            print("Training the {} st rbm layer".format(i+1))

            tensor_x = tmp.type(torch.FloatTensor)
            tensor_y = train_labels.type(torch.FloatTensor)
            _dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
            _dataloader = torch.utils.data.DataLoader(_dataset)

            self.rbm_layers[i].trains(_dataloader,num_epochs,batch_size)
            print(type(_dataloader))
            # print(train_data.shape)
            v = tmp.view((tmp.shape[0] , -1)).type(torch.FloatTensor)
            v,_ = self.rbm_layers[i].forward(v)
            tmp = v
            # print(v.shape)
        return

    def train_ith(self, train_data,num_epochs,batch_size,ith_layer,rbm_layers):
        '''
        只训练某一层，可用作调优
        '''
        if(ith_layer>len(rbm_layers)):
            return

        v = train_data
        for ith in range(ith_layer):
            v,out_ = self.rbm_layers[ith].forward(v)


        self.rbm_layers[ith_layer].trains(v, num_epochs,batch_size)
        return

    def trainBP(self,trainloader):
        optimizer = torch.optim.SGD(self.BPNN.parameters(), lr=0.005, momentum=0.7)
        loss_func = torch.nn.CrossEntropyLoss()
        for epoch in range(0):
            for step,(x,y) in enumerate(trainloader):
                bx = Variable(x)
                by = Variable(y)
                out=self.forward(bx)[1]
                # print(out)
                loss=loss_func(out,by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    print('Epoch: ', epoch, 'step:', step, '| train loss: %.4f' % loss.data.numpy())





def train_and_test(traind,trainl,testdat,testlabel,loader):
    print(type(traind),type(trainl),type(traind),type(trainl),type(loader))
    start_time = time.time()
    dbn=DBN()

    dbn.train()
    dbn.train_static(train_data=traind,train_labels=trainl,num_epochs=0,batch_size=20)

    optimizer = torch.optim.SGD(dbn.parameters(), lr=0.001, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    train_loader = loader
    dbn.trainBP(train_loader)

    for epoch in range(0):
        for step,(x,y) in enumerate(train_loader):
            # print(x.data.numpy(),y.data.numpy())

            b_x=Variable(x)
            b_y=Variable(y)

            output=dbn(b_x)
            # print(output)
            # print(prediction);print(output);print(b_y)

            loss=loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%10==0:
                print('Epoch: ', epoch, 'step:',step,'| train loss: %.4f' % loss.data.numpy())
    duration=time.time()-start_time

    dbn.eval()
    test_x = Variable(testdat);test_y = Variable(testlabel)
    test_out = dbn(test_x)
    # print(test_out)
    test_pred = torch.max(test_out, 1)[1]
    pre_val = test_pred.data.squeeze().numpy()
    y_val = test_y.data.squeeze().numpy()
    print('prediciton:',pre_val);print('true value:',y_val)
    accuracy = float((pre_val == y_val).astype(int).sum()) / float(test_y.size(0))
    print('test accuracy: %.2f' % accuracy,'duration:%.4f' % duration)
    return accuracy, duration