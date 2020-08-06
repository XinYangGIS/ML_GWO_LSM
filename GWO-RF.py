# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# from sklearn import cross_validation
from sklearn import model_selection
import numpy.random as rd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

## 1.加载数据
def load_data(data_file):
    data = []
    label = []
    f = open(data_file)
    for line in f.readlines():
        lines = line.strip().split(' ')
        # 提取得出label
        label.append(float(lines[0]))
        # 提取出特征，并将其放入到矩阵中
        index = 0
        tmp = []
        for i in range(1, len(lines)):
            li = lines[i].strip().split(":")

            if int(li[0]) - 1 == index:
                tmp.append(float(li[1]))
            else:
                while(int(li[0]) - 1 > index):
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp) < 13:
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.array(data), np.array(label).T

def quick_load_data(datafile):
    # data = pd.read_csv(datafile, names=None)
    data = pd.read_csv(datafile)
    data = data.fillna(0) #use 0 to fill 
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    a = X.shape
    return X, y

## 2. GWO优化算法
def gwo(train_feature,test_feature,train_label,test_label,SearchAgents_no,Max_iteration,dim,lb,ub):
    Alpha_pos=[0,0] # 初始化Alpha狼的位置
    Alpha_score = float("inf") # 初始化Alpha狼的目标函数值 (正无穷
    Beta_pos=[0,0]
    Delta_pos=[0,0]  
    Beta_score = float("inf")
    Delta_score = float("inf")
 
    # Positions = np.dot(rd.rand(SearchAgents_no,dim),(ub-lb))+lb #初始化首次搜索位置
    Positions = rd.randint(lb,ub, (SearchAgents_no, dim)) #初始化首次搜索位置
    
    # Convergence_curve = np.zeros((1,Max_iteration))#初始化融合曲线

    iterations = []
    accuracy = []
    position_list = []
    Best_alpha_x = []
    Best_alpha_y = []

    #主循环
    index_iteration = 0 
    while index_iteration < Max_iteration:
        # 遍历每个狼
        for i in range(0,(Positions.shape[0])):
            #若搜索位置超过了搜索空间，需要重新回到搜索空间 
            for j in range(0,(Positions.shape[1])): 
                Flag4ub=Positions[i,j]>ub
                Flag4lb=Positions[i,j]<lb
                #若狼的位置在最大值和最小值之间，则位置不需要调整，若超出最大值，最回到最大值边界
                if Flag4ub:                   
                    Positions[i,j] = ub
                if Flag4lb:                   
                    Positions[i,j] = lb
            #RF模型训练
            # rbf_svm = svm.SVC(kernel = 'rbf', C = Positions[i][0], gamma = Positions[i][1]).fit(train_feature, train_label)  #svm的使用函数
            clf2 = RandomForestClassifier(n_estimators=Positions[i][0], min_samples_leaf=Positions[i][1],random_state=None).fit(train_feature,train_label)
            #RF模型预测及其精度
            cv_scores = model_selection.cross_val_score(clf2,test_feature,test_label,cv =3,scoring = 'accuracy')
            #以错误率最小化为目标test_one = test_one
            # from sklearn.metrics import mean_squared_error
            # Vlid_y_pred = clf2.predict(test_feature)
            # mse_valid = np.sqrt(mean_squared_error(test_label, Vlid_y_pred))

            scores = cv_scores.mean()            
            fitness = (1 - scores)*100
            # fitness =  mse_valid *100
            if fitness<Alpha_score: #如果目标函数值小于Alpha狼的目标函数值
                Alpha_score=fitness # 则将Alpha狼的目标函数值更新为最优目标函数值
                Alpha_pos=Positions[i] #同时将Alpha狼的位置更新为最优位置
            if fitness>Alpha_score and fitness<Beta_score: #如果目标函数值介于于Alpha狼和Beta狼的目标函数值之间
                Beta_score=fitness # 则将Beta狼的目标函数值更新为最优目标函数值
                Beta_pos=Positions[i]
            if fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score: #如果目标函数值介于于Beta狼和Delta狼的目标函数值之间
                Delta_score=fitness # 则将Delta狼的目标函数值更新为最优目标函数值
                Delta_pos=Positions[i]


        a=2-index_iteration*(2/Max_iteration)
        
        # 遍历每个狼
        for i in range(0,(Positions.shape[0])):
            #遍历每个维度
            for j in range(0,(Positions.shape[1])): 
                #包围猎物，位置更新                
                r1=rd.random(1)#生成0~1之间的随机数
                r2=rd.random(1)               
                A1=2*a*r1-a # 计算系数A
                C1=2*r2 # 计算系数C

                #Alpha狼位置更新
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j])
                X1=Alpha_pos[j]-A1*D_alpha
                       
                r1=rd.random(1)
                r2=rd.random(1)

                A2=2*a*r1-a
                C2=2*r2

                # Beta狼位置更新
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j])
                X2=Beta_pos[j]-A2*D_beta

                r1=rd.random(1)
                r2=rd.random(1)

                A3=2*a*r1-a
                C3=2*r2

                # Delta狼位置更新
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j])
                X3=Delta_pos[j]-A3*D_delta

                # 位置更新
                Positions[i,j]=(X1+X2+X3)/3
                
                
                position_list.append(Positions[i,j])
        index_iteration = index_iteration + 1
        iterations.append(index_iteration)
        accuracy.append((100-Alpha_score)/100)
        # accuracy.append((Alpha_score)/100)
        Best_alpha_x.append(Alpha_pos[0])
        Best_alpha_y.append(Alpha_pos[1])
        step = 2
        All_position = [position_list[i:i+step] for i in range(0, len(position_list), step)]
        print('----------------迭代次数--------------------' + str(index_iteration))
        print(Positions)
        print('n_trees and min_samples_leaf:' + str(Alpha_pos))
        print('accuracy:' + str((100-Alpha_score)/100))
    
    best_n_trees=Alpha_pos[0]
    best_min_samples_leaf=Alpha_pos[1]
    # print(position_list)
    # recording the searching space 
    Bx = pd.DataFrame(Best_alpha_x)
    By = pd.DataFrame(Best_alpha_y)
    Bxy = pd.concat([Bx, By], axis=1)
    Bxy.to_csv('Best_Alpha_RF.csv')
    searching_pos = pd.DataFrame(All_position)
    searching_pos.to_csv('All_Position_RF.csv')
    print(Bxy)
    print(All_position)
    return best_n_trees,best_min_samples_leaf,iterations,accuracy

def plot(iterations,accuracy):
    fig = plt.figure(figsize=(6,3))
    plt.plot(iterations,accuracy)
    a = pd.DataFrame(iterations)
    b = pd.DataFrame(accuracy)
    c = pd.concat([a, b],axis=1)
    c.to_csv("GWO-RF-Scurve.csv")
    plt.xlabel('Number of iteration',size = 20)
    plt.ylabel('accuracy',size = 20)
    plt.title('GWO_RB parameter optimization')
    # plt.show()
    plt.savefig('GWO-RF600dpi.png', dpi=600)

if __name__ == '__main__':
    print('----------------1.加载数据-------------------')
    feature,label = quick_load_data('TrainDataset_AfterMulti.csv')
    #前200个为训练集，后70个为测试集
    train_feature, test_feature, train_label, test_label = model_selection.train_test_split(feature, label, test_size=0.3, random_state=None)
    # train_feature = train_feature.values.tolist()
    # test_feature = test_feature.values.tolist()
    # train_label = train_label.values.tolist()
    # test_label = test_label.values.tolist()
    print('----------------2.参数设置------------')
    SearchAgents_no=20 #狼群数量
    Max_iteration=300 #最大迭代次数
    dim=2 #需要优化的参数数量
    lb=15 #参数取值下界
    ub=32 #参数取值上界

    print('----------------3.GWO-----------------')
    best_n_trees,best_min_samples_leaf,iterations,accuracy = gwo(train_feature,test_feature,train_label,test_label,SearchAgents_no,Max_iteration,dim,lb,ub)

    print('----------------4.结果显示-----------------')
    print("The best n_trees is " + str(best_n_trees))
    print("The best min_samples_leaf is " + str(best_min_samples_leaf))
    plot(iterations,accuracy)