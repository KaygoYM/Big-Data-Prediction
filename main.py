# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:40:32 2018

@author: Kaygo
"""
'''
因涉及项目机密，本程序不会使用原数据，而用相似的boston数据集
区别在于原数据集数据量少而特征多，因而要做特征工程，而boston数据集数据量多而特征少
因而旨在介绍一般搭建模型框架的思路
本程序只是一个demo版本，用于学习使用
'''
from sklearn.preprocessing import *
from sklearn.cross_validation import * 
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import xgboost as xgb #XGB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%导入数据
from sklearn.datasets import load_boston
boston=load_boston()
y=np.array(boston['target'])#预测值target y
feature_names=list(boston['feature_names'])#特征
X=np.array(boston['data'])#X
#%%数据预处理
data_X=pd.DataFrame(columns=feature_names,data=X)#转成df形式，个人比较习惯
#有缺失值则补0
data_X=data_X.fillna(0)
#或者有缺失值的扔掉也可以
#data_X=data_X.dropna()       

#数据预处理之后，在做数据变换之前，做初步的相关性分析，为特征工程做准备
corr=data_X.corr()
print(corr)
#corr.to_excel('corr_analyze.xlsx')
#%%数据变换
#X标准化
X_std=StandardScaler().fit_transform(X)#仅限连续变量
#如果train和test分开则test集要和train集做同尺度的标准化
#std=StandardScaler().fit(X_train)
#X_std_train=std.transform(X_train)
#X_std_test=std.transform(X_test)
#如果有类别变量（分类特征）注意要做编码（Onehot）
#Class_X=OneHotEncoder().fit_transform(Class_X).toarray()

#%%降维-特征工程（类别变量最好不参与，参与可能会比较麻烦）
#这里我们尝试过很多方式，如什么都不做、PCA、前向选择（逐步回归）、随机森林法等等
#你可以使用每个特征选择方法都试一下看看效果
#前向选择法见子函数
#为了方便，我们只用PCA做一个demo
fe_name='PCA'
X_std_pca=PCA(n_components=0.95).fit_transform(X_std)#保留95%信息
#如果train和test分开则test集和train集也要做同尺度的特征选择,方法同上
#如果有分类变量合并过来
#X_std_pca=np.hstack((Class_X,X_std_pca))

#%%训练train、验证valid、测试test
#实际情况一般都会给出训练集（知道y值）、和测试集（不知道y值）（比如实际项目就有）
#然后从训练集中再选出一部分作为验证集，剩下的作为训练集
#这里因为没有测试集，我们只用train_test_split随机分为训练集和验证集
X_train,X_valid, y_train, y_valid = train_test_split(X_std_pca, y,test_size=0.2,random_state=1)
valid_index=list(list(y).index(i) for i in y_valid)#验证集的索引

#%%多元回归建模
#一般预测都是多元回归问题
#你可以使用多种模型，看看各【模型】+【特征工程】的验证效果，选择一个效果好的去测试集上预测
#常用模型有Linear_regression/SVR/random_forest/GBDT/XGB以及数据挖掘大杀器stacking等
#stacking见子函数
#为了方便，我们只用xgb做一个demo
model_name='xgb'
model_xgb=xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=255, silent=True, objective='reg:gamma')
#模型训练和评估
model_xgb.fit(X_train,y_train)
print(fe_name+' '+model_name,model_xgb.feature_importances_)#这里是PCA的各特征重要性，PCA的特征无实际意义
y_pred=model_xgb.predict(X_valid)#验证值
#根据业务需要，选择不同的评价指标
error=abs(y_pred-y_valid)#误差
mse=mean_squared_error(y_valid,y_pred)
mae=mean_absolute_error(y_valid,y_pred)#绝对误差
print("MSE: %.4f" % mse)
print("MAE: %.4f" % mae)


#%%可视化
#从验证集里抽10个误差百分比最小的可视化一下
show=error.argsort()[0:10]
t = np.arange(len(show))
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(t, y_valid[show], 'r*-', linewidth=2, label=u'True_price')#真实值/千万
ax.plot(t,y_pred[show],'bo-', linewidth=2, label=u'Pred_price')#预测值
ax.set_title(u'DEMO', fontsize='xx-large')
ax.set_xticks(t)
ax.set_xticklabels(show,rotation=30,fontsize='medium')
ax.legend(loc='best',fontsize='large')
ax.set_xlabel('Valid_index',fontsize='x-large')
ax.set_ylabel('Price',fontsize='xx-large')
plt.grid()
plt.show()

#%%预测
#y_test=model.fit(X_test)
