#!/usr/bin/env python
# coding: utf-8

# # Decesion Tree

# ## Summary

# ### 目的

# + 實作Decision Tree 進行二元分類預測
# + 用底層Mapreduce方式來和MLlib進行比對

# ### 資料

# + **StumbleUpon Evergreen Classification Challenge**
# + Dataset:https://www.kaggle.com/c/stumbleupon/data
# + Predict the pages is ephemeral or evergreen

# ## Set up

# ### colab-environment

# In[ ]:


get_ipython().system('apt-get install openjdk-8-jdk-headless -qq > /dev/null')
get_ipython().system('wget -q http://www-eu.apache.org/dist/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz')
get_ipython().system('tar xf spark-2.4.4-bin-hadoop2.7.tgz')
get_ipython().system('pip install -q findspark')

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.4-bin-hadoop2.7"

import findspark
findspark.init()


# In[9]:


from google.colab import drive
drive.mount('/content/drive')


# ### local-environment

# In[ ]:


import os
os.environ['JAVA_HOME'] = 'C:\Program Files\Java\jdk1.8.0_201'


# ### import and set sc

# In[ ]:


import numpy as np # for preprocess
import math 

import pyspark
from pyspark import SparkConf, SparkContext


# In[4]:


conf = SparkConf().set('spark.driver.host','127.0.0.1').setMaster("local").setAppName("DececisionTree").set("spark.default.parallelism", 4)
sc = SparkContext(conf=conf)
sc


# In[ ]:


# Parameter
category_Numbers = 14 # 一共14個categories類別
spilt_rate = [9,1] # 用8：2的比例分割資料成 訓練/測試 資料集 
Min_leaf_size = 250
N = 10


# ## Input

# In[16]:


Input = sc.textFile("./data/train.tsv")
Input.count()


# In[17]:


Input.first()


# ## Preprocess

# ### 資料清洗

# #### 清洗標題

# In[ ]:


title = Input.first()
Data = Input.filter(lambda x : x!= title)


# In[19]:


Data.first()


# #### 分割資料
# + 原始資料是以`\t`分割,並由`"`包覆

# In[20]:


lines = Data.map(lambda x : x.replace("\"","")).map(lambda x : x.split("\t"))
lines.first()[3:]


# ### 提取特徵

# #### 建立one-hot encode table

# In[ ]:


category_with_index = lines.map(lambda x: x[3]).distinct().zipWithIndex()


# In[22]:


category_Numbers_list = list(range(category_Numbers))
category_Numbers_array = np.array(category_Numbers_list).reshape(category_Numbers, -1)
category_Numbers_array


# In[23]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(category_Numbers_array)
encoder_table = enc.transform(category_Numbers_array).toarray()


# In[24]:


category_Map = category_with_index.map(lambda x : (x[0],encoder_table[x[1]])).collectAsMap()
category_Map


# In[ ]:


from pyspark.mllib.regression import LabeledPoint
def extract_features(row):
    category_features = category_Map[row[3]]
    number_features = row[4:-2]
    number_features = [0.0 if x=="?" else float(x) for x in number_features]
    
    features = np.concatenate((category_features,number_features))
    label = float(row[-1])
    
    return (label,features)


# In[26]:


labelRDD = lines.map(extract_features).map(lambda x: LabeledPoint(x[0],x[1]))
labelRDD.first()


# ### 切分資料

# In[27]:


(trainRDD,testRDD) = labelRDD.randomSplit(spilt_rate)
print("train: " + str(trainRDD.count()))
print("test:  " + str(testRDD.count()))


# ### 持久化

# In[28]:


trainRDD.persist()
testRDD.persist()


# ## Train Model By mllib

# + 用內建的model來做對比

# In[ ]:


from pyspark.mllib.tree import DecisionTree
model = DecisionTree.trainClassifier(
    data=trainRDD,numClasses=2,categoricalFeaturesInfo={},
    impurity="entropy", maxDepth=5, maxBins=5)


# In[30]:


right_count = wrong_count = 0
for test_data in testRDD.take(testRDD.count()):
    ans = test_data.label
    gus = model.predict(test_data.features)
    if ans==gus:
        right_count += 1
    else :
        wrong_count += 1
    print(str(right_count) + ":" + str(wrong_count), end = "\r")


# In[31]:


accuracy = right_count/(right_count+wrong_count)*100
print("正確率:%.1f" % accuracy)


# ## Train Model By MapReduce

# + category feature : `0~14` 0 or 1
# + numerical feature: `15~34` float

# ### 分裂節點的參數計算

# #### 計算entropy

# In[ ]:


# 計算 2-state system 的entropy
def entropy(state1,state2):
    if state1==0 or state2==0:
        return 0
    else:
        p1 = state1/(state1+state2)
        p2 = state2/(state1+state2)
#         p2 = 1-p1
        return -(p1*math.log2(p1)) -(p2*math.log2(p2))


# In[ ]:


def RDD_entropy(RDD,count="no_value"):
    count = RDD.count() if count=="no_value" else count
    state1 = RDD.filter(lambda x: x[0]).count()
    return entropy(state1,count-state1)


# In[36]:


entropy(1,100)


# #### 一個feature中尋找產生最大的information gain

# In[ ]:


def max_split_gain(RDD,sample_node = 0):
    # RDD (label,feature)
    split_points = RDD.values().distinct().collect()
    split_points.sort()
    R0_count = RDD.count()
    R0_entropy = RDD_entropy(RDD,R0_count)
    
    # sample data for fastser
    if sample_node<len(split_points) and sample_node>0:
        sample_rate = int(len(split_points)/sample_node)
        split_points = [split_points[i] for i in range(0,len(split_points),sample_rate)]
    
    # try every point in split_points
    # to get the max information gain
    gain_list = []
    for point in split_points:
        R1 = RDD.filter(lambda x : x[1]<point)
        R2 = RDD.filter(lambda x : x[1]>=point)
        R1_count = R1.count()
        R2_count = R0_count-R1_count
        
        gain = R0_entropy - (R1_count/R0_count)*RDD_entropy(R1,R1_count) - (R2_count/R0_count)*RDD_entropy(R2,R2_count)
        gain_list.append((gain,point))
    
    return max(gain_list) # (gain,split_point)
        


# #### 所有feature內找最大information gain

# In[ ]:


def max_feature_gain(RDD,sample_node=0):
    feature_types = len(RDD.first().features) # 35
    
    gain_list = []
    for feature_index in range(feature_types):
        RDD_one_feature = RDD.map(lambda x: (x.label,x.features[feature_index])) # (key,value)
        one_feature_max_gain = max_split_gain(RDD_one_feature,sample_node)
        print("Now in feature[%d],max gain is %.6f with split at %.3f " % (feature_index,one_feature_max_gain[0],one_feature_max_gain[1]),end="\r")
        gain_list.append((one_feature_max_gain,feature_index))
    
    max_gain = max(gain_list) # ((gain,split_point),feature_index)
    print("Best gain in feature[%d] with split at %.3f is : %.6f" % (max_gain[1],max_gain[0][1],max_gain[0][0]))
    return (max_gain[1],max_gain[0][1])


# In[39]:


example_RDD = trainRDD.randomSplit([0.05,0.95])[0] # 取1/10的資料來做示範
max_feature_gain(RDD=example_RDD,sample_node=2)


# ### 建樹

# In[ ]:


class node:
    def __init__(self,RDD):
        # value
        self.RDD = RDD
        self.count = self.RDD.count()
        self.level = 0
        self.feature = None
        self.split_point = None
        self.predict_value = None
        self.RDD.persist()
        # tree
        self.left = None
        self.right = None
  
    def setLeft(self, left):
        self.left = left
        self.left.level = self.level + 1
        
    def setRight(self, right):
        self.right = right
        self.right.level = self.level + 1
    
    def get_count(self):
        self.count = self.RDD.count()
        return self.count
    
    def get_predict(self):
        label_one_count = self.RDD.map(lambda x: x.label).filter(lambda x: x).count()
        self.predict_value = int(label_one_count/self.count *2  + 1e-9)
        return self.predict_value
    
    def get_split(self):
        (self.feature,self.split_point) = max_feature_gain(self.RDD,sample_node=N)
        print("split at " + str((self.feature,self.split_point)))
        
    def is_leaf(self):
        return self.count <= Min_leaf_size
        
    def build(self):
        if self.is_leaf():
            return
        
        self.get_split()
        (feature_index,split_point_value) = (self.feature,self.split_point)
        print(self.count)
        
        R1 = self.RDD.filter(lambda x : x.features[feature_index]<split_point_value)
        self.setLeft(node(R1))
        print("build left at %s" % str((feature_index,split_point_value)))
        self.left.build()

        R2 = self.RDD.filter(lambda x : x.features[feature_index]>=split_point_value)
        self.setRight(node(R2))
        print("build right at %s" % str((feature_index,split_point_value)))
        self.right.build()
    
    def level_order_print(self):
        
        if self.is_leaf():
            self.get_predict()
            print("\t"*self.level + str(self.predict_value))
            return
        else :
            print("\t"*self.level + str((self.feature,self.split_point)))
        
        print("\t"*self.level + "left")
        self.left.level_order_print()
        print("\t"*self.level + "right")
        self.right.level_order_print()
    
    def predict(self,features):
        if self.is_leaf():
            self.get_predict()
#             print(self.predict_value)
            return self.predict_value
        else:
            if features[self.feature] < self.split_point:
                return self.left.predict(features)
            else :
                return self.right.predict(features)


# In[ ]:


root = node(trainRDD)


# In[42]:


root.build()


# In[43]:


root.level_order_print()


# ## Test Model

# In[45]:


right_count = wrong_count = 0
for test_data in testRDD.take(testRDD.count()):
    ans = test_data.label
    gus = root.predict(test_data.features)
    if ans==gus:
        right_count += 1
    else :
        wrong_count += 1
    print(str(right_count) + ":" + str(wrong_count), end = "\r")
print(str(right_count) + ":" + str(wrong_count))


# In[46]:


accuracy = right_count/(right_count+wrong_count)*100
print("正確率:%.1f" % accuracy)


# ## 結論

# + 建立決策樹
#     + 用二元樹遞迴建立
#     + 選擇最大資訊增益之feature及split_point
#     + 設定最小子葉大小
# + 比較預測結果
#     + 與使用Mllib的結果差不多(0.4%)
# + 未來改良方向
#     + 減枝優化
#     + feature篩選優化
#     + random forest

# In[ ]:




