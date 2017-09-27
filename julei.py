#!/usr/bin/python
#-*-coding:utf-8-*-

import pandas
import numpy 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

raw_data = pandas.read_csv('group.csv')
kmeans_model = KMeans(n_clusters=5, random_state=1)

x = raw_data.iloc[:,2].tolist()
y = raw_data.iloc[:,0].tolist()
z = raw_data.iloc[:,1].tolist()

data = numpy.array(list(zip(x,y,z))).reshape(len(x), 3)



senator_distances = kmeans_model.fit(data)

labels = kmeans_model.labels_


save = pandas.DataFrame({'s0':(raw_data.iloc[:, 0]).tolist(),'s10':(raw_data.iloc[:, 1]).tolist(),'s90' : (raw_data.iloc[:, 2]).tolist(),'s90d' : (raw_data.iloc[:, 3]).tolist(),'red_flag' : labels.tolist()})  
save.to_csv('test.csv',index=False,sep=',')


num1 = 0
num0 = 0
num2 = 0
num3 = 0

for item in labels.tolist():
	if(item == 0):
		num0 = num0 + 1
	elif(item == 1):
		num1 = num1 + 1
	elif(item == 2):
		num2 = num2 + 1
	elif(item == 3):
		num3 = num3 + 1

print "0     1      2      3"
print num0,num1,num2,num3



colors = ['b', 'g', 'r','p','o']  
for i,l in enumerate(kmeans.labels_):  
    plt.plot(x[i],y[i],color=colors[l],marker='0',ls='None')  
plt.show()  
  
#print type(labels.tolist()),type((raw_data.iloc[:, 2]).tolist())


