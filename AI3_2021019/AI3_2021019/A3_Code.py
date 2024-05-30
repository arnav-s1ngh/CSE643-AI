# (2 mark) Dataset : Load the wine quality dataset https://archive.ics.uci.edu/
# dataset/109/wine. The dataset is continuous and therefore discretization would
# be required to build the network. You can explore continuous/hybrid models also.
# Build classification model based on class variable in the data for performance evalua-
# tion (accuracy) https://scikit-learn.org/stable/modules/generated/sklearn.metrics.
# accuracy score.html of the network. You can use any available open-source pack-
# ages to build the network, example bnlearn package {https://pypi.org/project/ bnlearn/}.
#Section Begins----------------------------------------------------

from sklearn.metrics import accuracy_score
import pandas as pd
from pgmpy.metrics import correlation_score
from pgmpy.metrics import log_likelihood_score

import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.stats as stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import bnlearn as bn
data=pd.read_csv("wine.csv")
for i in data.columns[1:]:
    data[i] = pd.qcut(data[i], q=4, labels=False, duplicates='drop')

#Section Ends------------------------------------------------------

# (2 Mark) Construct a Bayesian network (A) for the data. Visualise the network and the probability distribution. Describe a few examples of parent and child nodes.
#Section Begins
A_original = bn.structure_learning.fit(data, methodtype='naivebayes',root_node='class')
A_original = bn.parameter_learning.fit(A_original, data)
# bn.plot(A_original)
#Section Ends

# (1 mark) Prune the network (A) for better performance on the class variable. Let the new network be (B). Explain your method of pruning.
#Section Begins
B = bn.independence_test(A_original, data, test='chi_square')
#These two had the two least weighted edges with the target
data=data.drop(columns='Ash')
data=data.drop(columns='Proanthocyanins')
bdata=data
B = bn.structure_learning.fit(data, methodtype='naivebayes',root_node='class')
B = bn.parameter_learning.fit(B, data)
X = data.drop(columns='class')
y = data['class']
BX_train, BX_test, By_train, By_test = train_test_split(X, y, test_size=0.33)
bn.plot(B)
#Section Ends

#(1.5 marks) Use methods other than pruning to construct the best model on the dataset. Let (A) be the new improved network.
#Section Begins
#Domain Knowledge, Hue is not very relavant (it tells stuff about the color intensity, but that already is a factor)
data=data.drop(columns='Hue')
adata=data
A_new = bn.structure_learning.fit(data, methodtype='naivebayes',root_node='class')
A_new = bn.parameter_learning.fit(A_new, data)
X = data.drop(columns='class')
y = data['class']
AX_train, AX_test, Ay_train, Ay_test = train_test_split(X, y, test_size=0.33)
bn.plot(A_new)
#Section Ends

#Section Begins
#Manually performing K-Best Feature Selection
lst=['class','Malic acid','Alcalinity of ash', 'Magnesium', 'Total phenols', 'Color intensity']
for i in data.columns:
    if i not in lst:
        data=data.drop(columns=i)
C = bn.structure_learning.fit(data, methodtype='naivebayes',root_node='class')
C = bn.parameter_learning.fit(C, data)
X = data.drop(columns='class')
y = data['class']
CX_train, CX_test, Cy_train, Cy_test = train_test_split(X, y, test_size=0.33)
query1 = bn.inference.fit(C, variables=['class'], evidence={'Magnesium': 2})
query2 = bn.inference.fit(C, variables=['class'], evidence={'Malic acid':1})
query3= bn.inference.fit(C, variables=['class'], evidence={'Color intensity':3})
query4 = bn.inference.fit(C, variables=['class'], evidence={'Total phenols': 1})
bn.plot(C)
#Section Ends
#Section Begins
AXtest = bn.sampling(A_new, n=1000)
BXtest = bn.sampling(B, n=1000)
CXtest = bn.sampling(C, n=1000)
A_pred=bn.predict(A_new,AXtest,variables=data.columns)
B_pred=bn.predict(B,BXtest,variables=data.columns)
C_pred=bn.predict(C,CXtest,variables=data.columns)
print(A_pred,B_pred,C_pred)
#Section Ends
















# pruned_model = bn.structure.bn_struct_bic(A)
# for i in range(1,13):
#     sns.histplot(data, ax=axes[i-1], x=data.columns[i], kde=True, color='r')
# plt.show()
