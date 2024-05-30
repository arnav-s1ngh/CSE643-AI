import numpy as np
X=np.arange(-20,20,0.1)
np.random.shuffle(X)
rate=0.0000339981133456345218917431998345261
randomval=np.random.rand(400)*10
y=23*X+43+randomval
weight=0
#according to linear reg, answer is something close to, w=23 and b=48
#will have to modify learning rate and initial bias accordingly
i=0
bias=49.2
while i in range(100):
    prediction=weight*X
    prediction+=bias
    det_error=y-prediction
    weight+=rate*np.sum(X*det_error)
    bias+=rate*np.sum(det_error)
    i+=1
print("w = {}, b = {}".format(weight,bias))
