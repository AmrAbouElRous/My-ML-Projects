"""my custom made python functions"""
import numpy as np;
import matplotlib.pyplot as plt;
import math,copy
def get_housing_data():
    data = np.loadtxt("./data/houses.txt",delimiter=",",skiprows=1) 
    #delimiter default is space or u can choose how to separate between data
    #skiprows number of lines to ignore at the start of the txt file
    X = data[:,:4] 
    """all rows and columns( 0 index to 3 index)"""
    y = data[:,4]
    """all rows and column (4 index)"""
    return X,y

"""
Implement gradiant descent function for multiple variables
f(X) =w.X + b
compute_cost
j(w,b) = 1/2m sum of ((wX+b)-y)^2
compute_gradiant
dj/dw = 1/m   sum of ((wx+b)-y)*X
dj/db = 1/m   sum of ((wx+b)-y)

gradiant_descent
repeat {
            w = w - Learning rate * dj/dw
            b = b - Learning rate * dj/db
}
"""

def compute_cost(X,y,w,b):
    cost = 0
    m = X.shape[0]
    for i in range(m):
        cost += (np.dot(w,X[i])+b-y[i])**2

    cost = cost / (2*m)
    return cost




def compute_gradient(X,y,w,b):
    d_dw = np.zeros(X.shape[1])
    d_db = 0
    m = X.shape[0]
    for i in range(m):
        err = (np.dot(w,X[i])+b)-y[i]
        # d_dw = d_dw + np.dot(err,X[i]) #***** no need dot product when multipying scalar * vector
        d_dw = d_dw + (err*X[i])
        d_db = d_db + err
    d_dw = d_dw/m
    d_db = d_db/m
    return d_dw,d_db





def gradient_descent(X,y,w_in,b_in,alpha,iteration):
    w = copy.deepcopy(w_in)
    b = copy.deepcopy(b_in)
    J_history={}
    J_history["cost"] = []; J_history["params"] = []; J_history["grads"]=[]; J_history["iter"]=[]
    for i in range(iteration):
        dj_dw,dj_db =compute_gradient(X,y,w,b)
        w = w - (alpha*dj_dw) 
        b = b - (alpha*dj_db)
        J_history["cost"].append(compute_cost(X, y, w, b))
        J_history["params"].append([w,b])
        J_history["grads"].append([dj_dw,dj_db])
        J_history["iter"].append(i)
        if i % math.ceil(iteration/10) == 0:
            print(f"iteration {i:4d} w equals {w} and b equals {b} , cost equals {J_history['cost'][-1]}")
    return w,b,J_history


