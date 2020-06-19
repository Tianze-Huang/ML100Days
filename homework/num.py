import numpy as np
import matplotlib.pylab as plt

def mean_absolute_error(y,yp):
    mae=MAE=sum(abs(y-yp))/len(y)
    return mae

def mean_squared_error(y,yp):
    mse=MSE=sum((y-yp)**2)
    return mse
    
w=3
b=0.5
x_lin=np.linspace(0,100,101)
y=(x_lin+np.random.randn(101)*5)*w+b

plt.plot(x_lin,y,'b.',label='data points')
plt.title("Assume we have data points")
plt.legend(loc=2)
plt.show()

y_hat=x_lin*w+b
plt.plot(x_lin,y,'b.',label='data')
plt.plot(x_lin,y_hat,'r-',label='prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc=2)
plt.show()

MSE=mean_squared_error(y,y_hat)
MAE=mean_absolute_error(y,y_hat)
print("The Mean squared error is %.3f" %(MSE))
print("The Mean absolute error is %.3f" %(MAE))


