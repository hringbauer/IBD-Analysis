'''
Created on Nov 28, 2016
Area created for testing tensor flow
@author: hringbauer
'''
import numpy as np
import tensorflow as tf
import pylab
import matplotlib.pyplot as plt


# 
# x = np.linspace(-1,1,10000)
# y= 1/ (1 + np.exp(-x))
# y1 = 0.5 + 0.25*x
# plt.figure()
# plt.plot(x,y,label="Inverese Logit")       # The Logit
# plt.plot(x,y1,label=r"$0.5+0.25x$")   # The identity shift
# plt.xlabel("f")
# plt.ylabel("Value")
# plt.legend(fontsize=20)
# #plt.legend()
# plt.show()

# def arc_sin_lin(x):
#     '''Arcus-Sinus Link function'''
#     x=np.where(x<0, 0, x)       # If x smaller 0 make it 0
#     x=np.where(x>np.pi, np.pi, x) # If x bigger Pi keep it Pi
#     y = np.sin(x / 2.0) ** 2
#     return y
# 
# #mean=np.array([np.pi/2.0, np.pi/2])
# mean=np.array([1, 1])
# cov=np.array([[0.05, 0.01],[0.01, 0.05]])   # Covariance Matrix
# #print(cov)
#   
# a=np.random.multivariate_normal(mean, cov, 10000000)    # Empirical simulate correlated Gaussian.
# f=np.transpose(a)                                      # Transpose so that later better 
#   
# print("\nCovariance of Gaussian Random Field f:\n")    
# print(np.cov(f))                                       # Empirical Covariance Matrix
#   
# #p = 1/(1+np.exp(-f)) 
# p = arc_sin_lin(f)                             # Logit Transform
# y = np.random.binomial(1, p)                           # Do binomial Draws
#   
#   
# m = np.mean(y, axis=1)
# print("\n Mean of draws:")
# print(m)
# 
# print("Covariance Factor: %.4f" % (m[0] * (1-m[0])))
#   
# print("\n Covariance of draws:")                       # Empirical Covariance of Draws
# print(np.cov(y))
a=np.linspace(0,100,10000)
b=a**2
print(len(a))

plt.figure()
plt.plot(a,b)
plt.show()



# WORKS


# with tf.device('/cpu:0'):
#     X = tf.Variable(dtype=tf.float64, initial_value=x, trainable=False)
#     y=tf.nn.sigmoid(x)
#     y1=tf.log(y)-tf.log(1-y)
# 
# with tf.Session() as sess:
#     r = sess.run([y1,y])
# 
# y=r[-1]
# y1=r[-2]
# print(y1-x)
# 
# plt.figure()
# plt.plot(x,y, label="Initial Function")
# plt.plot(x,y1,label="Inverse Function")
# plt.legend()
# plt.show()

