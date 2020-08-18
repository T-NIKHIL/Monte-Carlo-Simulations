# This program is an RK-4th order program for simultaneous solution of set of differential equations

import numpy as np
# For general plotting
import matplotlib.pyplot as plt
# For creating dataframe to store data for pairwise joint pdf
import pandas as pd
# For creating the joint pairwise pdf
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------------------------------------------------
# PART 1 INITIAL CONDITIONS KNOWN (x(0)=y(0)=z(0)=1)
# ----------------------------------------------------------------------------------------

t0 = 0
inputFile = open("input.txt","r")
inputList = inputFile.readlines()
tf = int(inputList[0])
print("Entered end time : " + str(tf))
# number of steps
n = 100*tf

h = (tf - t0)/n

# Defining the differential equations

def f1(t,x,y,z):
	return (10*(y-x))

def f2(t,x,y,z):
	return (x*(28-z)-y)

def f3(t,x,y,z):
	return ((x*y)-(8*z/3))

# RK-4th order function

def RK4(IC,n):

	# Creating solution and time arrays
	# Number of elements in array is n+1
	t = np.zeros(n+1, dtype=float)
	x = np.zeros(n+1, dtype=float)
	y = np.zeros(n+1, dtype=float)
	z = np.zeros(n+1, dtype=float)

	# Adding initial conditions to arrays
	t[0] = t0
	x[0] = IC[0]
	y[0] = IC[1]
	z[0] = IC[2]

	for i in range(0,n):

		k1 = h*f1(t[i],x[i],y[i],z[i])
		l1 = h*f2(t[i],x[i],y[i],z[i])
		m1 = h*f3(t[i],x[i],y[i],z[i])

		k2 = h*f1(t[i]+h/2, x[i]+k1/2, y[i]+l1/2, z[i]+m1/2)
		l2 = h*f2(t[i]+h/2, x[i]+k1/2, y[i]+l1/2, z[i]+m1/2)
		m2 = h*f3(t[i]+h/2, x[i]+k1/2, y[i]+l1/2, z[i]+m1/2)

		k3 = h*f1(t[i]+h/2, x[i]+k2/2, y[i]+l2/2, z[i]+m2/2)
		l3 = h*f2(t[i]+h/2, x[i]+k2/2, y[i]+l2/2, z[i]+m2/2)
		m3 = h*f3(t[i]+h/2, x[i]+k2/2, y[i]+l2/2, z[i]+m2/2)

		k4 = h*f1(t[i]+h, x[i]+k3, y[i]+l3, z[i]+m3)
		l4 = h*f2(t[i]+h, x[i]+k3, y[i]+l3, z[i]+m3)
		m4 = h*f3(t[i]+h, x[i]+k3, y[i]+l3, z[i]+m3)


		x[i+1] = x[i] + (k1 + 2*k2 + 2*k3 + k4)/6
		y[i+1] = y[i] + (l1 + 2*l2 + 2*l3 + l4)/6
		z[i+1] = z[i] + (m1 + 2*m2 + 2*m3 + m4)/6
		t[i+1] = t[i] + h

	return t,x,y,z

ic = [1,1,1]
t,x,y,z = RK4(ic,n)

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.plot(t,x)
ax1.set_xlabel("t")
ax1.set_ylabel("x")
ax1.set_title("x vs t")

ax2.plot(t,y)
ax2.set_xlabel("t")
ax2.set_ylabel("y")
ax2.set_title("y vs t")

ax3.plot(t,z)
ax3.set_xlabel("t")
ax3.set_ylabel("z")
ax3.set_title("z vs t")

plt.show()

# ----------------------------------------------------------------------------------------
# PART 2 INITIAL CONDITION IS NOT KNOWN (SAMPLING FROM MULTIVARIATE GAUSSIAN DISTRIBUTION)
# ----------------------------------------------------------------------------------------

# Multivariate gaussian distribution specified by mean and covariance matrix 
# Univariate gaussian distributions are specified by mean and standard deviation

mean = [1,1,1]
cov = [[1,0,0],[0,1,0],[0,0,1]]

x_sample,y_sample,z_sample = np.random.multivariate_normal(mean, cov, 10000).T

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x,y,z,s=1,c=None, depthshade=True)

# plt.show()

# The below matrices store the x,y and z values after each iteration with different initial conditions
X = np.zeros((10000,n+1), dtype=float)
Y = np.zeros((10000,n+1), dtype=float)
Z = np.zeros((10000,n+1), dtype=float)

for j in range(0,10000):
	IC = [x_sample[j],y_sample[j],z_sample[j]]
	t,X[j][:],Y[j][:],Z[j][:] = RK4(IC,n)
	if j%100 == 0:
		print("Iteration" + " " + str(j))

# Lets extract data at 5 time steps n = 0,250,500,750,1000
# Extracting data at n = 0
dataXY_0 = np.concatenate((X[:,0].reshape(-1,1),Y[:,0].reshape(-1,1)),axis=1)
dataYZ_0 = np.concatenate((Y[:,0].reshape(-1,1),Z[:,0].reshape(-1,1)),axis=1)
dataZX_0 = np.concatenate((Z[:,0].reshape(-1,1),X[:,0].reshape(-1,1)),axis=1)

dfXY_0 = pd.DataFrame(dataXY_0,columns=["X","Y"])
dfYZ_0 = pd.DataFrame(dataYZ_0,columns=["Y","Z"])
dfZX_0 = pd.DataFrame(dataZX_0,columns=["Z","X"])

sns.jointplot(x="X",y="Y", data=dfXY_0, color="b", space=0, joint_kws={'s': 1})
sns.jointplot(x="Y",y="Z", data=dfYZ_0, color="b", space=0, joint_kws={'s': 1})
sns.jointplot(x="Z",y="X", data=dfZX_0, color="b", space=0, joint_kws={'s': 1})

# Extracting data at n = 250
dataXY_250 = np.concatenate((X[:,250].reshape(-1,1),Y[:,250].reshape(-1,1)),axis=1)
dataYZ_250 = np.concatenate((Y[:,250].reshape(-1,1),Z[:,250].reshape(-1,1)),axis=1)
dataZX_250 = np.concatenate((Z[:,250].reshape(-1,1),X[:,250].reshape(-1,1)),axis=1)

dfXY_250 = pd.DataFrame(dataXY_250,columns=["X","Y"])
dfYZ_250 = pd.DataFrame(dataYZ_250,columns=["Y","Z"])
dfZX_250 = pd.DataFrame(dataZX_250,columns=["Z","X"])

sns.jointplot(x="X",y="Y", data=dfXY_250, color="g", space=0, joint_kws={'s': 1})
sns.jointplot(x="Y",y="Z", data=dfYZ_250, color="g", space=0, joint_kws={'s': 1})
sns.jointplot(x="Z",y="X", data=dfZX_250, color="g", space=0, joint_kws={'s': 1})

# Extracting data at n = 500
dataXY_500 = np.concatenate((X[:,500].reshape(-1,1),Y[:,500].reshape(-1,1)),axis=1)
dataYZ_500 = np.concatenate((Y[:,500].reshape(-1,1),Z[:,500].reshape(-1,1)),axis=1)
dataZX_500 = np.concatenate((Z[:,500].reshape(-1,1),X[:,500].reshape(-1,1)),axis=1)

dfXY_500 = pd.DataFrame(dataXY_500,columns=["X","Y"])
dfYZ_500 = pd.DataFrame(dataYZ_500,columns=["Y","Z"])
dfZX_500 = pd.DataFrame(dataZX_500,columns=["Z","X"])

sns.jointplot(x="X",y="Y", data=dfXY_500, color="y", space=0, joint_kws={'s': 1})
sns.jointplot(x="Y",y="Z", data=dfYZ_500, color="y", space=0, joint_kws={'s': 1})
sns.jointplot(x="Z",y="X", data=dfZX_500, color="y", space=0, joint_kws={'s': 1})

# Extracting data at n = 750
dataXY_750 = np.concatenate((X[:,750].reshape(-1,1),Y[:,750].reshape(-1,1)),axis=1)
dataYZ_750 = np.concatenate((Y[:,750].reshape(-1,1),Z[:,750].reshape(-1,1)),axis=1)
dataZX_750 = np.concatenate((Z[:,750].reshape(-1,1),X[:,750].reshape(-1,1)),axis=1)

dfXY_750 = pd.DataFrame(dataXY_750,columns=["X","Y"])
dfYZ_750 = pd.DataFrame(dataYZ_750,columns=["Y","Z"])
dfZX_750 = pd.DataFrame(dataZX_750,columns=["Z","X"])

sns.jointplot(x="X",y="Y", data=dfXY_750, color="#ffa500", space=0, joint_kws={'s': 1})
sns.jointplot(x="Y",y="Z", data=dfYZ_750, color="#ffa500", space=0, joint_kws={'s': 1})
sns.jointplot(x="Z",y="X", data=dfZX_750, color="#ffa500", space=0, joint_kws={'s': 1})

# Extracting data at n = 1000
dataXY_1000 = np.concatenate((X[:,1000].reshape(-1,1),Y[:,1000].reshape(-1,1)),axis=1)
dataYZ_1000 = np.concatenate((Y[:,1000].reshape(-1,1),Z[:,1000].reshape(-1,1)),axis=1)
dataZX_1000 = np.concatenate((Z[:,1000].reshape(-1,1),X[:,1000].reshape(-1,1)),axis=1)

dfXY_1000 = pd.DataFrame(dataXY_1000,columns=["X","Y"])
dfYZ_1000 = pd.DataFrame(dataYZ_1000,columns=["Y","Z"])
dfZX_1000 = pd.DataFrame(dataZX_1000,columns=["Z","X"])

sns.jointplot(x="X",y="Y", data=dfXY_1000, color="r", space=0, joint_kws={'s': 1})
sns.jointplot(x="Y",y="Z", data=dfYZ_1000, color="r", space=0, joint_kws={'s': 1})
sns.jointplot(x="Z",y="X", data=dfZX_1000, color="r", space=0, joint_kws={'s': 1})

plt.show()