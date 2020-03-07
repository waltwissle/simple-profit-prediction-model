import numpy as np
import matplotlib.pyplot as plt
import costFunction,gradientDescent
import time
from mpl_toolkits import mplot3d



# Prepare the data
data = np.loadtxt('maindata1.txt', delimiter = ',')

#print(data)
x_1 = data[:,0]
Y = data[:,1].reshape(97,1)

print('Visualize the data')
plt.figure(1)
plt.plot(x_1, Y, 'x', label = 'Training data')
plt.title('Plot of Population against Profits')
plt.xlabel('Population of cities in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend()
plt.show()
time.sleep(1.5) # pause for 1.5 secs

m = len(Y)
X = np.ones((m,2))
X[:,1] = x_1
theta = np.zeros((2,1))

# Gradient Decent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function...\n')

J = costFunction.costFunction(X, Y, theta)
print(f'\nwith theta at [0,0], the cost function is {J}')
print('Expected cost function value (approx) = 32.07\n ')
print('Program paused for 5.5 seconds\n')
time.sleep(1.5) # pause for 1.5 secs

# Further testing of the cost function
J = costFunction.costFunction(X, Y, [[-1],[2]])
print(f'With theta = [-1 ; 2]\n the cost computed is {J}')
print('Expected cost value (approx) 54.24\n')
print('Program paused for 5.5 seconds\n')
time.sleep(1.5) # pause for 1.5 secs

print('\nRunning Gradient Descent ...\n')
theta, costFunc = gradientDescent.gradientDescent(X, Y, theta, alpha, iterations)


#print theta to screen
print(f'Theta found by gradient descent: {theta}\n');
print('Expected theta values (approx)\n');
print(' -3.6303\n  1.1664\n\n');

plt.figure(2)
plt.plot(X[:,1], X @ theta, '-', color = 'red', label = 'Linear regression')
plt.plot(x_1, Y, 'x', label = 'Training data')
plt.title('Plot of Population against Profits')
plt.xlabel('Population of cities in 10,000s')
plt.ylabel('Profit in $10,000s')

plt.legend()
plt.show()

# Predict the profits for population sixe of 35,000 and 70,000

predict_1 =  (([1, 3.5] @ theta)*10000)[0] # for a population of 35000

print(f'For population = 35,000, we predict a profit of \n {predict_1}')

time.sleep(0.5) # pause for 0.5 secs

predict_2 = (([1, 7.0] @ theta)*10000)[0] # for a population of 70000

print(f'For population = 35,000, we predict a profit of \n {predict_2}')
time.sleep(1.5) # pause for 1.5 secs


## ================ Visualizing J(theta_0 and theha_1)================ ##

print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta0_vals)):
        t = [theta0_vals[i], theta1_vals[j]]
        J_vals[i,j] = costFunction.costFunction(X, Y, t)
        
J_vals = J_vals.T


ax = plt.axes(projection='3d')
#ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride = 2, cstride =2)
ax.set_zlabel('Population')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

plt.show()


# Contour Plot
plt.figure(4)
plt.contour(theta0_vals, theta1_vals, J_vals)
plt.xlabel(r'$\theta_0$'); plt.ylabel(r'$\theta_1$');
plt.plot(theta[0], theta[1], 'rx')
plt.show()












