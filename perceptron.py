import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

data = pd.read_csv('breastcancer.csv')

diagnosis = data['diagnosis'].to_list()
x1 = data['radius_mean'].to_list()
x2 = data['texture_mean'].to_list()

diagnosis = [0 if i=='M' else 1 for i in diagnosis]
n = len(x1)

w1=0
w2=0
bias=0
learningRate=0.01

def calc_z(w1,x1,w2,x2,b):
  return w1*x1+w2*x2+b 

def ycalc(z):
  if z>=0:
    return 1
  return 0

def new_w(learningRate,error,x):
  return learningRate*error*x 

def new_bias(learningRate,error):
  return learningRate*error

epochs = 520 #iterations
for epoch in range(epochs):
    for i in range(n):
        z = calc_z(w1, x1[i], w2, x2[i], bias)
        yCalculated = ycalc(z)
        error = diagnosis[i] - yCalculated
        # if error==0:
        #   break
        w1 += new_w(learningRate, error, x1[i])
        w2 += new_w(learningRate, error, x2[i])
        bias += new_bias(learningRate, error)

def predict(radius, texture):
    z = w1 * radius + w2 * texture + bias
    return 1 if z >= 0 else 0

radius = float(input("Enter the radius: "))
texture = float(input("Enter the texture: "))
print(f"{"Malignant" if predict(radius,texture)==0 else "Benign"} breast cancer.")

X = np.column_stack((x1,x2))
y = np.array(diagnosis)
for label in np.unique(y):
    plt.scatter(
        X[y == label, 0],  # x-axis: mean radius
        X[y == label, 1],  # y-axis: mean texture
        label=f"Class {label}"
    )

# decision boundary
x_vals = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 100)
y_vals = -(w1 * x_vals + bias) / w2  # Solve for x2

plt.plot(x_vals, y_vals, 'k--', label="Decision Boundary")

# Labels and legend
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.title("Perceptron Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()