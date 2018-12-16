import numpy, math
import matplotlib.pyplot as plt

x1 = [[2],[2.5],[3],[1],[6]]
x2 = [[6],[6.5],[7]]

sigma = 1
pi = math.sqrt(3.14159265359*2)

def gauss(xs,data):
    sum = 0
    #a = 1 / (x * pi)
    for ds in data:
        sos = 0
        for d in range(len(ds)):
            sos += (ds[d]-xs[d])**2
        sum += math.exp(-sos/(2*sigma**2))
    return sum/len(data)

c1 = gauss([5],x1)
print(c1)
c2 = gauss([3],x2)
print(c2)

x3 = [[2,4]]
c3 = gauss([3,1],x3)
print(c3)

x4 = [[1,0],[0,1],[1,1]]
x5 = [[-1,0],[0,-1]]
c4 = gauss([0.5,0.5],x4)
print(c4)
c5 = gauss([0.5,0.5],x5)
print(c5)

#Plot
i=numpy.linspace(-12,12,100)
x = []
for e in i:
    x.append(gauss([e],x1))
plt.plot(x)

x = []
for e in i:
    x.append(gauss([e],x2))
plt.plot(x)

plt.show()