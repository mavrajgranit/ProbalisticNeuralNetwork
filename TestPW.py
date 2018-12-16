import torch
import torch.tensor as tensor
import torch.nn as nn
from Layers.ParzenWindow import ParzenWindow
import torch.optim as opt
import plotly
import plotly.graph_objs as go

#Just tests

#Mini-Batches currently not supported
x1 = tensor([2,2.5,3,1,6])
y1= tensor([0.0,1.0])

x2 = tensor([6,6.5,7,3,2])
y2 = tensor([1.0,0.0])

network = nn.Sequential(nn.Linear(5,5),nn.Sigmoid(),ParzenWindow(5,2))
optimizer = opt.SGD(network.parameters(),lr=0.005)
criterion = nn.MSELoss()

for i in range(1000):
    out1 = network.forward(x1)
    out2 = network.forward(x2)
    optimizer.zero_grad()
    loss1 = criterion(out1,y1)
    loss1.backward()
    loss2 = criterion(out2, y2)
    loss2.backward()
    optimizer.step()

print(network(x1))
print(network(x2))

network = nn.Sequential(nn.Linear(2,5),nn.Sigmoid(),ParzenWindow(5,2))
optimizer = opt.SGD(network.parameters(),lr=0.005)

x1 = tensor([-1.0,0.0])
y1= tensor([0.2,1.0])
x2 = tensor([0.2,0.8])
y2= tensor([0.6,0.3])
x3 = tensor([0.8,-0.4])
y3= tensor([0.8,1.0])

for i in range(10000):
    out1 = network.forward(x1)
    out2 = network.forward(x2)
    out3 = network.forward(x3)
    optimizer.zero_grad()
    loss1 = criterion(out1,y1)
    loss1.backward()
    loss2 = criterion(out2, y2)
    loss2.backward()
    loss3 = criterion(out3, y3)
    loss3.backward()
    optimizer.step()
print("new")
print(network(x1))
print(network(x2))
print(network(x3))

px = []
py = []
pz1 = []
pz2 = []
xs = -1.0
while xs<=1.0:
    ys = -1.0
    while ys <= 1.0:
        out = network(torch.tensor([xs,ys])).data.numpy()
        pz1.append(out[0])
        pz2.append(out[1])
        px.append(xs)
        py.append(ys)
        ys+=0.1
    xs+=0.1

trace = go.Scatter3d(
    x=px,
    y=py,
    z=pz1,
    mode='markers',
    marker={
        'size': 10,
        'opacity': 0.8,
    }
)

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)
data = [trace]
plot_figure = go.Figure(data=data, layout=layout)
plotly.offline.plot(plot_figure,filename="pwt1.html")

trace = go.Scatter3d(
    x=px,
    y=py,
    z=pz2,
    mode='markers',
    marker={
        'size': 10,
        'opacity': 0.8,
    }
)
layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)
data = [trace]
plot_figure = go.Figure(data=data, layout=layout)
plotly.offline.plot(plot_figure,filename="pwt2.html")

