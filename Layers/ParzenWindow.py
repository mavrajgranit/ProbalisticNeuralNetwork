import torch, math
import torch.nn as nn
#import torch.optim as opt
#import plotly
#import plotly.graph_objs as go

#After reading the "Pattern3.pdf" I think we should learn x and Sigma(?)

#RESOURCES
#http://www.personal.reading.ac.uk/~sis01xh/teaching/CY2D2/Pattern3.pdf
#http://www.ehu.eus/ccwintco/uploads/8/89/Borja-Parzen-windows.pdf

class ParzenWindow(nn.Module):

    def __init__(self, in_features, out_features, std_init=0.4):
        super(ParzenWindow, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.mikro = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        invsqr = 1 / math.sqrt(self.in_features)
        mu_range = invsqr
        self.mikro.data.uniform_(-mu_range, mu_range)
        self.sigma.data.uniform_(mu_range)

    def forward(self, input):
        '''
        inp = [input]
        for i in range(self.out_features-1):
            inp.append(input)
        inp = torch.stack(inp)'''
        sum = torch.sum((self.mikro-input) ** 2,1)
        return torch.exp(-sum)/ (2*self.sigma ** 2)

'''Beatiful Plots
pw = ParzenWindow(2,2)
optimizer = opt.SGD(pw.parameters(),lr=0.01)
criterion = nn.MSELoss()
x = torch.tensor([1.0,0.0])
y = torch.tensor([0.0,1.0])

for i in range(1500):
    out = pw.forward(x)
    if((i+1)%1500==0):print(out)
    optimizer.zero_grad()
    loss = criterion(out,y)
    loss.backward()
    optimizer.step()

px = []
py = []
pz1 = []
pz2 = []
xs = -1.0
while xs<=1.0:
    ys = -1.0
    while ys <= 1.0:
        out = pw(torch.tensor([xs,ys])).data.numpy()
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
plotly.offline.plot(plot_figure,filename="pw1.html")

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
plotly.offline.plot(plot_figure,filename="pw2.html")
'''