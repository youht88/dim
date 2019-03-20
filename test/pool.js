dim = require('../dim.js').dim
a=dim.array([[[1,2,3,4,5,6,7]]])
b=dim.array([[[1,2]]])
c=dim.nn.conv1d(a,b)
c.setGrad()
x=[]
d=dim.nn.maxPool1d(c,2,x)
e=dim.nn.maxUnpool1d(d,x,2)

a1=dim.arange(2*25*3).reshape(2,3,5,5)
b1=dim.arange(3*4*3).reshape(3,3,2,2)
c1=dim.nn.conv2d(a1,b1)
y=[]
d1=dim.nn.maxPool2d(c1,2,y)
e1=dim.nn.maxUnpool2d(d1,y,2)

d2=dim.nn.avgPool2d(c1,2)
e2=dim.nn.avgUnpool2d(d1,2)
