dim = require('../dim.js').dim
a=dim.array([[1,2,3],[4,5,6]])
a.setGrad()
b=dim.array([[2],[1]])
c=dim.nn.crossEntropy(a,b)
