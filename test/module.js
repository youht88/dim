dim = require('../dim.js').dim
//x=dim.arange(12).reshape(3,4)
//y=dim.arange(3).reshape(3,1)
x=dim.random.randint(100,20,2)
y=x.slice([],0).sin().mul(x.slice([],1).cos()).reshape(20,1)
//x=x.normal(0)
//y=y.normal(0)
class Net extends dim.nn.Module{
  constructor(){
    super()
    this.layer1=dim.nn.Sequential()
    this.layer1.addModule("fc1",dim.nn.Linear(2,10))
    this.layer1.addModule("relu1",dim.nn.ReLU())
    this.layer2=dim.nn.Sequential(
      dim.nn.Linear(10,5),
      dim.nn.ReLU()
    )
    this.layer1.addModule("layer2",this.layer2)
    //crossEntropyLoss
    //this.layer1.addModule("out",dim.nn.Linear(1,10))
    //mseLoss
    this.layer1.addModule("out",dim.nn.Linear(5,1))
    //this.layer1.addModule("relu2",dim.nn.ReLU())
        
    this.addModule("all",this.layer1)
  }
  forward(x){
    return this.moduleList[0].module.forward(x)  
  }
}


let net = dim.nn.Module1(Net)
let preds=net.forward(x)
//let criterion=dim.nn.CrossEntropyLoss()  
let criterion=dim.nn.MSELoss()  
let optim = dim.optim.Adam(net.parameters(),{r:1})

for (let i=0;i<50000;i++){
  let loss = criterion.forward(preds,y)
  loss.backward()
  optim.step()
  optim.zeroGrad()
  loss.gradFn.setCatch(false)
  if (i%1000==0)console.log("epoch=",i,"loss=",loss.gradFn.eval().value)
}
y.print()
preds.gradFn.eval().print()


x1=dim.random.randint(100,20,2)
y1=x1.slice([],0).sin().mul(x1.slice([],1).cos()).reshape(20,1)
//x1=x1.normal(0)
//y1=y1.normal(0)
preds1=net.forward(x1)
preds1.print()
y1.print()
/*//test
let preds1=preds.gradFn.eval() //net(x1)
let hat=preds1.argmax(1).reshape(y.shape)
let total=y.shape[0]
let correct=hat.eq(y1).sum()
let accuracy=correct/total
console.log(`准确度为:%${accuracy*100}`)

/*
//conv2d layer
a=dim.arange(5*1*6*6).reshape(5,1,6,6)
conv2d=dim.nn.Sequential(
  dim.nn.Conv2d(1,3,3),
  dim.nn.MaxPool2d(2),
  dim.nn.ReLU()
)
m3=conv2d(a)
*/