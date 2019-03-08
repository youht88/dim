grad = require('./autograd.js')


class NN{
  constructor(dim){
    this.dim = dim
    this.grad = grad
    this.Vector = dim.Vector
    this.random=new Random(dim)
  }
  //Classification Function
  softmax(x,axis){x=this.dim.ensureVector(x);return x.softmax(axis)}
  softmaxDeri(x,a){
    [x,a]=this.dim.ensureVector(x,a)
    let argmax=a.argmax(1).value
    let data=x.value
    let rst = new this.Vector(data.map((y,i)=>{
      return y.map((z,j)=>argmax[i]==j?z*(1-z):-z*y[argmax[i]])
    }))
    return rst
  }
  //Activation Function
  relu(x){
    x=this.dim.ensureVector(x);
    let rst= new this.Vector(x.data.map((a,i)=>{
      if (a instanceof this.Vector) return this.relu(a)
      return (a>0)?a:0
    }))
    if (x.requiresGrad){
      rst.requiresGrad=true
      rst.gradFn=grad.Operate.wrapper(x.gradFn,null,"relu")
    }
    return rst
  }
  reluDeri(x){
    x=this.dim.ensureVector(x);
    let rst= new this.Vector(x.data.map((a,i)=>{
      if (a instanceof this.Vector) return this.reluDeri(a)
      return (a>0)?1:0
    }))
    return rst
  }
  relu6(a){
    x=this.dim.ensureVector(x);
    let rst= new this.Vector(x.data.map((a,i)=>{
      if (a instanceof this.Vector) return this.relu(a)
      return (a>6)?6:(a<0)?0:a
    }))
    if (x.requiresGrad){
      rst.requiresGrad=true
      rst.gradFn=grad.Operate.wrapper(x.gradFn,null,"relu6")
    }
    return rst
  }
  relu6Deri(x){
    x=this.dim.ensureVector(x);
    let rst= new this.Vector(x.data.map((a,i)=>{
      if (a instanceof this.Vector) return this.reluDeri(a)
      return (a>6)?1:0
    }))
    return rst
  }
  softplus(x){
    x=this.dim.ensureVector(x);
    let rst = new this.Vector(x.data.map((a,i)=>{
      if (a instanceof this.Vector) return this.softplus(a)
      return Math.log(Math.exp(a)+1)
    }))
    if (x.requriesGrad){
      rst.requiresGrad=true
      let part1=grad.Operate.wrapper(x.gradFn,null,'exp')
      let part2=grad.Operate.wrapper(part1,1,'add')
      rst.gradFn=grad.Operate.wrapper(part2,null,'log')
    }
    return rst
  }
  sigmoid(x){
    x=this.dim.ensureVector(x);
    let rst = new this.Vector(x.data.map((a,i)=>{
      if (a instanceof this.Vector) return this.sigmoid(a)
      return 1/(1+Math.exp(-a))
    }))
    if (x.requiresGrad){
      rst.requiresGrad=true
      rst.gradFn=grad.Operate.wrapper(x.gradFn,null,"sigmoid")
    }
    return rst
  }
  sigmoidDeri(x){
    x=this.dim.ensureVector(x);
    let rst= new this.Vector(x.data.map((a,i)=>{
      if (a instanceof this.Vector) return this.sigmoidDeri(a)
      let y=1/(1+Math.exp(-a))
      return y*(1-y)
    }))
    return rst
  }

  tanh(x){x=this.dim.ensureVector(x);return x.tanh()}
  
  dropout(a,keep){
    if (keep<=0 || keep>1) throw new Error("keep_prob参数必须属于(0,1]")
    a=this.dim.ensureVector(a);
    let arr=[]
    return new this.Vector(a.data.map((x,i)=>{
      if (x instanceof this.Vector) return this.dropout(x,keep)
      if (i==0){
        let remain=a.data.length*keep
        for (let j=0;j<a.data.length;j++) arr.push(j)
        arr = this.random.shuffle(arr).slice(0,remain)
      }
      return (arr.indexOf(i)>=0)?x/keep:0
    }))
  }
   
  //Loss Function
  mseLoss(a,y){
    //also named L2
    [a,y]=this.dim.ensureVector(a,y)
    return y.sub(a).square().mean()
  }
  binaryCrossEntropy(a,y){
  }
  crossEntropy(a,y){
    [a,y]=this.dim.ensureVector(a,y)
    let b=this.softmax(a,1)
    let y_onehot=this.dim.onehot(y,b.shape[1])
    this.dim.ensureSameShape(b,y_onehot)
    let rst = y_onehot.mul(b.log()).sum(1).neg().mean()
    if (b.requiresGrad){
      rst = new dim.Vector([rst])
      rst.requiresGrad=true
      let leftFn,rightFn
      leftFn=(a.gradFn)?a.gradFn:new grad.Constant(a)
      rightFn=(y!=null)?((y.gradFn)?y.gradFn:new grad.Constant(y)):null
      rst.gradFn=grad.Operate.wrapper(leftFn,rightFn,"crossEntropy")
    }
    return rst
  }
  crossEntropyDeri(a,y){
    [a,y]=this.dim.ensureVector(a,y)
    let b=this.softmax(a,1)
    let y_onehot=this.dim.onehot(y,a.shape[1])
    let rst = new dim.Vector(b.sub(y_onehot).div(b.shape[0]))
    return rst
  }
  logcoshLoss(a,y){
    [a,y]=this.dim.ensureVector(a,y)
    return y.sub(a).cosh().log().sum()
  }

  //cnn function
  conv1d(input, filter, strides=1, padding=0){
    let ih=input.shape[0]
    let iw=input.shape[1]
    let fh=filter.shape[0]
    let fw=filter.shape[1]
    let a=[]
    for (let i=0;i<(iw-fw+2*padding)/strides+1;i++){
      a[i]=[]
      for (let j=0;j<(ih-fh+2*padding)/strides+1;j++){
        //console.log(i,j,input.slice([i,i+fh],[j,j+fw]).mul(filter).sum())
        a[i][j]=input.slice([i,i+fh],[j,j+fw]).mul(filter).sum()
     }
    }
    return new dim.Vector(a)
  }
  conv2d(input, filter, strides, padding){}
  conv3d(input, filter, strides, padding){}
  convTranspose1d(){}
  convTranspose2d(){}
  convTranspose3d(){}
  //Pool Function
  maxPool1d(a){}
  avgPool1d(a){}
  maxPool2d(a){}
  avgPool2d(a){}
  maxPool3d(a){}
  avgPool3d(a){}
  maxUnpool1d(a){}
  avgUnpool1d(a){}
  maxUnpool2d(a){}
  avgUnpool2d(a){}
  maxUnpool3d(a){}
  avgUnpool3d(a){}
}

class Optimizer{
  constructor(params){
    this.Optimizer(params)
  }
  Optimizer(params){
    if (params && !Array.isArray(params)) params=[params]
    if (params)
      this.params = params
    return this
  }
  step(){
    //console.log("this function have not been implemented")
    this.params.map(x=>{
      if (x.requiresGrad){
        x.sub_(x.grad.mul(this.lr))
      }
    })
  }
  zeroGrad(){
    //console.log("this function have not been implemented")
    this.params.map(x=>{
      if (x.requiresGrad){
        x.gradClear()
      }
    })
  }
  Adam(params,args){
    if (params && !Array.isArray(params)) params=[params]
    if (!params) params=this.params
    return new  Adam(params,args)
  }
}
class Adam extends Optimizer{
  constructor(params,args={}){
    super()
    if (params){
      this.params = params
    }
    this.lr  = args.lr || 0.001
    this.rho = args.rho || 0.9
    this.eps = args.eps || 1e-08
    this.weight_decay = args.weight_decay || 0
  }
}
exports.NN = NN
exports.Optimizer = Optimizer