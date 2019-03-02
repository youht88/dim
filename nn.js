grad = require('./autograd.js')

class NN{
  constructor(dim){
    this.dim = dim
    this.grad = grad
    this.Vector = dim.Vector
    this.random=new Random(dim)
  }
  //base function
  backward(op,partial=null){
    return op.backward(partial)
  }
  eval(op){
    return op.eval()
  }
  //Classification Function
  softmax(x){
    x=this.dim.ensureVector(x);
    let exp,sum
    return new this.Vector(x.data.map((a,i)=>{
      if (a instanceof this.Vector) return this.softmax(a)
      if (i==0){
        exp = this.dim.exp(x.data)
        sum = exp.sum()      
      }
      let part1 = grad.Operate.wrapper(this.dim.Variable(a),null,"exp")
      let part2 = this.dim.Constant(sum - exp.data[i])
      let part3 = grad.Operate.wrapper(part1,part2,"add")
      let rst = grad.Operate.wrapper(part1,part3,"div")
      return rst
    }))
  }
  crossEntropy(a,y){
    [a,y]=this.dim.ensureVector(a,y)
    this.dim.ensure1D(a,y)
    this.dim.ensureSameShape(a,y)
    a=a.minmaxNormal()
    y=y.minmaxNormal()
    let part1=y.mul(a.log())
    let part2=y.neg().add(1)
    let part3=a.neg().add(1).log()
    let part4=part2.mul(part3)
    let part5=part1.add(part4).neg()
    let rst = part5.mean()
    //console.log(part1.value)
    //console.log(part2.value)
    //console.log(part3.value)
    //console.log(part4.value)
    //console.log(part5.value)
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
    return new this.Vector(x.data.map((a,i)=>{
      if (a instanceof this.Vector) return this.relu6(a)
      if (a<=0) return this.dim.Constant(0)
      if (a>6) return this.dim.Constant(6)
      return this.dim.Variable(this.dim.Variable(a))
    }))
  }
  softplus(x){
    x=this.dim.ensureVector(x);
    return new this.Vector(x.data.map((a,i)=>{
      if (a instanceof this.Vector) return this.softplus(a)
      let part1=grad.Operate.wrapper(this.dim.Variable(a),null,'exp')
      let part2=grad.Operate.wrapper(part1,1,'add')
      let rst=grad.Operate.wrapper(part2,null,'log')
      return rst
    }))
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

  sigmoid1(x){
    x=this.dim.ensureVector(x);
    return new this.Vector(x.data.map((a,i)=>{
      if (a instanceof this.Vector) return this.sigmoid(a)
      let part1=new grad.Constant(1)
      let part2=grad.Operate.wrapper(new grad.Constant(0),this.dim.Variable(a),'sub')
      let part3=grad.Operate.wrapper(part2,null,'exp')
      let part4=grad.Operate.wrapper(part1,part3,'add')
      let rst=grad.Operate.wrapper(part1,part4,'div')
      return rst
    }))
  }
  tanh(x){
    x=this.dim.ensureVector(x);
    return new this.Vector(x.data.map((a,i)=>{
      if (a instanceof this.Vector) return this.tanh(a)
      //also equal tanh=sinhx/conhx=(Math.exp(x)-Math.exp(-x))/(Math.exp(x)+Math.exp(-x))
      let part1=grad.Operate.wrapper(this.dim.Variable(a),null,"sinh")
      let part2=grad.Operate.wrapper(this.dim.Variable(a),null,"cosh")
      let rst=grad.Operate.wrapper(part1,part2,"div")
      return rst
   }))
  }
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
  
  //Pool Function
  maxPool(a){}
  avgPool(a){}
  meanPool(a){}
 
  //Loss Function
  MSELoss(a,y){
    //also named L2
    [a,y]=this.dim.ensureVector(a,y)
    return y.sub(a).square().mean()*0.5
  }
  logcoshLoss(a,y){
    [a,y]=this.dim.ensureVector(a,y)
    return y.sub(a).cosh().log().sum()
  }

  //cnn function
  conv2d(input, filter, strides, padding){
  
  }
  
}

exports.NN = NN