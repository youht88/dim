np = require('./numpy.js').np

class NN{
  constructor(dim){
    this.dim = dim
    this.Vector = dim.Vector
    this.random=new Random(dim)
  }
  //Classification Function
  softmax(a){
    a=dim.ensureVector(a);
    let exp = dim.exp(a)
    let sum = exp.sum()      
    return new this.Vector(a.data.map((x,i)=>{
      if (x instanceof this.Vector) return this.softmax(x)
      return exp.data[i]/sum
    }))
  }
  crossEntropy(a,y){
    [a,y]=np.ensureNdarray(a,y)
    return np.mean(np.sum(
                    y.mul(np.log(a)).neg().add(
                      y.sub(1).mul(
                          np.log(a.neg().add(1))
                        )
                      )
                    )
                  )
  }

  //Activation Function
  relu(a){
    a=dim.ensureVector(a);
    return new this.Vector(a.data.map((x,i)=>{
      if (x instanceof this.Vector) return this.relu(x)
      return x<=0?0:x
    }))
  }
  relu6(a){
    a=np.ensureVector(a);
    return new this.Vector(a.data.map((x,i)=>{
      if (x instanceof this.Vector) return this.relu6(x)
      return x<=0?0:(x>6?6:x)
    }))
  }
  softplus(a){
    a=np.ensureVector(a);
    return new this.Vector(a.data.map((x,i)=>{
      if (x instanceof this.Vector) return this.softplus(x)
      return Math.log(Math.exp(x)+1)
    }))
  }
  sigmoid(a){
    a=np.ensureVector(a);
    return new this.Vector(a.data.map((x,i)=>{
      if (x instanceof this.Vector) return this.softplus(x)
      return 1/(1+Math.exp(-x))
    }))
  }
  tanh(a){
    a=np.ensureVector(a);
    //also equal tanh=sinhx/conhx=(Math.exp(x)-Math.exp(-x))/(Math.exp(x)+Math.exp(-x))
    return new this.Vector(a.data.map((x,i)=>{
      if (x instanceof this.Vector) return this.softplus(x)
      return Math.tanh(x)
    }))
  }
  tanhDeriv(a){
    a=np.ensureVector(a);
    return  a.tanh().square().neg().add(1)
  }
  sigmoidDeriv(a){
    a=np.ensureVector(a);
    return this.sigmoid(a).mul((this.sigmoid(a).neg().add(1)))
  }
  dropout(a,keep){
    a=np.ensureNdarray(a);
    if (keep<=0 || keep>1) throw new Error("keep_prob参数必须属于(0,1]")
    if (a instanceof Matrix)
      return new Matrix(a.data.map(x=>np.nn.dropout(x,keep)))
    let remain=a.length*keep
    let arr=[]
    for (let i=0;i<a.length;i++) arr.push(i)
    arr = this.random.shuffle(arr)
    arr = arr.slice(0,remain)
    return new Vector(a.data.map((x,i)=>(arr.indexOf(i)>=0)?x/keep:0))
  }
  
  //Pool Function
  maxPool(a){}
  avgPool(a){}
  meanPool(a){}
 
  //Loss Function
  MSE(a,y){
    //also named L2
    [a,y]=np.ensureNdarray(a,y)
    return np.square(a.sub(y)).mean()*0.5
  }
  logcosh(a,y){
    [a,y]=np.ensureNdarray(a,y)
    return np.log(np.cosh(a.sub(y))).sum()
  }

  //cnn function
  conv2d(input, filter, strides, padding){
  
  }
  
}

exports.NN = NN