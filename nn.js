np = require('./numpy.js').np
dim = require('./dim.js').dim

class NN{
  constructor(){
    this.random=new Random(np)
  }
  //Classification Function
  softmax(a){
    a=np.ensureVector(a);
    let exp = np.exp(a)
    let sum = exp.sum()
    return new Vector(a.data.map((x,i)=>exp.data[i]/sum))
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
    a=np.ensureNdarray(a);
    if (a instanceof Matrix)
      return new Matrix(a.data.map(x=>np.nn.relu(x)))
    return new Vector(a.data.map(x=>x<=0?0:x))
  }
  relu6(a){
    a=np.ensureNdarray(a);
    if (a instanceof Matrix)
      return new Matrix(a.data.map(x=>np.nn.relu6(x)))
    return new Vector(a.data.map(x=>x<=0?0:(x>6?6:x)))
  }
  softplus(a){
    a=np.ensureNdarray(a);
    if (a instanceof Matrix)
      return new Matrix(a.data.map(x=>np.nn.softplus(x)))
    return new Vector(a.data.map(x=>Math.log(Math.exp(x)+1)))
  }
  sigmoid(a){
    a=np.ensureNdarray(a);
    if (a instanceof Matrix)
      return new Matrix(a.data.map(x=>np.nn.sigmoid(x)))
    return new Vector(a.data.map(x=>1/(1+Math.exp(-x))))
  }
  tanh(a){
    a=np.ensureNdarray(a);
    //also equal tanh=sinhx/conhx=(Math.exp(x)-Math.exp(-x))/(Math.exp(x)+Math.exp(-x))
    if (a instanceof Matrix)
      return new Matrix(a.data.map(x=>np.nn.tanh(x)))
    return new Vector(a.data.map(x=>Math.tanh(x)))
  }
  tanhDeriv(a){
    a=np.ensureNdarray(a);
    return  a.tanh().square().neg().add(1)
  }
  sigmoidDeriv(a){
    a=np.ensureNdarray(a);
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

exports.nn = new NN()