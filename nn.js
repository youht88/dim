grad = require('./autograd.js')


class NN{
  constructor(dim){
    this.dim = dim
    this.grad = grad
    this.Vector = dim.Vector
    this.random=new Random(dim)
  }
  Linear(input,outFeatures,bias){
    return new Linear(input,outFeatures,bias)
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
  conv1d(input, filter, stride=1, padding=0){
    if (input.shape.length!=3)
      throw new Error(`input(${input.shape})不符合[miniBatch*inChannels*W]的形状要求`) 
    if (filter.shape.length!=3)
      throw new Error(`filter(${filter.shape})不符合[outChannels*inChannels*W]的形状要求`) 
    if (input.shape[1]!=filter.shape[1]) 
      throw new Error(`input(${input.shape})与filter(${filter.shape})中channels数不一致`)
    return new dim.Vector(input.data.map((outChannel)=>{
       return new dim.Vector(filter.data.map((channel,ch)=>{
          return new dim.Vector(channel.data.map((kernel,k)=>{
            let bat = outChannel.data[k].pad(padding)
            let iw=bat.size
            let fw=kernel.size
            let a=[]
            let w=Math.floor((iw-fw)/stride+1)
            for (let j=0;j<w;j++){
              a[j]= bat.slice([j*stride,j*stride+fw]).mul(kernel).sum()
            }
            return a
          })).data.reduce((x,y)=>x.add(y))
       }))
    }))
  }
  conv2d(input, filter, stride=1, padding=0){
    if (input.shape.length!=4)
      throw new Error(`input(${input.shape})不符合[miniBatch*inChannels*H*W]的形状要求`) 
    if (filter.shape.length!=4)
      throw new Error(`filter(${filter.shape})不符合[outChannels*inChannels*H*W]的形状要求`) 
    if (input.shape[1]!=filter.shape[1]) 
      throw new Error(`input(${input.shape})与filter(${filter.shape})中channels数不一致`)
    return new dim.Vector(input.data.map((outChannel,bh)=>{
       return new dim.Vector(filter.data.map((channel,ch)=>{
         return new dim.Vector(channel.data.map((kernel,k)=>{
            let bat = outChannel.data[k].pad(padding)
            console.log(bat.shape,kernel.shape)
            let ih=bat.shape[0]
            let iw=bat.shape[1]
            let fh=kernel.shape[0]
            let fw=kernel.shape[1]
            let a=[]
            let w=Math.floor((iw-fw)/stride+1)
            let h=Math.floor((ih-fh)/stride+1)
            for (let i=0;i<h;i++){
              a[i]=[]
              for (let j=0;j<w;j++){
                a[i][j]=bat.slice([i*stride,i*stride+fh],[j*stride,j*stride+fw]).mul(kernel).sum()
             }
            }
            return new dim.Vector(a)
         })).data.reduce((x,y)=>x.add(y))
       }))
    }))
  }
  conv3d(input, filter, stride=1, padding=0){
    if (input.shape.length!=5)
      throw new Error(`input(${input.shape})不符合[miniBatch*inChannels*D*H*W]的形状要求`) 
    if (filter.shape.length!=5)
      throw new Error(`filter(${filter.shape})不符合[outChannels*inChannels*D*H*W]的形状要求`) 
    if (input.shape[1]!=filter.shape[1]) 
      throw new Error(`input(${input.shape})与filter(${filter.shape})中channels数不一致`)
    return new dim.Vector(input.data.map((outChannel)=>{
       return new dim.Vector(filter.data.map((channel,ch)=>{
         return new dim.Vector(channel.data.map((kernel,k)=>{
            let bat = outChannel.data[k].pad(padding)
            let id=bat.shape[0]
            let ih=bat.shape[1]
            let iw=bat.shape[2]
            let fd=kernel.shape[0]
            let fh=kernel.shape[1]
            let fw=kernel.shape[2]
            let a=[]
            let w=(iw-fw)/stride+1
            let h=(ih-fh)/stride+1
            let d=(id-fd)/stride+1
            for (let s=0;s<d;s++){
              a[s]=[]
              for (let i=0;i<h;i++){
                a[s][i]=[]
                for (let j=0;j<w;j++){
                  a[s][i][j]=bat.slice([s*stride,s*stride+fd],[i*stride,i*stride+fh],[j*stride,j*stride+fw]).mul(kernel).sum()
               }
              }
            }
            return new dim.Vector(a)
         })).data.reduce((x,y)=>x.add(y))
       }))
    }))
  }
  convTranspose1d(input, filter, stride=1, padding=0){
    if (input.shape.length!=3)
      throw new Error(`input(${input.shape})不符合[miniBatch*inChannels*W]的形状要求`) 
    if (filter.shape.length!=3)
      throw new Error(`filter(${filter.shape})不符合[inChannels*outChannels*W]的形状要求`) 
    if (input.shape[1]!=filter.shape[0]) 
      throw new Error(`input(${input.shape})与filter(${filter.shape})中channels数不一致`)
    //change channel
    filter = filter
    return new dim.Vector(input.data.map((outChannel)=>{
       return new dim.Vector(filter.data.map((channel,ch)=>{
          return new dim.Vector(channel.data.map((kernel,k)=>{
            let bat = outChannel.data[k].pad(padding)
            kernel = kernel.rot180()
            let iw=bat.size
            let fw=kernel.size
            let a=[]
            let w=Math.floor((iw-fw)/stride+1)
            for (let j=0;j<w;j++){
              a[j]= bat.slice([j*stride,j*stride+fw]).mul(kernel).sum()
            }
            return a
          })).data.reduce((x,y)=>x.add(y))
       }))
    }))
  }
  convTranspose2d(input, filter, stride=1, padding=0){
    //要实现还原运算，padding=((Out-1)*stride-Input+Filter)/2
    if (input.shape.length!=4)
      throw new Error(`input(${input.shape})不符合[miniBatch*inChannels*H*W]的形状要求`) 
    if (filter.shape.length!=4)
      throw new Error(`filter(${filter.shape})不符合[outChannels*inChannels*H*W]的形状要求`) 
    if (input.shape[1]!=filter.shape[0]) 
      throw new Error(`input(${input.shape})与filter(${filter.shape})中channels数不一致`)
    filter = dim.swapaxes(filter,0,1)
    return new dim.Vector(input.data.map((outChannel,bh)=>{
       return new dim.Vector(filter.data.map((channel,ch)=>{
         return new dim.Vector(channel.data.map((kernel,k)=>{
            let bat = outChannel.data[k].pad(padding)
            kernel = kernel.rot180()
            console.log(bat.shape,kernel.shape)
            let ih=bat.shape[0]
            let iw=bat.shape[1]
            let fh=kernel.shape[0]
            let fw=kernel.shape[1]
            let a=[]
            let w=Math.floor((iw-fw)/stride+1)
            let h=Math.floor((ih-fh)/stride+1)
            for (let i=0;i<h;i++){
              a[i]=[]
              for (let j=0;j<w;j++){
                a[i][j]=bat.slice([i*stride,i*stride+fh],[j*stride,j*stride+fw]).mul(kernel).sum()
             }
            }
            return new dim.Vector(a)
         })).data.reduce((x,y)=>x.add(y))
       }))
    }))  
  }
  convTranspose3d(){}
  //Pool Function
  maxPool1d(input,ks,indices,padding=0){
    if (!(Array.isArray(indices) && indices.length==0)) {
      console.log("⚠️警告：应当传入空的indices数组已接收argmax")
      indices=[]
    }
    let rst= new dim.Vector(input.data.map((channel,ci)=>{
      indices[ci]=[]
      return channel.data.map((kernel,ki)=>{
        indices[ci][ki]=[]
        kernel=kernel.pad(padding)
        let iw=kernel.size
        let fw=ks
        let a=[],b=[]
        let w=Math.floor((iw-fw)/ks+1)
        for (let j=0;j<w;j++){
          let k=kernel.slice([j*ks,j*ks+fw])
          a[j]=k.max()
          b[j]=k.argmax()
        }
        indices[ci][ki]=b
        return new dim.Vector(a)
      })
    }))
    return rst
  }
  avgPool1d(input,ks,padding=0){
    return new dim.Vector(input.data.map((channel)=>{
      return channel.data.map((kernel)=>{
        kernel=kernel.pad(padding)
        let iw=kernel.size
        let fw=ks
        let a=[]
        let w=Math.floor((iw-fw)/ks+1)
        for (let j=0;j<w;j++){
          let k=kernel.slice([j,j+fw])
          a[j]=k.mean()
        }
        return new dim.Vector(a)
      })
    }))
  }
  maxPool2d(input,ks,indices,padding=0){
    if (!(Array.isArray(indices) && indices.length==0)) {
      console.log("⚠️警告:必须传入空的indices数组已接收argmax")
      indices=[]
    }
    let rst= new dim.Vector(input.data.map((channel,ci)=>{
      indices[ci]=[]
      return channel.data.map((kernel,ki)=>{
        indices[ci][ki]=[]
        kernel = kernel.pad(padding)
        let ih=kernel.shape[0]
        let iw=kernel.shape[1]
        let fh=ks
        let fw=ks
        let a=[],b=[]
        let w=Math.floor((iw-fw)/ks+1)
        let h=Math.floor((ih-fh)/ks+1)
        console.log(iw,fw,w,ih,fh,h,ks)
        for (let i=0;i<h;i++){
          a[i]=[];b[i]=[]
          for (let j=0;j<w;j++){
            let k=kernel.slice([i*ks,i*ks+fh],[j*ks,j*ks+fw])
            a[i][j]=k.max()
            b[i][j]=k.argmax()
          }
        }
        indices[ci][ki]=b
        console.log(a)
        return new dim.Vector(a)
      })
    }))
    return rst
  }
  avgPool2d(input,ks,padding=0){
    return new dim.Vector(input.data.map((channel)=>{
       return channel.data.map((kernel)=>{
          kernel = kernel.pad(padding)
          let ih=kernel.shape[0]
          let iw=kernel.shape[1]
          let fh=ks
          let fw=ks
          let a=[]
          let w=Math.floor((iw-fw+2*padding)/ks+1)
          let h=Math.floor((ih-fh+2*padding)/ks+1)
          for (let i=0;i<h;i++){
            a[i]=[]
            for (let j=0;j<w;j++){
              a[i][j]=kernel.slice([i*ks,i*ks+fh],[j*ks,j*ks+fw]).mean()
           }
          }
          return new dim.Vector(a)
      })
    }))
  }
  maxPool3d(){}
  avgPool3d(){}
  maxUnpool1d(input,indices,ks){
    let rst= new dim.Vector(input.data.map((channel,ci)=>{
      return new dim.Vector(channel.data.map((kernel,ki)=>{
          let factor=[]
          for (let i=0;i<kernel.data.length;i++){
            let temp=dim.fill(0,ks)
            let r=indices[ci][ki][i]%ks
            temp.data[r]=1
            factor.push(temp)
          }
          factor=new dim.Vector(factor)
          return dim.kron(kernel,factor,true)
      }))
    }))
    return rst
  }
  avgUnpool1d(input,ks){
    let rst= new dim.Vector(input.data.map((channel,ci)=>{
      return new dim.Vector(channel.data.map((kernel,ki)=>{
          let factor = dim.fill(1/ks/ks,[ks,ks])
          return dim.kron(kernel,factor,false)
      }))
    }))
    return rst
  }
  maxUnpool2d(input,indices,ks){
    let rst= new dim.Vector(input.data.map((channel,ci)=>{
      return new dim.Vector(channel.data.map((kernel,ki)=>{
          let factor=[]
          let h=kernel.shape[0]
          let w=kernel.shape[1]
          for (let i=0;i<h;i++){
            for (let j=0;j<w;j++){
              let temp=dim.fill(0,[ks,ks])
              let r=Math.floor(indices[ci][ki][i][j]/ks)
              let c=indices[ci][ki][i][j]%ks
              temp.data[r].data[c]=1
              factor.push(temp)
            }
          }
          factor=new dim.Vector(factor)
          return dim.kron(kernel,factor,true)
      }))
    }))
    return rst
  }
  avgUnpool2d(input,ks){
    let rst= new dim.Vector(input.data.map((channel,ci)=>{
      return new dim.Vector(channel.data.map((kernel,ki)=>{
          let factor = dim.fill(1/ks/ks,[ks,ks])
          return dim.kron(kernel,factor,false)
      }))
    }))
    return rst
  }
  maxUnpool3d(){}
  avgUnpool3d(){}
}
class Module{
}
class Linear extends Module{
  constructor(input,outF,bias=true,){
    this.input = input
    this.weight=dim.random.random(input.shape[1],outF)
    this.weight.setGrad()
    if (bias){
      this.bias=dim.random.random(input.shape[0],outF)
      this.bias.setGrad()
    }
  }
  forward(){
    if (this.bias) return this.input.dot(this.weight).add(this.bias)
    return this.input.dot(this.weight)
  }
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
exports.Module = Module