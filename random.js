class Random{
  constructor(np){
    this.np = np
  }
  random(shape,dtype){//随机矩阵
    return this.np.__reset((i,args)=>{
        return Math.random()
      },dtype,shape)
  }
  rand(shape,dtype){return this.random(shape,dtype)}
  randint(range,shape,dtype){
    //range may be a number or a array like as [m,n]
    return this.np.__reset((i,args)=>{
        let range=args[0]
        if (typeof range=="number") range=[0,range]
        let [m,n]=range
        return parseInt(Math.random()*(n-m)+m)
      },dtype,shape,range)
  }
  randn(shape,dtype){//正态分布
    let v=this.np.__reset((i,args)=>{
        return Math.random()
      },dtype,shape)
    return v.sub(v.mean()).div(v.std())
  }
  normal([mu,sigma],shape,dtype){//随机矩阵
    let v=this.randn(shape,dtype)
    return v.mul(sigma).add(mu)
  }
  choice(a,n){
    if (typeof a == "number") {
      let t=[]
      for (let i=0;i<a;i++) t.push(i);
      a=t 
    }
    a=this.np.ensureVector(a)
    return this.shuffle(a.data).slice(0,n)
  }
  shuffle(aArr){
    var iLength = aArr.length,
      i = iLength,
      mTemp,
      iRandom;
    while(i--){
      if(i !== (iRandom = Math.floor(Math.random() * iLength))){
        mTemp = aArr[i];
        aArr[i] = aArr[iRandom];
        aArr[iRandom] = mTemp;
      }
    }
    return aArr;
  }
}

exports.Random = Random