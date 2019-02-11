fs=require('fs')
Random = require('./random.js').Random
Complex = require('./complex.js').Complex
Poly = require('./poly.js').Poly

fft = require('./fft.js').fft

class Vector{
  constructor(object,dtype=Float32Array){
    if (Array.isArray(object)) {
      if (Array.isArray(object[0])){
        return new Vector(object.map(x=>new Vector(x,dtype)),dtype)
      }
    }
    if (object instanceof Vector){
      return object
    }
    //自动判定复数
    if (Array.isArray(object) && (object[0] instanceof Complex)) dtype=Complex
    this.dtype = dtype
    this.data = this.ensureArray(object)

    this.shape=this.getShape()
  }
  ensureArray(data){
    if (this.dtype == Complex){
      if (Array.isArray(data)){
        return data.map(x=>{
          if (x instanceof Vector) return x
          if (x instanceof Complex) return x
          return new Complex(x)
        })
      }else{
        let c=[]
        for(let i=0;i<data;i++){
          c.push(new Complex())
        }
        return c
      }
    }else {
      return Array.isArray(data) && data || new this.dtype(data)
    }
  }
  get real(){
    if (this.dtype == Complex){
        return this.data.map(x=>x.real)
      }else{
        return this.data
      }
  }
  get imag(){
    if (this.dtype == Complex){
        return this.data.map(x=>x.imag)
      }else{
        return new Float32Array(this.real.length)
      }
  }
  get size(){return this.shape.reduce((a,b)=>a*b)}
  get ndim(){
    let i=0
    let a=this.data
    while (Array.isArray(a)){
      i++
      a=a[0].data?a[0].data:a[0]
    }
    return i
  }
  get T(){return this.transpose()}
  
  get value(){
    return this.data.map((x,i)=>(x instanceof Vector)?x.value:x)
  }
  cast(dtype){
    if (typeof dtype == "string" ){
      switch (dtype.toUpperCase()){
        case "INTEGER": dtype=Int32Array;break
        case "INT": dtype=Int32Array;break
        case "FLOAT"  : dtype=Float32Array;break;
        case "STRING" : dtype=String;break;      
        case "STR" : dtype=String;break;      
        case "BOOLEAN": dtype=Boolean;break;      
        case "BOOL": dtype=Boolean;break;      
      }
    }
    let result = this.flatten().data.map(item=>{
      if (dtype == Complex)    return item instanceof Complex?item:new Complex(item)
      if (dtype==Int32Array)   return parseInt(typeof(item)=="boolean"?(item?1:0):item)
      if (dtype==Float32Array) return parseFloat(typeof(item)=="boolean"?(item?1:0):item)
      if (dtype == String)         return String(item)      
      if (dtype == Boolean)        return Boolean(item)
      throw new Error("类型转换定义不合法")
    })
    return new Vector(result,dtype).reshape(this.shape)
  }
  getShape(){
    let shape=[]
    let data=this.data
    for (let i=0;i<this.ndim;i++){
      shape.push(data.length)
      data=data[0].data?data[0].data:data[0]
    }
    return shape
  }
  toVector(a,dtype){
    if (a instanceof Vector) return a
    return new Vector(a,dtype)
  }
  ensureSameShape(a){
    if (typeof a=="number") return 
    if (this.shape.toString()!=a.shape.toString()) throw new Error(`形状(${this.shape})与形状(${a.shape})不一致`)
  }
  ensureMatrix(...a){
    a.map(x=>{if (x.ndim==2) throw new Error(`要求是2维矩阵,但是参数是(${x.ndim})维`)})
  }
  ensureSquareMatrix(...a){
    a.map(x=>{if (x.ndim!=2 || x.shape[0]!=x.shape[1]) 
      throw new Error(`要求是方阵，但是参数形状是(${x.shape}),(${x.ndim})维`)
    }) 
  }
  ensureCanDot(a){
    if (typeof a =="number") return
    if (this.ndim!=2 && this.ndim!=1) throw new Error(`参数是(${a.ndim})维,仅支持一、二维`)
    if (this.ndim==1 && a.ndim!=1) throw new Error(`要求是一维向量，但是参数是(${a.ndim})维`)
    if (this.ndim==2 && a.ndim!=2) throw new Error(`要求是二维向量，但是参数是(${a.ndim})维`)
    if (this.ndim==2 && this.shape[1]!=a.shape[0]) throw new Error(`(${this.shape})和(${a.shape})形状不符合要求`)
  }
  
  flatten(item){
    if (!item) item=[]
    this.data.map((x,i)=>(x instanceof Vector)?x.flatten(item):item.push(x))
    //console.log(item)
    return new Vector(item)
  }

  copy(){return this.toVector(this.value,this.dtype)}
  save(file){return fs.writeFileSync(file,JSON.stringify(this.value))}

  reshape(...d){
    if (Array.isArray(d[0])) d=d[0]
    let a=this.flatten().data
    let t,p=[],plen=0
    plen=d[d.length - 1]
    if (this.size!=d.reduce((a,b)=>a*b)) 
      throw new Error(`尺寸为(${this.size})的数组无法匹配形状(${d})`)
    for (let i=0;i<this.size;i+=plen){
      p.push(a.slice(i,i+plen))
    }
    t=p
    let size = this.size / plen
    for (let i=d.length - 2 ;i>0;i--){
      size = size/d[i]
      t=[]
      for (let j=0;j<size;j++){
        t.push(p.slice(j*d[i],j*d[i]+d[i]))
      }
      p=[...t]
    }
    return this.toVector(t,this.dtype)
  }
  flat_idx(indices){
    const shape = this.shape

    if( indices.length != shape.length ) throw new Error(`Multi-index [${indices}] does not have expected length of ${shape.length}.`);
    
    let
      flat_idx = 0,
      stride = 1;
    for( let i=shape.length; i-- > 0; stride *= shape[i] )
    {
      let idx = indices[i];
      if( idx % 1 != 0 ) throw new Error(`Multi-index [${indices}] contains non-integer entries.`);
      if( idx < 0 )  idx += shape[i]
      if( idx < 0 || idx >= shape[i] ) throw new Error(`Multi-index [${indices}] out of bounds [${shape}].`);
      flat_idx  +=   idx * stride;
    }
    return flat_idx;
  }
  transpose(deep=0,ndim){
    /*if (!ases) ases=[...this.shape].reverse()
    let set = new Set(axes)
    if (set.size!=axes.length) throw new Error(`axes有重复的参数`)
    */
    if (!ndim) ndim=this.ndim
    if (deep==ndim - 2){
      let T = this.value.reduce((a,b)=>
         a.map((x,i)=>x[0]!=undefined?x.concat(b[i]):[x].concat(b[i])))
      if (ndim==2) return this.toVector(T)
      return T
    }
    let result = this.data.map((x,i)=>{
      if (i==0) deep++
      return (x instanceof Vector)?x.transpose(deep,ndim):x 
    })
    return this.toVector(result)
  }

  sin(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sin():Math.sin(x)))}
  cos(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.cos():Math.cos(x)))}
  tan(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.tan():Math.tan(x)))}
  asin(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.asin():Math.asin(x)))}
  acos(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.acos():Math.acos(x)))}
  atan(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.atan():Math.atan(x)))}
  asinh(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.asinh():Math.asinh(x)))}
  acosh(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.acosh():Math.acosh(x)))}
  atanh(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.atanh():Math.atanh(x)))}
  sinh(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sinh():Math.sinh(x)))}
  cosh(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.cosh():Math.cosh(x)))}
  tanh(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.tanh():Math.tanh(x)))}
  log(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.log():Math.log(x)))}
  log2(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.log2():Math.log2(x)))}
  log10(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.log10():Math.log10(x)))}
  exp(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.exp():Math.exp(x)))}
  sqrt(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sqrt():Math.sqrt(x)))}
  square(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.square():Math.pow(x,2)))}
  pow(n){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.pow(n):Math.pow(x,n)))}
  floor(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.floor():Math.floor(x)))}
  ceil(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.ceil():Math.ceil(x)))}
  around(n){return new Vector(this.data.map((x,i)=>{
      if (x instanceof Vector) return x.around(n)
      let a=10**n
      return Math.round(x*a)/a
    }))
  }

  add(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.add(a.data?a.data[i]:a):x.add(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.add(a.data?a.data[i]:a):x+(a.data?a.data[i]:a)))
  }
  sub(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sub(a.data?a.data[i]:a):x.sub(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sub(a.data?a.data[i]:a):x-(a.data?a.data[i]:a)))
  }
  mul(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.mul(a.data?a.data[i]:a):x.mul(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.mul(a.data?a.data[i]:a):x*(a.data?a.data[i]:a)))
  }
  div(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.div(a.data?a.data[i]:a):x.div(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.div(a.data?a.data[i]:a):x/(a.data?a.data[i]:a)))
  }
  power(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.power(a.data?a.data[i]:a):x.power(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.power(a.data?a.data[i]:a):x**(a.data?a.data[i]:a)))
  }
  mod(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.mod(a.data?a.data[i]:a):x.mod(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.mod(a.data?a.data[i]:a):x%(a.data?a.data[i]:a)))
  }
  subtract(x){return this.sub(x)}
  multiply(x){return this.mul(x)}
  divide(x){return this.div(x)}
  neg(){return this.mul(-1)}
  
  reciprocal(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.reciprocal():1/x))}
  sign(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sign():Math.sign(x)))}

  gt(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.gt(a.data?a.data[i]:a):x.gt(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.gt(a.data?a.data[i]:a):x>(a.data?a.data[i]:a)?true:false),Boolean)
  }
  gte(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.gte(a.data?a.data[i]:a):x.gte(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.gte(a.data?a.data[i]:a):x>=(a.data?a.data[i]:a)?true:false),Boolean)
  }
  lt(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.lt(a.data?a.data[i]:a):x.lt(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.lt(a.data?a.data[i]:a):x<(a.data?a.data[i]:a)?true:false),Boolean)
  }
  lte(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.lte(a.data?a.data[i]:a):x.lte(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.lte(a.data?a.data[i]:a):x<=(a.data?a.data[i]:a)?true:false),Boolean)
  }
  eq(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.eq(a.data?a.data[i]:a):x.eq(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.eq(a.data?a.data[i]:a):x=(a.data?a.data[i]:a)?true:false),Boolean)
  }
  ne(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.ne(a.data?a.data[i]:a):x.ne(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.ne(a.data?a.data[i]:a):x!=(a.data?a.data[i]:a)?true:false),Boolean)
  }
  close(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.close(a.data?a.data[i]:a):x.close(a.data?a.data[i]:a)))
    return new Vector(this.data.map((x,i)=>{
           let temp=a.data?a.data[i]:a
           if (x instanceof Vector) return x.close(temp)
           return Math.abs(x-temp)<(1e-05+1e-08*temp)?true:false
      }),Boolean)
  }
  sort(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sort():x).slice().sort((m, n)=>{
       if (m < n) return -1
       else if (m > n) return 1
       else return 0
      })
    )}
  normal(N){
    if (N==undefined) N=[0,1]
    let [mu,sigma]=N
    return this.sub(this.mean()).div(this.std()).mul(sigma).add(mu)
  }

  _fun(axis=null,deep=0,ndim=0,method=null){
    if (axis==null){ //没指定轴
      let flatten=this.flatten()
      if (method) return method(flatten)
    }
    if (!ndim) ndim=this.ndim
    if (axis==ndim-2 && deep==ndim - 2){ //axis为纵轴
      let T=new Vector(this.value.reduce((a,b)=>
         a.map((x,i)=>x[0]!=undefined?x.concat(b[i]):[x].concat(b[i]))))
      return T.data.map(x=>x._fun(null,0,0,method))
    }
    if (axis==ndim-1 && deep==ndim - 2){ //axis为横轴
      return this.data.map(x=>x._fun(null,0,0,method))
    }
    if (axis<ndim-2 && deep == axis){ //axis小于最后两个维度
      return this.data.map((x,i)=>(x instanceof Vector)?x._fun(null,0,0,method):x)
    }
    let result = this.data.map((x,i)=>{ //递归到数据层
      if (i==0) deep++
      return (x instanceof Vector)?x._fun(axis,deep,ndim,method):x 
    })
    return this.toVector(result)
  }
  sum(axis=null,deep=0,ndim=0){
    return this._fun(axis,0,0,(flatten)=>{
       return flatten.data.reduce((a,b)=>a+b)
    })
  }
  max(axis=null,deep=0,ndim=0){
    return this._fun(axis,0,0,(flatten)=>{
       return flatten.data.reduce((a,b)=>a>b?a:b)
    })
  }
  min(axis=null,deep=0,ndim=0){
    return this._fun(axis,0,0,(flatten)=>{
       return flatten.data.reduce((a,b)=>a<b?a:b)
    })
  }
  argmax(axis=null,deep=0,ndim=0){
    return this._fun(axis,0,0,(flatten)=>{
      return flatten.data.indexOf(flatten.max())
    })
  }
  argmin(axis=null,deep=0,ndim=0){
    return this._fun(axis,0,0,(flatten)=>{
      return flatten.data.indexOf(flatten.min())
    })
  }
  mean(axis=null,deep=0,ndim=0){
    return this._fun(axis,0,0,(flatten)=>{
      return flatten.data.reduce((a,b)=>a+b)/flatten.data.length
    })
  }
  var(axis=null,deep=0,ndim=0){
    return this._fun(axis,0,0,(flatten)=>{
      let mean = flatten.mean()
      return flatten.data.map(x=>(x-mean)**2).reduce((a,b)=>a+b)/flatten.data.length
    })
  }
  std(axis=null){
    let v = this.var(axis)
    if (typeof v == "number") return Math.sqrt(v)
    return this.toVector(v).sqrt()
  }
  cov(axis=null,deep=0,ndim=0){
    return this._fun(axis,0,0,(flatten)=>{
      let mean = flatten.mean()
      return flatten.data.map(x=>(x-mean)**2).reduce((a,b)=>a+b)/(flatten.data.length-1)
    })
  }
  ptp(axis=null,deep=0,ndim=0){
    return this._fun(axis,0,0,(flatten)=>{
      return flatten.max() - flatten.min()
    })
  }
  median(axis=null,deep=0,ndim=0){
    return this._fun(axis,0,0,(flatten)=>{
      let length=parseInt(flatten.data.length/2)
      if (flatten.data.length%2!=0) return (flatten.data[length]+flatten.data[length-1])/2 
      return flatten.data[length]
    })
  }

  allclose(x){return this.close(x).all()}
  all(){return this.data.map((x,i)=>x instanceof Vector?x.all():x).reduce((a,b)=>a&&b)}
  any(){return this.data.map((x,i)=>x instanceof Vector?x.any():x).reduce((a,b)=>a||b)}
  
  dot(a){
    this.ensureCanDot(a)
    if (this.ndim==1)
      return this.data.map((x,i)=>x*(a.data?a.data[i]:a)).reduce((i,j)=>i+j)
    if (this.ndim==2)
      return this.toVector(this.data.map((x,i)=>a.T.data.map((y,j)=>x.dot(y))))
  }
  clip(m,n){
    return new Vector(this.data.map(x=>{
      if (x instanceof Vector) return x.clip(m,n)
      let a=m,b=n,c=x
      if (m==null) a=x
      if (n==null) b=x
      if (x<a) c=a
      if (x>b) c=b
      return c
    }))
  }
  slice(p){
    let data=[]
    if (!Array.isArray(p)) p=[p]
    let [s,t,o]=p
    if (s==undefined) s=0
    if (s<0) s=this.length + s
    if (t==undefined) t=(s==0)?this.length:s+1
    if (t<0) t=this.length + t
    if (o==undefined) o=1
    for (let i=s;i<t;i+=o){
      data.push(this.data[i])
    }
    return new Vector(data)
  }
  take(p){
    if (!Array.isArray(p)) p=[p]
    return new Vector(p.map(n=>{
       if (n>this.length-1) throw new Error(`${n} 超过向量边界`)
       return this.data[n]
      }))
  }
  where(){}
  nonzero(){}
  
  hstack(a){
    a=new Vector(a)
    if (this.shape.slice(0,-1).toString()!=a.shape.slice(0,-1).toString())
      throw new Error(`(${this.shape.slice(0,-1).toString()})和(${a.shape.slice(0,-1).toString()})不符合横向堆叠要确保横向形状一致的要求`)
  }
  vstack(a){
    a=new Vector(a)
    if (this.shape.slice(1).toString()!=a.shape.slice(1).toString())
      throw new Error(`(${this.shape.slice(1).toString()})和(${a.shape.slice(1).toString()})不符合纵向堆叠要确保纵向形状一致的要求`)
    return new Vector(this.value.concat(a.value),a.dtype)
  }
  hsplit(m){}
  vsplit(m){}
  split(m,axis=1){
    if (axis==1) return this.hsplit(m)
    if (axis==0) return this.vsplit(m)
    throw new Error("必须制定axis为0-纵向分割，1-横向分割,默认axis=1")
  }
  pad(){}
}
class Dim{
  constructor(){
    this.Vector = Vector
    this.Complex= Complex
    this.random = new Random(this)
    this.fft = fft
  }
  ensureVector(...a){
    let v = a.map(x=>{
      if (x instanceof Vector) return x
      if (typeof x=="number" || x instanceof Complex) return x
      return new Vector(x)
    })
    if (v.length==1) v=v[0]
    return v
  }
  ensurePoly(...a){
    let v=a.map(x=>{
      if (x instanceof Poly) return x
      return new Poly(this.ensureVector(x).flatten().value)
    })
    if (v.length==1) v=v[0]
    return v
  }
  array(a,dtype){return new Vector(a,dtype)}
  flatten(a){a=this.ensureVector(a);return a.flatten()}
  transpose(a){a=this.ensureVector(a);return a.transpose()}
  copy(a){a=this.ensureVector(a);return a.copy()} 
  save(a,file){a=this.ensureVector(a);return a.save(file)}
  load(file){return new Vector(JSON.parse(fs.readFileSync(file,'utf8')))}
  
  arange(start,end,step,dtype){
    if (typeof end=="function"){
      dtype = end
      end = null
      step = null
    }
    if (!end) {
      end = start
      start = 0
    }
    if (!step) {
      step = 1
    }
    let arr=[]
    for (let i=start;i<end;i+=step){
      arr.push(i)
    }
    return this.array(arr,dtype)
  }
  linspace(start,end,num,dtype){
    if (typeof end == "function"){
      dtype = end
      num = start
      start = 0
      end = num
    }
    let step = (end - start ) / num
    let arr=[]
    for (let i=0;i<num;i++){
      arr.push(start)
      start +=step
    }
    return this.array(arr,dtype)
  }
  mat(str,dtype){
    let data=str.split(";")
    let arr = data.map(x=>x.replace(/\s+/g,",").split(",").map(x=>{
        let d=parseFloat(x)
        if (d!=NaN) return d
        return x
      }))
    if (arr.length==1) arr=arr[0]
    return this.array(arr,dtype)
  }
  __reset(value,dtype,shape=1,...args){
    let arr=[]
    let size=shape
    if (typeof shape!="number")
      size=shape.reduce((a,b)=>a*b)
    for (let i=0;i<size;i++){
      if (typeof value=="function"){
        if (dtype==Complex){
          arr.push(new Complex(value(i,args),value(i,args)))
        }else {
          arr.push(value(i,args))
        }
      }else{
        if (dtype==Complex){
          arr.push(new Complex(value))
        }else{
          arr.push(value)
        }
      }
    }
    if (typeof shape =="number")
      return this.array(arr,dtype)
    return this.array(arr,dtype).reshape(shape)
  }
  zeros(shape=1,dtype){
    return this.__reset(0,dtype,shape)
  }
  ones(shape=1,dtype){
    return this.__reset(1,dtype,shape)
  }
  eye(number,dtype){//对角矩阵
    return this.__reset((i,args)=>{
        let n=args[0]
        return (i%n==parseInt(i/n))?1:0
      },dtype,[number,numper],number)
  }
  diag(array,dtype){//自定义对角阵
    let len=array.length
    return this.__reset((i,args)=>{
        let n=args[0].length
        return (i%n==parseInt(i/n))?args[0][i%n]:0
      },dtype,[len,len],array)
  }
  reshape(a,...d){a=this.ensureVector(a);return a.reshape(...d)}
  poly1d(a){
    let b=a
    if (Array.isArray(a) && Array.isArray(a[0])){
      let p=new Poly([])
      return p.lagrange(a)
    }
    return this.ensurePoly(a) 
  }
  polyadd(p1,p2){[p1,p2]=this.ensurePoly(p1,p2);return p1.add(p2)}
  polysub(p1,p2){[p1,p2]=this.ensurePoly(p1,p2);return p1.sub(p2)}
  polymul(p1,p2){[p1,p2]=this.ensurePoly(p1,p2);return p1.mul(p2)}
  polydiv(p1,p2){[p1,p2]=this.ensurePoly(p1,p2);return p1.div(p2)}
  polyval(p,a){p=this.ensurePoly(p);return p.val(a)}
  deriv(p){p=this.ensurePoly(p);return p.deriv()}
  integ(p){p=this.ensurePoly(p);return p.integ()}
  roots(p){p=this.ensurePoly(p);return p.roots()}
  lagrange(points){return this.poly1d(points)}
  
  sin(a){a=this.ensureVector(a);a.sin()}
  cos(a){a=this.ensureVector(a);a.cos()}
  tan(a){a=this.ensureVector(a);a.tan()}
  asin(a){a=this.ensureVector(a);a.asin()}
  acos(a){a=this.ensureVector(a);a.acos()}
  atan(a){a=this.ensureVector(a);a.atan()}
  asinh(a){a=this.ensureVector(a);a.asinh()}
  acosh(a){a=this.ensureVector(a);a.acosh()}
  atanh(a){a=this.ensureVector(a);a.atanh()}
  sinh(a){a=this.ensureVector(a);a.sinh()}
  cosh(a){a=this.ensureVector(a);a.cosh()}
  tanh(a){a=this.ensureVector(a);a.tanh()}
  log(a){a=this.ensureVector(a);a.log()}
  log2(a){a=this.ensureVector(a);a.log2()}
  log10(a){a=this.ensureVector(a);a.log10()}
  exp(a){a=this.ensureVector(a);a.exp()}
  sqrt(a){a=this.ensureVector(a);a.sqrt()}
  square(a){a=this.ensureVector(a);a.square()}
  pow(a,n){a=this.ensureVector(a);a.pow(n)}
  floor(a){a=this.ensureVector(a);a.floor()}
  ceil(a){a=this.ensureVector(a);a.ceil()}
  around(a,n){a=this.ensureVector(a);a.around(n)}
 
  add(a,b){[a,b]=this.ensureVector(a,b);return a.add(b)}
  sub(a,b){[a,b]=this.ensureVector(a,b);return a.sub(b)}
  mul(a,b){[a,b]=this.ensureVector(a,b);return a.mul(b)}
  div(a,b){[a,b]=this.ensureVector(a,b);return a.div(b)}
  power(a,b){[a,b]=this.ensureVector(a,b);return a.power(b)}
  mod(a,b){[a,b]=this.ensureVector(a,b);return a.mod(b)}
  subtract(x){return this.sub(x)}
  multiply(x){return this.mul(x)}
  divide(x){return this.div(x)}
  neg(){return this.mul(-1)}
  
  reciprocal(a){a=this.ensureVector(a);return a.reciprocal()}
  sign(a){a=this.ensureVector(a);return a.sign()}

  gt(a,b){[a,b]=this.ensureVector(a,b);return a.gt(b)}
  gte(a,b){[a,b]=this.ensureVector(a,b);return a.gte(b)}
  lt(a,b){[a,b]=this.ensureVector(a,b);return a.lt(b)}
  lte(a,b){[a,b]=this.ensureVector(a,b);return a.lte(b)}
  eq(a,b){[a,b]=this.ensureVector(a,b);return a.eq(b)}
  ne(a,b){[a,b]=this.ensureVector(a,b);return a.ne(b)}
  close(a,b){[a,b]=this.ensureVector(a,b);return a.close(b)}
  
  sort(a){a=this.ensureVector(a);return a.sort(b)}

  normal(N){
    if (N==undefined) N=[0,1]
    let [mu,sigma]=N
    return this.sub(this.mean()).div(this.std()).mul(sigma).add(mu)
  }
  
  sum(a,axis=null){a=this.ensureVector(a);return a.sum(axis)}
  mean(a,axis=null){a=this.ensureVector(a);return a.mean(axis)}
  max(a,axis=null){a=this.ensureVector(a);return a.max(axis)}
  min(a,axis=null){a=this.ensureVector(a);return a.min(axis)}
  argmax(a,axis=null){a=this.ensureVector(a);return a.argmax(axis)}
  argmin(a,axis=null){a=this.ensureVector(a);return a.argmin(axis)}

  var(a,axis=null){a=this.ensureVector(a);return a.var(axis)}
  std(a,axis=null){a=this.ensureVector(a);return a.std(axis)}
  cov(a,axis=null){a=this.ensureVector(a);return a.cov(axis)}
  ptp(a,axis=null){a=this.ensureVector(a);return a.ptp(axis)}
  median(a,axis=null){a=this.ensureVector(a);return a.madian(axis)}

  allclose(a,b){[a,b]=this.ensureVector(a,b);return a.allclose(b)}
  all(a){a=this.ensureVector(a);return a.all()}
  any(a){a=this.ensureVector(a);return a.any()}

  dot(a,b){[a,b]=this.ensureVector(a,b);return a.dot(b)}
  matmul(a,b){return this.dot(a,b)}
  
  clip(a,m,n){a=this.ensureVector(a);return a.clip(m,n)}

  slice(a,...k){
    a=this.ensureVector(a)
    if (a instanceof Vector) return a.slice(k[0])
    if (a instanceof Matrix) return a.slice(k[0],k[1])
  }
  
  take(a,p,axis){
    a=this.ensureVector(a)
    if (a instanceof Vector) return a.take(p)
    if (a instanceof Matrix) return a.take(p,axis)
  }
  where(){}
  nonzero(){}
  
  hstack(a,b){[a,b]=this.ensureVector(a,b);return a.hstack(b)}
  vstack(a,b){[a,b]=this.ensureVector(a,b);return a.vstack(b)}

  hsplit(a,m){a=this.ensureVector(a);return a.hsplit(m)}
  vsplit(a,m){a=this.ensureVector(a);return a.vsplit(m)}
  split(a,m,axis=1){a=this.ensureVector(a);return a.split(m,axis)}
  
  pad(a,s,mode,v){a=this.ensureVector(a);return a.pad(s,mode,v)}

  fftConv(a,b){
    if (!Array.isArray(a) || !Array.isArray(b)) throw new Error(`a、b参数必须都是数组`)
    let n = a.length + b.length -1 
    let N = 2**(parseInt(Math.log2(n))+1)
    let numa=N-a.length
    let numb=N-b.length
    for(let i=0;i<numa;i++) a.unshift(0)
    for(let i=0;i<numb;i++) b.unshift(0)
    let A=this.array(this.fft.fft(a))
    let B=this.array(this.fft.fft(b))
    let C=A.mul(B)
    return this.fft.ifft(C.data)
  }
}
exports.dim = new Dim()
