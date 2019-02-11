fs=require('fs')

class V{
  constructor(object,dtype=Float32Array){
    if (object instanceof V){
      this.dtype = object.dtype
      this.data = object.data
    }else{
      //自动判定复数
      if (Array.isArray(object) && (object[0] instanceof Complex)) dtype=Complex
      this.dtype = dtype
      this.data = this.ensureArray(object)
    }
    this.shape=this.getShape()
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
    return this.data.map((x,i)=>(x instanceof V)?x.value:x)
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
    return new V(result,dtype).reshape(this.shape)
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
  
  ensureArray(data){
    if (this.dtype == Complex){
      if (Array.isArray(data)){
        return data.map(x=>{
          if (x instanceof V) {console.log("is V");return}
          if (x instanceof Complex) return x
          return new Complex(x)})
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
    this.data.map((x,i)=>(x instanceof V)?x.flatten(item):item.push(x))
    //console.log(item)
    return new V(item)
  }
  copy(){return dim.ensureVector(this.value,this.dtype)}

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
    return dim.ensureVector(t,this.dtype)
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
      if (ndim==2) return dim.ensureVector(T)
      return T
    }
    let result = this.data.map((x,i)=>{
      if (i==0) deep++
      return (x instanceof V)?x.transpose(deep,ndim):x 
    })
    return dim.ensureVector(result)
  }

  sin(){return new V(this.data.map((x,i)=>(x instanceof V)?x.sin():Math.sin(x)))}
  cos(){return new V(this.data.map((x,i)=>(x instanceof V)?x.cos():Math.cos(x)))}
  tan(){return new V(this.data.map((x,i)=>(x instanceof V)?x.tan():Math.tan(x)))}
  asin(){return new V(this.data.map((x,i)=>(x instanceof V)?x.asin():Math.asin(x)))}
  acos(){return new V(this.data.map((x,i)=>(x instanceof V)?x.acos():Math.acos(x)))}
  atan(){return new V(this.data.map((x,i)=>(x instanceof V)?x.atan():Math.atan(x)))}
  asinh(){return new V(this.data.map((x,i)=>(x instanceof V)?x.asinh():Math.asinh(x)))}
  acosh(){return new V(this.data.map((x,i)=>(x instanceof V)?x.acosh():Math.acosh(x)))}
  atanh(){return new V(this.data.map((x,i)=>(x instanceof V)?x.atanh():Math.atanh(x)))}
  sinh(){return new V(this.data.map((x,i)=>(x instanceof V)?x.sinh():Math.sinh(x)))}
  cosh(){return new V(this.data.map((x,i)=>(x instanceof V)?x.cosh():Math.cosh(x)))}
  tanh(){return new V(this.data.map((x,i)=>(x instanceof V)?x.tanh():Math.tanh(x)))}
  log(){return new V(this.data.map((x,i)=>(x instanceof V)?x.log():Math.log(x)))}
  log2(){return new V(this.data.map((x,i)=>(x instanceof V)?x.log2():Math.log2(x)))}
  log10(){return new V(this.data.map((x,i)=>(x instanceof V)?x.log10():Math.log10(x)))}
  exp(){return new V(this.data.map((x,i)=>(x instanceof V)?x.exp():Math.exp(x)))}
  sqrt(){return new V(this.data.map((x,i)=>(x instanceof V)?x.sqrt():Math.sqrt(x)))}
  square(){return new V(this.data.map((x,i)=>(x instanceof V)?x.square():Math.pow(x,2)))}
  pow(n){return new V(this.data.map((x,i)=>(x instanceof V)?x.pow(n):Math.pow(x,n)))}
  floor(){return new V(this.data.map((x,i)=>(x instanceof V)?x.floor():Math.floor(x)))}
  ceil(){return new V(this.data.map((x,i)=>(x instanceof V)?x.ceil():Math.ceil(x)))}
  around(n){return new V(this.data.map((x,i)=>{
      if (x instanceof V) return x.around(n)
      let a=10**n
      return Math.round(x*a)/a
    }))
  }

  add(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.add(a.data?a.data[i]:a):x.add(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>(x instanceof V)?x.add(a.data?a.data[i]:a):x+(a.data?a.data[i]:a)))
  }
  sub(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.sub(a.data?a.data[i]:a):x.sub(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>(x instanceof V)?x.sub(a.data?a.data[i]:a):x-(a.data?a.data[i]:a)))
  }
  mul(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.mul(a.data?a.data[i]:a):x.mul(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>(x instanceof V)?x.mul(a.data?a.data[i]:a):x*(a.data?a.data[i]:a)))
  }
  div(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.div(a.data?a.data[i]:a):x.div(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>(x instanceof V)?x.div(a.data?a.data[i]:a):x/(a.data?a.data[i]:a)))
  }
  power(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.power(a.data?a.data[i]:a):x.power(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>(x instanceof V)?x.power(a.data?a.data[i]:a):x**(a.data?a.data[i]:a)))
  }
  mod(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.mod(a.data?a.data[i]:a):x.mod(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>(x instanceof V)?x.mod(a.data?a.data[i]:a):x%(a.data?a.data[i]:a)))
  }
  subtract(x){return this.sub(x)}
  multiply(x){return this.mul(x)}
  divide(x){return this.div(x)}
  neg(){return this.mul(-1)}
  
  reciprocal(){return new V(this.data.map((x,i)=>(x instanceof V)?x.reciprocal():1/x))}
  sign(){return new V(this.data.map((x,i)=>(x instanceof V)?x.sign():Math.sign(x)))}

  gt(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.gt(a.data?a.data[i]:a):x.gt(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>(x instanceof V)?x.gt(a.data?a.data[i]:a):x>(a.data?a.data[i]:a)?true:false),Boolean)
  }
  gte(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.gte(a.data?a.data[i]:a):x.gte(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>(x instanceof V)?x.gte(a.data?a.data[i]:a):x>=(a.data?a.data[i]:a)?true:false),Boolean)
  }
  lt(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.lt(a.data?a.data[i]:a):x.lt(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>(x instanceof V)?x.lt(a.data?a.data[i]:a):x<(a.data?a.data[i]:a)?true:false),Boolean)
  }
  lte(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.lte(a.data?a.data[i]:a):x.lte(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>(x instanceof V)?x.lte(a.data?a.data[i]:a):x<=(a.data?a.data[i]:a)?true:false),Boolean)
  }
  eq(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.eq(a.data?a.data[i]:a):x.eq(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>(x instanceof V)?x.eq(a.data?a.data[i]:a):x=(a.data?a.data[i]:a)?true:false),Boolean)
  }
  ne(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.ne(a.data?a.data[i]:a):x.ne(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>(x instanceof V)?x.ne(a.data?a.data[i]:a):x!=(a.data?a.data[i]:a)?true:false),Boolean)
  }
  close(a){
    this.ensureSameShape(a)
    if (this.dtype == Complex)
      return new V(this.data.map((x,i)=>(x instanceof V)?x.close(a.data?a.data[i]:a):x.close(a.data?a.data[i]:a)))
    return new V(this.data.map((x,i)=>{
           let temp=a.data?a.data[i]:a
           if (x instanceof V) return x.close(temp)
           return Math.abs(x-temp)<(1e-05+1e-08*temp)?true:false
      }),Boolean)
  }
  sort(){return new V(this.data.map((x,i)=>(x instanceof V)?x.sort():[...x].sort()))}
  normal(N){
    if (N==undefined) N=[0,1]
    let [mu,sigma]=N
    return this.sub(this.mean()).div(this.std()).mul(sigma).add(mu)
  }

  sum(axis=null,deep=0,ndim=0){
    if (!ndim) ndim=this.ndim
    //console.log('deep=',deep,',axis=',axis,',length=',this.data.length,this.value)
    if (axis==ndim-2 && deep==ndim - 2){
      return this.value.reduce((a,b)=>a.map((m,k)=>m+b[k]))
    }
    if (axis==ndim-1 && deep==ndim - 2){
      return this.value.map(a=>a.reduce((m,n)=>m+n))
    }
    if (axis<ndim-2 && deep == axis){
      return this.data.map((x,i)=>(x instanceof V)?x.sum():x)
    }
    let result = this.data.map((x,i)=>{
      if (i==0) deep++
      return (x instanceof V)?x.sum(axis,deep,ndim):x 
    })
    if (axis==null)
      return result.reduce((a,b)=>a+b)
    return dim.ensureVector(result)
  }
  max(axis=null,deep=0,ndim=0){
    if (!ndim) ndim=this.ndim
    if (axis==ndim-2 && deep==ndim - 2){
      return this.value.reduce((a,b)=>a.map((m,k)=>m>b[k]?m:b[k]))
    }
    if (axis==ndim-1 && deep==ndim - 2){
      return this.value.map(a=>a.reduce((m,n)=>m>n?m:n))
    }
    if (axis<ndim-2 && deep == axis){
      return this.data.map((x,i)=>(x instanceof V)?x.max():x)
    }
    let result = this.data.map((x,i)=>{
      if (i==0) deep++
      return (x instanceof V)?x.max(axis,deep,ndim):x 
    })
    if (axis==null)
      return result.reduce((a,b)=>a>b?a:b)
    return dim.ensureVector(result)
  }
  min(axis=null,deep=0,ndim=0){
    if (!ndim) ndim=this.ndim
    if (axis==ndim-2 && deep==ndim - 2){
      return this.value.reduce((a,b)=>a.map((m,k)=>m<b[k]?m:b[k]))
    }
    if (axis==ndim-1 && deep==ndim - 2){
      return this.value.map(a=>a.reduce((m,n)=>m<n?m:n))
    }
    if (axis<ndim-2 && deep == axis){
      return this.data.map((x,i)=>(x instanceof V)?x.min():x)
    }
    let result = this.data.map((x,i)=>{
      if (i==0) deep++
      return (x instanceof V)?x.min(axis,deep,ndim):x 
    })
    if (axis==null)
      return result.reduce((a,b)=>a<b?a:b)
    return dim.ensureVector(result)
  }
  argmin(axis=null,deep=0,ndim=0){
    if (!ndim) ndim=this.ndim
    if (axis==ndim-2 && deep==ndim - 2){
      return this.value.reduce((a,b)=>a.map((m,k)=>m<b[k]?m:b[k]))
    }
    if (axis==ndim-1 && deep==ndim - 2){
      return this.value.map(a=>a.reduce((m,n)=>m<n?m:n))
    }
    if (axis<ndim-2 && deep == axis){
      return this.data.map((x,i)=>(x instanceof V)?x.argmin():x)
    }
    let result = this.data.map((x,i)=>{
      if (i==0) deep++
      return (x instanceof V)?x.argmin(axis,deep,ndim):x 
    })
    if (axis==null)
      return result.reduce((a,b)=>a<b?a:b)
    return dim.ensureVector(result)
  }
  mean(axis=null,deep=0,ndim=0){
    if (!ndim) ndim=this.ndim
    if (axis==ndim-2 && deep==ndim - 2){
      return this.value.reduce((a,b)=>a.map((m,k)=>m+b[k])).map(a=>a/this.value.length)
    }
    if (axis==ndim-1 && deep==ndim - 2){
      return this.value.map(a=>a.reduce((m,n)=>m+n)/a.length)
    }
    if (axis<ndim-2 && deep == axis){
      return this.data.map((x,i)=>(x instanceof V)?x.mean():x)
    }
    let result = this.data.map((x,i)=>{
      if (i==0) deep++
      return (x instanceof V)?x.mean(axis,deep,ndim):x 
    })
    if (axis==null)
      return result.reduce((a,b)=>a+b)/result.length
    return dim.ensureVector(result)
  }
  var(axis=null,deep=0,ndim=0){
    if (!ndim) ndim=this.ndim
    if (axis==ndim-2 && deep==ndim - 2){
      let T = this.value.reduce((a,b)=>
         a.map((x,i)=>x[0]!=undefined?x.concat(b[i]):[x].concat(b[i])))
      console.log(T)   
    }
    if (axis==ndim-1 && deep==ndim - 2){
      let mean = this.mean()
      return this.value.map(a=>a.map(x=>(x-mean)**2)).reduce((m,n)=>m+n)/this.value.length
    }
    if (axis<ndim-2 && deep == axis){
      return this.data.map((x,i)=>(x instanceof V)?x.var():x)
    }
    let result = this.data.map((x,i)=>{
      if (i==0) deep++
      return (x instanceof V)?x.var(axis,deep,ndim):x 
    })
    if (axis==null){
      let mean = result.reduce((a,b)=>a+b)/result.length
      return result.map(x=>(x-mean)**2).reduce((a,b)=>a+b)/result.length
    }
    return dim.ensureVector(result)
  }
  std(){return this.var().sqrt()}
  var1(){
    let mean = this.mean()
    return this.data.map(x=>(x-mean)**2).reduce((x,y)=>x+y)/this.data.length
  }

/*
  argmin(){return this.data.indexOf(this.min())}
  argmax(){return this.data.indexOf(this.max())}
  std(){return Math.sqrt(this.var())}
  var(){
    let mean = this.mean()
    return this.data.map(x=>(x-mean)**2).reduce((x,y)=>x+y)/this.data.length
  }
  cov(){
    let mean = this.mean()
    return this.data.map(x=>(x-mean)**2).reduce((x,y)=>x+y)/(this.data.length-1)
  }
  corrcoef(){
    return 1
  }
  ptp(){return this.max()-this.min()}
  median(){
    let length=parseInt(this.data.length/2)
    if (this.data.length%2!=0) return (this.data[length]+this.data[length-1])/2 
    return this.data[length]
  }

*/
  allclose(x){return this.close(x).all()}
  all(){return this.data.map((x,i)=>x instanceof V?x.all():x).reduce((a,b)=>a&&b)}
  any(){return this.data.map((x,i)=>x instanceof V?x.any():x).reduce((a,b)=>a||b)}
  
  dot(a){
    this.ensureCanDot(a)
    if (this.ndim==1)
      return this.data.map((x,i)=>x*(a.data?a.data[i]:a)).reduce((i,j)=>i+j)
    if (this.ndim==2)
      return dim.ensureVector(this.data.map((x,i)=>a.T.data.map((y,j)=>x.dot(y))))
  }
  clip(m,n){
    return new V(this.data.map(x=>{
      if (x instanceof V) return x.clip(m,n)
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
  save(file){
    return fs.writeFileSync(file,this.data)
  }
}
class Dim{
  constructor(){
    this.Vector = V
    this.Complex= Complex
    this.random = new Random(this)
  }
  ensureVector(a,dtype){
    if (a instanceof this.Vector) return a
    if (Array.isArray(a)) {
      if (Array.isArray(a[0])){
        return new this.Vector(a.map(x=>this.ensureVector(x)),dtype)
      }else {
        return new this.Vector(a,dtype)
      }
    }
    if (typeof a=="number") {
      let r=[]
      for (let i=0;i<a;i++) r.push(0)
      return new this.Vector(r,dtype)
    }
  }
  array(a,dtype){return this.ensureVector(a,dtype)}
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

}
exports.dim = new Dim()

exports.Complex = Complex
exports.Poly    = Poly
