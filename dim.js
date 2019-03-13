fs=require('fs')
Random = require('./random.js').Random
Complex = require('./complex.js').Complex
Poly = require('./poly.js').Poly
NN = require('./nn.js').NN
Optimizer = require('./nn.js').Optimizer
fft = require('./fft.js').fft
autograd = require('./autograd.js')

class Vector{
  constructor(object,dtype=Float32Array){
    if (object instanceof Vector){
      return object
    }
    if (Array.isArray(object)) {
      if (Array.isArray(object[0])){
        return new Vector(object.map(x=>new Vector(x,dtype)),dtype)
      }
    }
    //自动判定复数
    if (Array.isArray(object) && (object[0] instanceof Complex)) dtype=Complex
    this.dtype = dtype
    this.data = this.ensureArray(object)

    this.shape=this.getShape()
    
    //grad
    this.requiresGrad=false
    this.grad=null
    this.gradFn=null //new dim.nn.grad.Constant(this)
    return this
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
  get T(){
    let rst = this.transpose()
    return this.setGradFn(rst,"T")
  }
  
  get value(){
    return this.data.map((x,i)=>(x instanceof Vector)?x.value:x)
  }
  replace(a){
    this.ensureSameShape(a)
    this.data.map((x,i)=>{
      if (x instanceof Vector) return x.replace(a.data[i])
      this.data[i]=a.data[i]
    })
    return this
  }
  fill(n){
    this.data.map((x,i)=>{
      if (x instanceof Vector) return x.fill(n)
      this.data[i]=n
    })
    return this
  }
  zero_(){return this.fill(0)}
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
    if (typeof a == "number") return a
    return new Vector(a,dtype)
  }
  broadcast(n){
    if (!(typeof n=="number")) throw new Error("参数必须是数字")
    let a=[]
    for (let i=0;i<this.size;i++){
      a.push(n)
    }
    if (this.ndim==1) return new Vector(a)
    return new Vector(a).reshape(this.shape)
  }
  get isUnit(){return (this.shape.toString()=='1')} 
  isNumber(a){return (typeof a=="number")}
  isComplex(a){return (a instanceof Complex)}
  
  ensureSameShape(a){
    if (this.isNumber(a) || a.isUnit || this.isComplex(a)) return a
    if (this.shape.toString()!=a.shape.toString()) {
      console.log("this:",this.value)
      console.log("a:",a.value)
      throw new Error(`形状(${this.shape})与形状(${a.shape})不一致`)
    }
    return a
  }
  ensure1D(){
    if (this.ndim!=1) throw new Error(`要求是1维矩阵,但是参数是${this.ndim}维`)
    return true
  }
  ensure2D(){return this.ensureMatrix()}
  ensureMatrix(){
    if (this.ndim!=2) throw new Error(`要求是2维矩阵,但是参数是${this.ndim}维`)
    return true
  }
  ensure3D(){
    if (this.ndim!=3) throw new Error(`要求是3维矩阵,但是参数是${this.ndim}维`)
    return true
  }
  ensureSquareMatrix(){
    if (this.ndim!=2 || this.shape[0]!=this.shape[1]) 
      throw new Error(`要求是方阵，但是参数形状是(${this.shape}),维度是(${this.ndim})`)
  }
  ensureCanOnehot(){
    if (this.ndim==1) return this.reshape(this.size,1)
    if (this.ndim==2 && this.shape[1]==1) return this
    console.log("ensureCanOnehot",this.value)
    throw new Error(`对象要求是一维向量，或是n*1矩阵`)
  }
  ensureCanDot(a){
    if (typeof a =="number") return
    if (this.ndim!=2 && this.ndim!=1) throw new Error(`参数是(${a.ndim})维,仅支持一、二维`)
    if (this.ndim==1 && a.ndim!=1) throw new Error(`要求是一维向量，但是参数是(${a.ndim})维`)
    if (this.ndim==2 && a.ndim!=2) throw new Error(`要求是二维向量，但是参数是(${a.ndim})维`)
    if (this.ndim==2 && this.shape[1]!=a.shape[0]) {
      console.log("this:",this.value)
      console.log("a:",a.value)
      throw new Error(`(${this.shape})和(${a.shape})形状不符合要求`)
    }
  }
  rot180(){
    if (this.ndim==1){
      return new Vector(this.value.reverse())
    }else if (this.ndim==2){
      let h=this.shape[0]
      let w=this.shape[1]
      let value=this.value
      let a=[]
      for (let i=0;i<h;i++){
        a[i]=[]
        for(let j=0;j<w;j++){
          a[i][j] = value[h-1-i][w-1-j]
        }
      }
      return new Vector(a)    
    }else{
      throw new Error(`不支持高于二维操作`)
    }
  }
  flatten(item){
    if (!item) item=[]
    this.data.map((x,i)=>{
      x=dim.toVector(x);
      return (x instanceof Vector)?x.flatten(item).data:item.push(x)
    })
    return dim.toVector(item)
  }

  copy(){return new Vector(this.value,this.dtype)}
  save(file){return fs.writeFileSync(file,JSON.stringify(this.value))}
  
  print(lr=3,lc=3,first){
    if (first==undefined) {
      console.log(`<==== shape:${this.shape.join('*')} ====>`)
      let left=''
      for (let i=0;i<this.ndim-3;i++) left+='['
      console.log(left)
    }
    if ((this.ndim-3)>0){
      let row=this.shape[0]
      this.data.map((x,i)=>{
        if (i<lr || i>row-lr-1) x.print(lr,lc,false)
        if (i==lr) console.log("... ... ...")
      })
    }else{
      console.log(this._print(lr,lc))
    }
    if (first==undefined){
      let right=''
      for (let i=0;i<this.ndim-3;i++) right+=']'
      console.log(right)
    }
    return 'ok'
  }
  _print(lr=3,lc=3){
    let ndim=this.ndim
    let col = this.shape[this.shape.length - 1]
    let row = this.shape[0]
      return this.data.map((x,i)=>{
        if (x instanceof Vector) {
          if (i<lr || i>row-lr-1) return x._print(lr,lc)
          if (i==lr) {
            return ["... ... ... ... ... ... ..."]
          }else { 
            return ['omit']
          }
        }else{
          return (i<lc || i>col-lc-1)?Math.round(x*10000)/10000:(i==lc)?"... ...":"omit"
        }
      }).filter(y=>y!='omit')
    
  }
  reshape(...d){
    if (Array.isArray(d[0])) d=d[0]
    let a=this.flatten().data
    let t,p=[],plen=0
    if (d.length==1) return this.toVector(a,this.dtype) //如果一维直接返回
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
  swapaxes(m,n){
    if (m==n) return this
    if (m%1!=0 || n%1!=0 || m<0 || n<0 || m>this.ndim-1 || n>this.ndim-1) throw new Error('轴参数错误')
    let A=this.value
    let [...shape]= this.shape
    let a=shape[m]
    shape[m]=shape[n]
    shape[n]=a
    let B=dim.zeros(shape).value
    let indices = new Array(this.ndim)
    return new Vector(this._swapaxes(A,B,0,indices,m,n))
    /*
    for (let i=0;i<shape[0];i++){
      for (let j=0;j<shape[1];j++){
        for (let k=0;k<shape[2];k++){
          dataB[i][j][k] = dataA[i][k][j]
        }
      }
    }
    */
  }
  _swapaxes(A,B,deep,indices,m,n){
    return B.map((x,i)=>{
      indices[deep]=i
      if (Array.isArray(x)){
        if (i==0) indices[deep+1]=i
        return this._swapaxes(A,x,deep+1,indices,m,n)
      }
      let index = [...indices]
      index[index.length - 1]=i
      if (index[m]!=index[n]){
        let temp=index[n]
        index[n]=index[m]
        index[m]=temp
      }
      let item=A
      for (let k=0;k<index.length;k++){
        item=item[index[k]]
      }
      return item
    })
  }
  squeeze(){
    //仅对维数为1的Vector进行降维
    if (this.ndim==1) return this
    let v = this.data.map(x=>{
      if (!dim.isArray(x)) return x
      if (x.ndim!=1 && x.shape[0]==1) x=x.data[0]
      if (x.ndim!=1 && x instanceof Vector) return x.squeeze()
      return x
    })
    if (v.length==1) v=v[0]
    return new Vector(v)
  }
  flatIdx(indices){
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
  dimIdx(indices){
    return [1,1]
  }
  transpose(deep=0,ndim){
    /*if (!ases) ases=[...this.shape].reverse()
    let set = new Set(axes)
    if (set.size!=axes.length) throw new Error(`axes有重复的参数`)
    */
    if (!ndim) ndim=this.ndim
    if (deep==ndim - 2){
      let T = this.value.reduce((a,b)=>a.map((x,i)=>x[0]!=undefined?x.concat(b[i]):[x].concat(b[i])))
      if (!Array.isArray(T[0])) T=T.map(x=>[x])   
      if (ndim==2) return this.toVector(T)
      return T
    }
    let result = this.data.map((x,i)=>{
      if (i==0) deep++
      return (x instanceof Vector)?x.transpose(deep,ndim):x 
    })
    return this.toVector(result)
  }
  setGradFn(t,a,opStr){
    if (opStr==undefined) {
      opStr=a
      a=null
    }
    if (this.requiresGrad || (a && a.requiresGrad)){
      t.requiresGrad=true
      let leftFn,rightFn
      leftFn=(this.gradFn)?this.gradFn:new autograd.Constant(this)
      rightFn=(a!=null)?((a.gradFn)?a.gradFn:new autograd.Constant(a)):null
      t.gradFn=autograd.Operate.wrapper(leftFn,rightFn,opStr)
    }
    return t   
  }
  radians(){
    let rst = new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.randians():x/180*Math.PI))
    return this.setGradFn(rst,"randians")
  }
  sin(){
    let rst = new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sin():Math.sin(x)))
    return this.setGradFn(rst,"sin")
  }
  sin_(){
    return this.replace(this.sin())
  }
  cos(){
    let rst = new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.cos():Math.cos(x)))
    return this.setGradFn(rst,"cos")
  }
  tan(){
    let rst = new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.tan():Math.tan(x)))
    return this.setGradFn(rst,"tan")
  }
  asin(){
    let rst = new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.asin():Math.asin(x)))
    return this.setGradFn(rst,"asin")
  }
  acos(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.acos():Math.acos(x)))
    return this.setGradFn(rst,"acos")
  }
  atan(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.atan():Math.atan(x)))
    return this.setGradFn(rst,"atan")
  }
  asinh(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.asinh():Math.asinh(x)))
    return this.setGradFn(rst,"asinh")
  }
  acosh(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.acosh():Math.acosh(x)))
    return this.setGradFn(rst,"acosh")
  }
  atanh(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.atanh():Math.atanh(x)))
    return this.setGradFn(rst,"atanh")
  }
  sinh(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sinh():Math.sinh(x)))
    return this.setGradFn(rst,"sinh")
  }
  cosh(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.cosh():Math.cosh(x)))
    return this.setGradFn(rst,"cosh")
  }
  tanh(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.tanh():Math.tanh(x)))
    return this.setGradFn(rst,"tanh")
  }
  log(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.log():Math.log(x)))
    return this.setGradFn(rst,"log")
  }
  log2(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.log2():Math.log2(x)))
    return this.setGradFn(rst,"log2")
  }
  log10(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.log10():Math.log10(x)))
    return this.setGradFn(rst,"log10")
  }
  exp(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.exp():Math.exp(x)))
    return this.setGradFn(rst,"exp")
  }
  sqrt(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sqrt():Math.sqrt(x)))
    return this.setGradFn(rst,"sqrt")
  }
  square(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.square():Math.pow(x,2)))
    return this.setGradFn(rst,"square")
  }
  pow(n){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.pow(n):Math.pow(x,n)))
    return this.setGradFn(rst,n,"pow")
  }
  floor(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.floor():Math.floor(x)))
    return this.setGradFn(rst,"floor")
  }
  ceil(){
    let rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.ceil():Math.ceil(x)))
    return this.setGradFn(rst,"ceil")
  }
  around(n){return new Vector(this.data.map((x,i)=>{
      if (x instanceof Vector) return x.around(n)
      let a=10**n
      return Math.round(x*a)/a
    }))
  }
  abs(){
    let rst=new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.abs():Math.abs(x)))
    return this.setGradFn(rst,"abs")
  }
  add(a){
    a=this.ensureSameShape(a)
    let rst
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex){
      rst = new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.add(n==undefined?a.data[i]:n):x.add(n==undefined?a.data[i]:n)))
    }else{
      rst =new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.add(n==undefined?a.data[i]:n):x+(n==undefined?a.data[i]:n)))
    }
    return this.setGradFn(rst,a,"add")
  }
  add_(a){return this.replace(this.add(a))}
  sub(a){
    a=this.ensureSameShape(a)
    let rst
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex){
      rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sub(n==undefined?a.data[i]:n):x.sub(n==undefined?a.data[i]:n)))
    }else{
      rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sub(n==undefined?a.data[i]:n):x-(n==undefined?a.data[i]:n)))
    }
    return this.setGradFn(rst,a,"sub")
  }
  sub_(a){return this.replace(this.sub(a))}
  mul(a){
    a=this.ensureSameShape(a)
    let rst
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex){
      rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.mul(n==undefined?a.data[i]:n):x.mul(n==undefined?a.data[i]:n)))
    }else{
      rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.mul(n==undefined?a.data[i]:n):x*(n==undefined?a.data[i]:n)))
    }
    return this.setGradFn(rst,a,"mul")
  }
  div(a){
    a=this.ensureSameShape(a)
    let rst
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex){
      rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.div(n==undefined?a.data[i]:n):x.div(n==undefined?a.data[i]:n)))
    }else{
      rst= new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.div(n==undefined?a.data[i]:n):x/(n==undefined?a.data[i]:n)))
    }
    return this.setGradFn(rst,a,"div")
  }
  power(a){
    a=this.ensureSameShape(a)
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.power(n==undefined?a.data[i]:n):x.power(n==undefined?a.data[i]:n)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.power(n==undefined?a.data[i]:n):x**(n==undefined?a.data[i]:n)))
  }
  mod(a){
    a=this.ensureSameShape(a)
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.mod(n==undefined?a.data[i]:n):x.mod(n==undefined?a.data[i]:n)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.mod(n==undefined?a.data[i]:n):x%(n==undefined?a.data[i]:n)))
  }
  subtract(x){return this.sub(x)}
  multiply(x){return this.mul(x)}
  divide(x){return this.div(x)}
  neg(){return this.mul(-1)}
  
  reciprocal(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.reciprocal():1/x))}
  sign(){return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.sign():Math.sign(x)))}

  gt(a){
    a=this.ensureSameShape(a)
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.gt(n==undefined?a.data[i]:n):x.gt(n==undefined?a.data[i]:n)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.gt(n==undefined?a.data[i]:n):x>(n==undefined?a.data[i]:n)?true:false),Boolean)
  }
  gte(a){
    a=this.ensureSameShape(a)
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.gte(n==undefined?a.data[i]:n):x.gte(n==undefined?a.data[i]:n)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.gte(n==undefined?a.data[i]:n):x>=(n==undefined?a.data[i]:n)?true:false),Boolean)
  }
  lt(a){
    a=this.ensureSameShape(a)
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.lt(n==undefined?a.data[i]:n):x.lt(n==undefined?a.data[i]:n)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.lt(n==undefined?a.data[i]:n):x<(n==undefined?a.data[i]:n)?true:false),Boolean)
  }
  lte(a){
    a=this.ensureSameShape(a)
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.lte(n==undefined?a.data[i]:n):x.lte(n==undefined?a.data[i]:n)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.lte(n==undefined?a.data[i]:n):x<=(n==undefined?a.data[i]:n)?true:false),Boolean)
  }
  eq(a){
    a=this.ensureSameShape(a)
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.eq(n==undefined?a.data[i]:n):x.eq(n==undefined?a.data[i]:n)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.eq(n==undefined?a.data[i]:n):x=(n==undefined?a.data[i]:n)?true:false),Boolean)
  }
  ne(a){
    a=this.ensureSameShape(a)
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.ne(n==undefined?a.data[i]:n):x.ne(n==undefined?a.data[i]:n)))
    return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.ne(n==undefined?a.data[i]:n):x!=(n==undefined?a.data[i]:n)?true:false),Boolean)
  }
  close(a){
    a=this.ensureSameShape(a)
    let n=undefined 
    if (a.isUnit) n=a.data[0]
    else if (this.isNumber(a)) n=a
    if (this.dtype == Complex)
      return new Vector(this.data.map((x,i)=>(x instanceof Vector)?x.close(n==undefined?a.data[i]:n):x.close(n==undefined?a.data[i]:n)))
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
  _fun(axis=null,deep=0,ndim=0,method=null){
    if (axis==null){ //没指定轴
      let flatten=this.flatten()
      if (method) return method(flatten)
    }
    if (!ndim) ndim=this.ndim
    if (axis==ndim-2 && deep==ndim - 2){ //axis为纵轴
      let T=new Vector(this.value.reduce((a,b)=>
         a.map((x,i)=>x[0]!=undefined?x.concat(b[i]):[x].concat(b[i]))))
      let rst = T.data.map(x=>x._fun(null,0,0,method))
      return dim.ensureVector(rst).T
    }
    if (axis==ndim-1 && deep==ndim - 2){ //axis为横轴
      return this.data.map(x=>x._fun(null,0,0,method))
    }
    if (axis<ndim-2 && deep == axis){ //axis小于最后两个维度
      return this.data.map((x,i)=>(deep<ndim-1)?x._fun(null,0,0,method):x)
    }
    let result = this.data.map((x,i)=>{ //递归到数据层
      if (i==0) deep++
      return (deep<ndim-1)?x._fun(axis,deep,ndim,method):x 
    })
    return this.toVector(result)
  }
  softmax(axis=1){
    let rst=  this.toVector(this._fun(axis,0,0,(flatten)=>{
      let exp = dim.exp(flatten.data)
      let sum = exp.sum()
      return flatten.data.map((a,i)=>Math.exp(a)/sum)
    }))
    if (this.requiresGrad && axis==1){
      rst.requiresGrad=true
      rst.gradFn=grad.Operate.wrapper(this.gradFn,null,"softmax")
    }
    return rst
  }
  sum(axis=null){
    let rst=  this.toVector(this._fun(axis,0,0,(flatten)=>{
       return flatten.data.reduce((a,b)=>a+b)
    }))
    if (axis==null && this.requiresGrad){
      rst = new Vector([rst])
      rst.requiresGrad=true
      rst.gradFn=grad.Operate.wrapper(this.gradFn,null,"sum")
    }
    return rst
  }
  max(axis=null){
    let rst = this.toVector(this._fun(axis,0,0,(flatten)=>{
       return flatten.data.reduce((a,b)=>a>b?a:b)
    }))
    if (axis==null && this.requiresGrad){
      rst = new Vector([rst])
      rst.requiresGrad=true
      rst.gradFn=grad.Operate.wrapper(this.gradFn,null,"max")
    }
    return rst
  }
  min(axis=null){
    let rst = this.toVector(this._fun(axis,0,0,(flatten)=>{
       return flatten.data.reduce((a,b)=>a<b?a:b)
    }))
    if (axis==null && this.requiresGrad){
      rst = new Vector([rst])
      rst.requiresGrad=true
      rst.gradFn=grad.Operate.wrapper(this.gradFn,null,"min")
    }
    return rst
  }
  argmax(axis=null){
    return this.toVector(this._fun(axis,0,0,(flatten)=>{
      return flatten.data.indexOf(flatten.max())
    }))
  }
  argmin(axis=null){
    return this.toVector(this._fun(axis,0,0,(flatten)=>{
      return flatten.data.indexOf(flatten.min())
    }))
  }
  mean(axis=null){
    let rst = this.toVector(this._fun(axis,0,0,(flatten)=>{
      return flatten.data.reduce((a,b)=>a+b)/flatten.data.length
    }))
    if (axis==null && this.requiresGrad){
      rst = new Vector([rst])
      rst.requiresGrad=true
      rst.gradFn=grad.Operate.wrapper(this.gradFn,null,"mean")
    }
    return rst
  }
  var(axis=null){
    return this.toVector(this._fun(axis,0,0,(flatten)=>{
      let mean = flatten.mean()
      return flatten.data.map(x=>(x-mean)**2).reduce((a,b)=>a+b)/flatten.data.length
    }))
  }
  std(axis=null){
    let v = this.var(axis)
    if (typeof v == "number") return Math.sqrt(v)
    return this.toVector(v).sqrt()
  }
  cov(axis=null){
    return this.toVector(this._fun(axis,0,0,(flatten)=>{
      let mean = flatten.mean()
      return flatten.data.map(x=>(x-mean)**2).reduce((a,b)=>a+b)/(flatten.data.length-1)
    }))
  }
  ptp(axis=null){
    return this.toVector(this._fun(axis,0,0,(flatten)=>{
      return flatten.max() - flatten.min()
    }))
  }
  median(axis=null){
    return this.toVector(this._fun(axis,0,0,(flatten)=>{
      let length=parseInt(flatten.data.length/2)
      if (flatten.data.length%2!=0) return (flatten.data[length]+flatten.data[length-1])/2 
      return flatten.data[length]
    }))
  }
  normal(axis=null,N){
    if (N==undefined) N=[0,1]
    let [mu,sigma]=N
    return this.toVector(this._fun(axis,0,0,(flatten)=>{
      return flatten.sub(flatten.mean()).div(flatten.std()).mul(sigma).add(mu)
    }))
  }
  minmaxNormal(axis=null){
    return this.toVector(this._fun(axis,0,0,(flatten)=>{
      let min = flatten.min()
      let max = flatten.max()
      return flatten.data.map(x=>{
        let rst=(x-min)/(max-min)
        return rst
      })
    }))
  }

  allclose(x){return this.close(x).all()}
  all(){return this.data.map((x,i)=>x instanceof Vector?x.all():x).reduce((a,b)=>a&&b)}
  any(){return this.data.map((x,i)=>x instanceof Vector?x.any():x).reduce((a,b)=>a||b)}
  
  dot(a){
    this.ensureCanDot(a)
    let rst
    if (this.ndim==1){
      rst = this.data.map((x,i)=>x*(a.data?a.data[i]:a)).reduce((i,j)=>i+j)
    }else if (this.ndim==2){
      rst = this.toVector(this.data.map((x,i)=>a.T.data.map((y,j)=>x.dot(y))))
      return  this.setGradFn(rst,a,"dot")
    }
    return rst
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
  slice(...d){
    if (!Array.isArray(d)) d=[d]
    let v= new Vector([this])
    for (let idx=0;idx<d.length;idx++){
      v =v._slice(idx,0,0,(x)=>{
        let p=d[idx]
        x=x.data
        let data=[]
        if (!Array.isArray(p)) p=[p]
        let [s,t,o]=p
        let k=false
        if (s==undefined) {s=0;k=true}
        if (s<0) s=x.length + s
        if (t==undefined) t=(s==0&&k)?x.length:s+1
        if (t<0) t=x.length + t
        if (t>x.length) t=x.length
        if (o==undefined) o=1
        for (let i=s;i<t;i+=o){
          data.push(x[i])
        }
        //仅对最后的数值判断为一个值的时候才降低一维
        if (data.length==1 && !(data[0] instanceof Vector)) data=data[0] 
        return data
      })
      v= new Vector(v)
    }
    return v.squeeze()
  }
  _slice(axis=null,deep=0,ndim=0,method=null){
    if (axis==null){ //没指定轴
      if (method) return method(this)
    }
    if (!ndim) ndim=this.ndim
    if (deep == axis){ //axis小于最后两个维度
      return this.data.map((x,i)=>(deep<ndim-1)?x._slice(null,0,0,method):x)
    }
    let result = this.data.map((x,i)=>{ //递归到数据层
      if (i==0) deep++
      return (deep<ndim-1)?x._slice(axis,deep,ndim,method):x
    })
    return this.toVector(result)
  }
  take(axis=null,p){
    if (!Array.isArray(p)) p=[p]
    return this._take(axis,0,0,(x)=>{
      return new Vector(p.map(n=>{
        if (n>x.length-1) throw new Error(`${n} 超过向量边界`)
        return x.data[n]
      }))
    })
  }
  _take(axis=null,deep=0,ndim=0,method=null){
    if (axis==null){ //没指定轴
      if (method) return method(this)
    }
    if (!ndim) ndim=this.ndim
    if (deep == axis){ //axis小于最后两个维度
      return new Vector(this.data.map((x,i)=>x._take(null,0,0,method)))
    }
    let result = this.data.map((x,i)=>{ //递归到数据层
      if (i==0) deep++
      return (deep<ndim-1)?x._take(axis,deep,ndim,method):x
    })
    return this.toVector(result)
  }
  where(){}
  nonzero(){}
  
  hstack(a,deep=0,ndim=0){
    if (!ndim) ndim=this.ndim
    if (this.shape.slice(0,-1).toString()!=a.shape.slice(0,-1).toString())
      throw new Error(`(${this.shape.slice(0,-1).toString()})和(${a.shape.slice(0,-1).toString()})不符合横向堆叠要确保横向形状一致的要求`)
    let v= this.data.map((x,i)=>{
        if (i==0) deep++
        return deep<ndim - 1?x.hstack(a.data[i],deep,ndim):x.data.concat(a.data[i].data)
      })
    return new Vector(v)
  }
  vstack(a){
    a=new Vector(a)
    if (this.shape.slice(1).toString()!=a.shape.slice(1).toString())
      throw new Error(`(${this.shape.slice(1).toString()})和(${a.shape.slice(1).toString()})不符合纵向堆叠要确保纵向形状一致的要求`)
    return new Vector(this.value.concat(a.value),a.dtype)
  }
  vsplit(m){
    let n=[]
    let matrix=[]
    if (typeof m =="number"){
      let range = this.shape[0] 
      let fromIndex=0
      let count = m
      for(let idx=0;idx<count;idx++){
        let end=Math.ceil(range * (idx+1) / count) - 1 + fromIndex
        let start =Math.ceil( range * idx / count) +1 - 1 + fromIndex
        //console.log(`${idx}:${start}-${end},有${end-start+1}个元素`)
        matrix.push(this.slice([start,end+1]))
       }
    }else{
      let start=0
      matrix=m.map(end=>{
        let temp = this.slice([start,end])
        start = end
        return temp
      })
      matrix.push(this.slice([start,this.shape[0]]))
    }
    return matrix
  }
  
  hsplit(m){
    let n=[]
    let matrix=[]
    if (typeof m =="number"){
      let range = this.shape[this.ndim-1]
      let fromIndex=0
      let count = m
      for(let idx=0;idx<count;idx++){
        let end=Math.ceil(range * (idx+1) / count) - 1 + fromIndex
        let start =Math.ceil( range * idx / count) +1 - 1 + fromIndex
        //console.log(`${idx}:${start}-${end},有${end-start+1}个元素`)
        matrix.push(this.slice([],[start,end+1]))
       }
    }else{
      let start=0
      matrix=m.map(end=>{
        let temp = this.slice([],[start,end])
        start = end
        return temp
      })
      matrix.push(this.slice([],[start,this.shape[1]]))
    }
    return matrix  
  }

  split(m,axis=1){
    if (axis==1) return this.hsplit(m)
    if (axis==0) return this.vsplit(m)
    throw new Error("必须制定axis为0-纵向分割，1-横向分割,默认axis=1")
  }
  pad(s,mode="constant",v){
    if (this.ndim>3) throw new Error(`pad函数只支持一维、二维或三维`)
    if (this.ndim==1){
      if (!v) v=[0,0]
      if (typeof v == "number") v=[v,v]
      if (typeof s == "number") s=[s,s]
      let [x,y]=s
      let value
      let data=[]
      switch (mode){
        case "constant":
          value=0;break;
        case "mean":
          value=this.mean();break;
        case "maximum":
          value=this.max();break;
        case "minimum":
          value=this.min();break;
        default:
          value=0
      }
      for (let i=0;i<x;i++) data.push(mode=="constant"?v[0]:value)
      data=data.concat(this.data)
      for (let i=0;i<y;i++) data.push(mode=="constant"?v[1]:value)
      return new Vector(data)
    }else if (this.ndim==2){
      if (typeof s=="number") s=[[s,s],[s,s]]
      if (typeof s[0]=="number") s[0]=[s[0],s[0]]
      if (typeof s[1]=="number") s[1]=[s[1],s[1]]
      if (!v) v=[0,0]
      if (typeof v=="number") v=[v,v]
      let value,value1
      switch (mode){
        case "constant":
          value=dim.zeros(this.shape[1]);break;
        case "mean":
          value=this.mean(0);
          v=this.mean();
          break;
        case "maximum":
          value=this.max(0);
          v = this.max();
          break;
        case "minimum":
          value=this.min(0);
          v = this.min();
          break;
        default:
          value=dim.zros(this.shape[1])
      }
      let f,d=[]
      let data=this.data
      f=value.pad([s[0][1],s[1][1]],"constant",v)
      for (let i=0;i<s[0][0];i++) d.push(f)
      d = d.concat(data.map(x=>x.pad([s[0][1],s[1][1]],mode,v)))
      for (let i=0;i<s[1][0];i++) d.push(f)
  
      return new Vector(d)   
    }else if (this.ndim==3){
      return new Vector(this.data.map(x=>x.pad(s,mode,v)))
    } 
  }

  //SquareMatrix function
  lu(){//求三角阵
    this.ensureMatrix()
    let data = this.value
    let row = this.shape[0]
    let col = this.shape[1]
    let s = (row < col)?row:col
    for (let k=0;k<s;k++){
      let x=1/data[k][k]
      for (let i=k+1;i<row;i++){
        data[i][k] = data[i][k] * x        
      }
      for (let i=k+1;i<row;i++){
        for (let j=k+1;j<col;j++){
          data[i][j] = data[i][j] - data[i][k]*data[k][j]
        }
      }
    }
    return new Vector(data)
  }
  det(permute,lu=true){//行列式求值
    this.ensureSquareMatrix()
    if (lu) {  //lu分块分解快速计算det
      let x=1
      let m=this.lu()
      let row=m.shape[0]
      for (let i=0;i<row;i++){
        x=x*m.data[i].data[i]
      }    
      return x    
    }
    
    let data=this.value
    switch (row){
      case 2:
        return data[0][0]*data[1][1] - data[0][1]*data[1][0]
      case 3:
        return data[0][0]*data[1][1]*data[2][2] +
               data[1][0]*data[2][1]*data[0][2] +
               data[2][0]*data[0][1]*data[1][2] -
               data[0][2]*data[1][1]*data[2][0] -
               data[1][2]*data[2][1]*data[0][0] -
               data[2][2]*data[0][1]*data[1][0]
      default:
        if (!permute){
          let argN=[]
          for(let i=0;i<row;i++){
            argN.push(i)
          }
          permute = dim.permute(argN)
        }
        return permute.map(x=>{
          let invert = dim.invertCount(x)
          return x.split("").map((y,i)=>data[i][y])
                            .reduce((x,y)=>x*y)*(-1)**invert
        }).reduce((x,y)=>x+y)
    }
  }
  adjoint(){ //伴随矩阵
    this.ensureSquareMatrix()
    let data=this.value
    let arr=[]
    let det=[]
    let permute=[]
    let row=this.shape[0]
    let col=this.shape[1]
    switch (row){
      case 2:
        arr[0]=data[1][1]*(-1)**(1+1)
        arr[1]=data[0][1]*(-1)**(0+1)
        arr[2]=data[1][0]*(-1)**(1+0)
        arr[3]=data[0][0]*(-1)**(0+0)                                                 
        return new Vector(arr).reshape(2,2)
      default:
        let temp=[]
        for(let i=0;i<row-1;i++){temp.push(i)}
        permute = dim.permute(temp)
        for(let i=0;i<row;i++){
          let m = [...data]
          m.splice(i,1)
          for(let j=0;j<col;j++){
            let n = m.map(y=>{
              let yy=[...y]
              yy.splice(j,1);return yy
            })
           det.push(new Vector(n).det(permute)*(-1)**(i+j))
           //console.log(i,j,n,det)
          }
        }
        return new Vector(det).reshape(row,row).transpose()
    }
  }
  inv(){//求逆矩阵
    this.ensureSquareMatrix()
    let det = this.det()
    if (det==0) throw new Error("矩阵det=0,该矩阵不存在可逆矩阵")
    return this.adjoint().divide(det)
  }
  solve(b){//行列式求值
    this.ensureSquareMatrix()
    b=new Vector(b).reshape(this.shape[0],1)
    return this.inv().dot(b).value
  }
  trace(){
    this.ensureMatrix()
    let a=this.value
    let t=0
    for (let i=0;i<a.length;i++){
      for (let j=0;j<a[i].length;j++){
        if (i==j) t+=a[i][j]
      }
    }
    if (this.requiresGrad){
      t = new Vector([t])
      t.requiresGrad=true
      t.gradFn=grad.Operate.wrapper(this.gradFn,null,"tr")
    }
    return t
  }
  setGrad(bool=true){
    this.requiresGrad=bool
    this.grad=null
    this.gradFn=null
    if (bool && this.isLeaf){
      this.gradFn=new autograd.Variable(this)
    }
    return this
  }
  get isLeaf(){
    return !(this.gradFn instanceof autograd.Operate)
  }
  expression(){return this.gradFn && this.gradFn.expression()}
  gradExpression(){
    if (!this.gradFn) return null
    return this.gradFn.gradExpression()
  }
  backward(prevOp){
    if (!this.requiresGrad) throw new Error("after call setGrad(true) ,then use this function")
    if (prevOp) prevOp = new autograd.Constant(prevOp)
    let variables=this.gradFn.variables()
    variables.map(v=>{
      let op=this.gradFn.backward(prevOp,v)
      let a=dim.ensureUnit(op.eval())
      v.data.grad = v.data.grad?v.data.grad.add(a):a
    })  
  }
  gradClear(){
    this.gradFn.clearData()
    this.grad=null
  }
  onehot(n){
    let a=this.ensureCanOnehot()
    if (n==undefined) n=a.max()+1
    let b=dim.zeros([a.shape[0],n])
    a.value.map((x,i)=>b.data[i].data[x[0]]=1)
    return new Vector(b)
  }
  kron(a,indices=false,dir="v"){
    return this.data.map((x,i)=>{
      if (x instanceof Vector) return x.kron(a,indices,"h")
      if (indices){ 
        return a.data[i].mul(x)
      }else{
        return a.mul(x)
      }
    }).reduce((m,n)=>{
      if (dir=="h") {
        if (m.ndim==1) return new Vector(m.data.concat(n.data))
        return m.hstack(n)
      }
      if (dir=="v") {
        if (m.ndim==1) return new Vector(m.data.concat(n.data))
        return m.vstack(n)
      }
    })
  }
}
class Dim{
  constructor(){
    this.Vector = Vector
    this.Complex= Complex
    this.random = new Random(this)
    this.fft = fft
    this.nn = new NN(this)
    this.optim = new Optimizer()
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
  ensureSameShape(a,b){a.ensureSameShape(b);return true}
  ensureUnit(a){
    if (typeof a=="number") return new Vector([a])
    return a
  }
  ensure1D(...a){a.map(x=>x.ensure1D());return true}
  ensure2D(...a){a.map(x=>x.ensure2D());return true}
  ensure3D(...a){a.map(x=>x.ensure3D());return true}

  array(a,dtype){return new Vector(a,dtype)}
  toVector(a,dtype){a=this.ensureVector(a);return a}
  isArray(a){return Array.isArray(a)}
  flatten(a){a=this.ensureVector(a);return a.flatten()}
  transpose(a){a=this.ensureVector(a);return a.transpose()}
  copy(a){a=this.ensureVector(a);return a.copy()} 
  save(a,file){a=this.ensureVector(a);return a.save(file)}
  load(file){return new Vector(JSON.parse(fs.readFileSync(file,'utf8')))}
  
  invertCount(str){//计算字符串逆序个数
    let a = str.split("")
    let c=0
    while(a.length>1){
      let b=a.splice(0,1)[0]
      c+=a.map(x=>b>x?1:0).reduce((x,y)=>x+y)
    }
    return c
  }
  permute(arr){//求出数组元素的n!种组合
    let data=[]
    function inner(arr,s){
      let a=[...arr]
      a.map(x=>{
        if (a.length>1) return inner(a.filter(y=>y!=x),s+x)
        data.push(s+a[0])
      })
    }
    inner(arr,"")
    //console.log("permute:",arr.length,data.length)
    return data
  }
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
  fill(n,shape=1,dtype){
    return this.__reset(n,dtype,shape)
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
  squeeze(a){a=this.ensureVector(a);return a.squeeze()}
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
  
  replace(a,b){[a,b]=this.ensureVector(a,b);return a.replace(b)}

  randians(a){
    a=this.ensureVector(a);
    return (typeof a=="number")?a/180*Math.PI:a.randians()
  }
  sin(a){a=this.ensureVector(a);return (typeof a=="number")?Math.sin(a):a.sin()}
  cos(a){a=this.ensureVector(a);return (typeof a=="number")?Math.cos(a):a.cos()}
  tan(a){a=this.ensureVector(a);return (typeof a=="number")?Math.tan(a):a.tan()}
  asin(a){a=this.ensureVector(a);return (typeof a=="number")?Math.asin(a):a.asin()}
  acos(a){a=this.ensureVector(a);return (typeof a=="number")?Math.acos(a):a.acos()}
  atan(a){a=this.ensureVector(a);return (typeof a=="number")?Math.atan(a):a.atan()}
  asinh(a){a=this.ensureVector(a);return (typeof a=="number")?Math.asinh(a):a.asinh()}
  acosh(a){a=this.ensureVector(a);return (typeof a=="number")?Math.acosh(a):a.acosh()}
  atanh(a){a=this.ensureVector(a);return (typeof a=="number")?Math.atanh(a):a.atanh()}
  sinh(a){a=this.ensureVector(a);return (typeof a=="number")?Math.sinh(a):a.sinh()}
  cosh(a){a=this.ensureVector(a);return (typeof a=="number")?Math.cosh(a):a.cosh()}
  tanh(a){a=this.ensureVector(a);return (typeof a=="number")?Math.tanh(a):a.tanh()}
  log(a){a=this.ensureVector(a);return (typeof a=="number")?Math.log(a):a.log()}
  log2(a){a=this.ensureVector(a);return (typeof a=="number")?Math.log2(a):a.log2()}
  log10(a){a=this.ensureVector(a);return (typeof a=="number")?Math.log10(a):a.log10()}
  exp(a){a=this.ensureVector(a);return (typeof a=="number")?Math.exp(a):a.exp()}
  sqrt(a){a=this.ensureVector(a);return (typeof a=="number")?Math.sqrt(a):a.sqrt()}
  square(a){a=this.ensureVector(a);return (typeof a=="number")?a*a:a.square()}
  pow(a,n){a=this.ensureVector(a);return (typeof a=="number")?Math.pow(a,n):a.pow(n)}
  floor(a){a=this.ensureVector(a);return (typeof a=="number")?Math.floor(a):a.floor()}
  ceil(a){a=this.ensureVector(a);return (typeof a=="number")?Math.ceil(a):a.ceil()}
  around(a,n){
    a=this.ensureVector(a);
    if (typeof a=="number"){
      let a0=10**n
      return Math.round(a*a0)/a0
    }else{
      return a.around(n)
    }
  }
  abs(a){a=this.ensureVector(a);return (typeof a=="number")?Math.abs(a):a.abs()}
  
  add(a,b){
    [a,b]=this.ensureVector(a,b);
    if (typeof a=="number" && typeof b=="number") return a+b
    if (typeof a=="number" || a.isUnit){
      return b.add(a)
    }
    return a.add(b)
  }
  sub(a,b){
    [a,b]=this.ensureVector(a,b);
    if (typeof a=="number" && typeof b=="number") return a-b
    if (typeof a=="number" || a.isUnit){
      return b.neg().add(a)
    }
    return a.sub(b)
  }
  mul(a,b){
    [a,b]=this.ensureVector(a,b);
    if (typeof a=="number" && typeof b=="number") return a*b
    if (typeof a=="number" || a.isUnit){
      return b.mul(a)
    }
    return a.mul(b)
  }
  div(a,b){
    [a,b]=this.ensureVector(a,b);
    if (typeof a=="number" && typeof b=="number") return a/b
    if (typeof a=="number" || a.isUnit){
      return b.reciprocal().mul(a)
    }
    return a.div(b)
  }
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

  normal(a,N){a=this.ensureVector(a);return a.normal(N)}
  minmaxNormal(a){a=this.ensureVector(a);return a.minmaxNormal()}
  
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

  dot(a,b){
    [a,b]=this.ensureVector(a,b);
    if (typeof a=="number" && typeof b=="number") return a*b
    if (typeof a=="number" || a.isUnit){
      if (a==0) return 0
      throw new Error("dot 函数错误")
    }
    if (typeof b=="number" || b.isUnit){
      if (b==0) return 0
      throw new Error("dot 函数错误")
    }
    return a.dot(b)
  }
  matmul(a,b){return this.dot(a,b)}
  
  clip(a,m,n){a=this.ensureVector(a);return a.clip(m,n)}

  slice(a,...k){a=this.ensureVector(a);return a.slice(...k)}
  take(a,axis,p){a=this.ensureVector(a);return a.take(axis,p)}
  
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
  trace(a){a=this.ensureVector(a);return a.trace()}
  Constant(n){
    n=(typeof n=="number")?parseFloat(n):n;
    if (n instanceof autograd.Constant) return n
    return new autograd.Constant(n)
  }
  Variable(data,name){
    if (data instanceof autograd.Variable) return data
    return new autograd.Variable(data,name)
  }
  Operate(left,right,mode){
    return autograd.Operate.wrapper(left,right,mode)
  }
  onehot(a,n){a=this.ensureVector(a);return a.onehot(n)}
  kron(a,b,indices){[a,b]=this.ensureVector(a,b);return a.kron(b,indices)}
  print(a,lr,lc){a=this.ensureVector(a);a.print(lr,lc)}
  swapaxes(a,m,n){a=this.ensureVector(a);return a.swapaxes(m,n)}
}
exports.dim = new Dim()
exports.Vector = Vector
