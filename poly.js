Complex = require('./complex.js')

class Poly {
  constructor(a,dtype=Float32Array){
    this.c=a  
    this.coef=this.c
    this.o=this.c.length - 1
    this.order = this.o
    //自动判定复数
    if (Array.isArray(a) && (a[0] instanceof Complex)) dtype=Complex
    this.dtype=dtype
  }
  ensurePoly(p){if (!(p instanceof Poly)) throw new Error("参数必须是Poly对象")}
  add(p){
    this.ensurePoly(p);
    if (this.c==0) return p
    let x = [...this.c]
    let y = [...p.c]
    for (let i=0 ; i<Math.abs(x.length - y.length);i++){
      (x.length>y.length)?y.unshift(0):x.unshift(0)
    }
    if (this.dtype == Complex){
      return new Poly(x.map((v,i)=>v.add((y[i]!=undefined)?y[i]:0)))
    }
    return new Poly(x.map((v,i)=>v+(y[i]!=undefined)?y[i]:0))
  }
  sub(p){
    this.ensurePoly(p);
    if (this.c==0) return p
    let x = [...this.c]
    let y = [...p.c]
    for (let i=0 ; i<Math.abs(x.length - y.length);i++){
      (x.length>y.length)?y.unshift(0):x.unshift(0)
    }
    if (this.dtype == Complex){
      return new Poly(x.map((v,i)=>v.sub((y[i]!=undefined)?y[i]:0)))
    }
    return new Poly(x.map((v,i)=>v-((y[i]!=undefined)?y[i]:0)))
  }
  mul(p){
    if (this.c==0) return p
    if (typeof p=="number") {
      if (this.dtype==Complex){
        return new Poly(this.c.map(x=>x.mul(p)))
      }
      return new Poly(this.c.map(x=>x*p))
    }
    if (p instanceof Complex) {
      if (this.dtype==Complex){
        return new Poly(this.c.map(x=>x.mul(p)))
      }
      throw new Error("不支持非复数多项式与复数相乘")
    }
    let c1=this.c
    let c2=p.c
    let T=[]
    if (this.dtype == Complex){
      c1.map((x,i)=>c2.map((y,j)=>(T[i+j]!=undefined)?T[i+j]=T[i+j].add(x.mul(y)):T.push(x.mul(y))))
      return new Poly(T)
    }
    c1.map((x,i)=>c2.map((y,j)=>(T[i+j]!=undefined)?T[i+j]+=x*y:T.push(x*y)))
    return new Poly(T)
  }
  div(p){
    this.ensurePoly(p);
    if (this.c==0) return p
    let c1=[...this.c] 
    let c2=[...p.c]
    let c1l = c1.length
    let c2l = c2.length
    let r=[]
    let l=[]
    let ta,tb;
    ta=0;
    for(let i=0;i<c1l-c2l+1;i++){
      if (this.dtype == Complex){
        r[i]=c1[i].div(c2[0])
      }else{
        r[i]=c1[i]/c2[0];
      }
      tb=ta;
      for(let j=0;j<c2l;j++){
        if (this.dtype == Complex){
          c1[tb]=c1[tb].sub(r[i].mul(c2[j]))
        }else{
          c1[tb]-=r[i]*c2[j];
        }
        tb+=1;
      }
      ta+=1;
    }
    ta=0
    for(let i=0;i<c1.length;i++){
      if (this.dtype == Complex){
        if (!ta && Math.abs(c1[i].real)<=1e-05 && Math.abs(c1[i].imag)<=1e-05) continue
      }else{
        if (!ta && Math.abs(c1[i])<=1e-05) continue
      }
      l[ta]=c1[i];
      ta+=1
    }
    //{q:quotient,r:remainder}
    return {"q":new Poly(r),"r":new Poly(l)}
  }
  pow(n){
    if (n==0) return new Poly([1])
    let p=this
    for (let i=0;i<=n-2;i++){
      p=p.mul(p)
    }
    return p
  }
  val(a){
    if (Array.isArray(a)){
      if (this.dtype == Complex){
        return a.map(x=>this.c.map((v,i)=>
          v.mul((x instanceof Complex)?x.pow(this.o-i):x**(this.o-i))).reduce((m,n)=>m.add(n)))
      }
      return a.map(x=>this.c.map((v,i)=>v*x**(this.o-i)).reduce((m,n)=>m+n))
    }
    if (this.dtype == Complex){
      if (a instanceof Complex)
        return this.c.map((v,i)=>v.mul(a.pow(this.o-i))).reduce((m,n)=>m.add(n))
      return this.c.map((v,i)=>v.mul(a**(this.o-i))).reduce((m,n)=>m.add(n))
    }
    return this.c.map((v,i)=>v*a**(this.o-i)).reduce((m,n)=>m+n)
  } 
  deriv(){
    let c=this.c
    let p=c.length - 1
    if (this.dtype == Complex)
      return new Poly(this.c.map((x,i)=>x.mul(p-i)).slice(0,p))    
    return new Poly(this.c.map((x,i)=>x*(p-i)).slice(0,p))    
  }
  integ(){
   let c=this.c
   let p=c.length - 1
   let c1
   if (this.dtype==Complex){
     c1=this.c.map((x,i)=>x.div(p-i+1))
     c1.push(new Complex(0,0))
   }else{
     c1=this.c.map((x,i)=>x/(p-i+1))
     c1.push(0)
   }
   return new Poly(c1)
  }
  roots(){} 
  lagrange(points) {
    let p = [];
    for (let i=0; i<points.length; i++) {
      p[i] = new Poly([1,-points[i][0]]);
    }

    let sum = new Poly([]);
    for (let i=0; i<points.length; i++) {
      let mpol=new Poly([])
      let factor=1
      for (let j=0; j<points.length; j++){
        if (j==i) continue
        mpol = mpol.mul(p[j])
        factor = factor * (points[i][0]-points[j][0])
      }
      factor = points[i][1] / factor;
      mpol = mpol.mul(factor)
      //console.log(i,factor,points[i][1],mpol.c)
      sum = sum.add(mpol);
    }
    return sum;
  }
  toVector(){return new Vector(this.c)}
  toString(){
    if (this.dtype==Complex){
      return this.c.map((a,i)=>{
        let s1,s2
        if (a.real==0 && a.imag==0) return ''
        s1 = "+"+a.toString()
        s2 = (this.o-i>=2)?'x^'.concat(this.o - i):(this.o-i==1)?'x':''
        return s1+s2
        }).join("").slice(1)
    }
    return this.c.map((a,i)=>{
      let s1,s2
      if (a==0) return ''
      s1 = (a==1)?"+":((a==-1)?"-":((a>0)?"+"+a.toString():a.toString()))
      if (i==0 && a<0) s1="+"+s1
      s2 = (this.o-i>=2)?'x^'.concat(this.o - i):(this.o-i==1)?'x':''
      return s1+s2
      }).join("").slice(1)
  }
}

exports.Poly = Poly
