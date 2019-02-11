
class Complex{
  constructor(r=0,j=0){
    if (typeof(r)!="number" || typeof(j)!="number") {
      console.log(r,j)
      throw new Error("复数定义不合法")
    }
    this.real = r
    this.imag = j
  }
  add(c){return new Complex(this.real+((c.real!=undefined)?c.real:c),this.imag+((c.imag!=undefined)?c.imag:0))}
  sub(c){return new Complex(this.real-((c.real!=undefined)?c.real:c),this.imag-((c.imag!=undefined)?c.imag:0))}
  mul(c){
    if (c instanceof Complex){
      return new Complex(this.real*c.real-this.imag*c.imag,
                         this.real*c.imag+this.imag*c.real)
    }
    return new Complex(this.real*c,this.imag*c)
  }
  div(c){
    if (c instanceof Complex){
      let dis=c.real**2+c.imag**2
      return new Complex((this.real*c.real+this.imag*c.imag)/dis,
                         (this.imag*c.real-this.real*c.imag)/dis)
    }
    return new Complex(this.real/c,this.imag/c)
  }
  pow(n){
    if (n==0) return new Complex(1,0)
    let p=this
    for (let i=0;i<=n-2;i++){
      p=p.mul(p)
    }
    return p
  }
  conjugate() { //求每个复数的共轭复数
    return new Complex(this.real,value.imag * -1);
  }

  magnitude() { //求每个复数到原点的长度,即模
    return Math.sqrt(this.real**2 + this.imag**2);
  }
  toString(){
    return "("+this.real+"+"+this.imag+"j)"
  }
}

exports.Complex = Complex