
class GF{
  constructor(g=3,p=123479){
    this.init(g,p)
    this.gf8={
    }
    //p=2^256 − 2^32 − 2^9 − 2^8 − 2^7 − 2^6 − 2^4 − 1
    this.curve={
      name:"secp256k1",
      a:0n,
      b:7n,
      p:0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2Fn,
      g:{x:0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798n,
         y:0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8n},
      n:0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141n,
      h:1n
    }
   /*
   this.curve={
     name:"test",
     a:4n,
     b:20n,
     p:29n,
     g:{x:13n,y:23n},
     n:37n
   }
   
   this.curve={
     name:"test1",
     a:16546484n,
     b:4548674875n,
     p:15424654874903n,
     g:{x:6478678675n,y:5636379357093n},
     n:null
   }
   */
  }
  curveInit(a,b,p,g,name){
    this.curve.name=name
    this.curve.a=BigInt(a)
    this.curve.b=BigInt(b)
    this.curve.p=BigInt(p)
    this.curve.g={x:BigInt(g.x),y:BigInt(g.y)}
    return this.curve
  }
  init(g,p){
    this.gn=BigInt(g)
    this.pn=BigInt(p)
  }
  E(x){
    let xn=BigInt(x)
    return (this.gn**xn)%this.pn
  }
  verify(sum,...e){
    let en=e.map(x=>BigInt(x))
    let sumn = BigInt(sum)
    let ep=en.reduce((x,y)=>x*y)%this.pn
    return (ep == this.E(sumn))
  }
  isPrime(d){
    let dn = BigInt(d) 
    let i=Math.floor(Math.sqrt(d))
    if (dn==2n || dn==3n || dn==5n) return true
    if (dn%2n==0) return false
    if (dn%6n!=1 && dn%6n!=5) {
      for (let j=3n;j<=9n;i++){
        if (dn%j==0){
          console.log(`${j}*${dn/j}=${dn}`)
          return false
        }
      }
    }
    for (let j=5n;j<=i;j+=6n){
      if (dn%j==0){
        console.log(`${j}*${dn/j}=${dn}`)
        return false
      }
      if (dn%(j+2n)==0){
        console.log(`${j+2n}*${dn/(j+2n)}=${dn}`)
        return false
      }
    }
    return true
  }
  mod(a,p=null){
    let a1,ax,ay
    let pn=this.curve.p
    if (p!=null)
      pn = BigInt(p)
    if (a instanceof Complex){
      ax = a.real>0 ? BigInt(a.real)%pn : pn-BigInt(-a.real)%pn
      ay = a.imag>0 ? BigInt(a.imag)%pn : pn-BigInt(-a.imag)%pn
      return new Complex(ax,ay)
    }
    a1 = a>0 ? BigInt(a)%pn : pn-BigInt(-a)%pn
    return a1
  }
  invmod(a,p=null){
    let a1
    let an = a>0?BigInt(a):BigInt(-a)
    let pn = this.curve.p
    if (p!=null)
      pn = BigInt(p)
    //a1=this.mod(an**(pn-2n),pn)
    a1=this.inv2(an,pn)
    return a<0 ? pn-a1 : a1 
  }
  add(a,b,p){
    if (a instanceof Complex){
      return this.mod(a.add(b),p)
    }else if (b instanceof Complex){
      return this.mod(b.add(a),p)
    }
    return this.mod(BigInt(a)+BigInt(b),p||this.curve.p)
  }
  sub(a,b,p){
    if (a instanceof Complex){
      return this.mod(a.sub(b),p)
    }
    return this.mod(BigInt(a)-BigInt(b),p||this.curve.p)
  }
  mul(a,b,p){
    if (a instanceof Complex){
      return this.mod(a.mul(b),p)
    }else if (b instanceof Complex){
      return this.mod(b.mul(a),p)
    }
    return this.mod(BigInt(a)*BigInt(b),p||this.curve.p)
  }
  div(a,b,p){
    return this.mod(BigInt(a)*this.invmod(b,p),p||this.curve.p)
  }
  subtract(a,b,p){return this.sub(a,b,p)}
  multiply(a,b,p){return this.mul(a,b,p)}
  divide(a,b,p)  {return this.div(a,b,p)}
  
  inv(a,p){//乘法逆元
    let inv=[]
    inv[1] = 1;
    for(let i=2;i<a;i++)
        inv[i]=(p-parseInt(p/i))*inv[p%i]%p;
        
    return inv
  }
  inv1(a,p){
    if(a==1) return 1
    return (p-parseInt(p/a))*(this.inv1(p%a,p))%p
  }
  inv2(a,p){
    a=BigInt(a)
    p=BigInt(p)
    let res=1n,base=a%p;
    let b=p-2n
    while(b)
    {
        if(b&1n)
          res=(base*res)%p;
        base=(base*base)%p;
        b>>=1n;
    }
    return res;
  }
  gcd(a,b){//求最大公约数
    let k=parseInt(a/b);
    let remainder = a%b;
    while (remainder !=0){
      a = b;
      b = remainder
      k = parseInt(a/b)
      remainder = a%b
    }
    return b
  }
  lcm(a,b){//求最小公倍数
    //Finds the least common multiple of a and b
  }
  curveNav(m,p){
    let x = BigInt(m.x)
    let y = BigInt(this.mod(-m.y,p))
    return {x:x,y:y}  
  }
  curveAdd(m,n,p){
    let lambda,x,y
    m.x=BigInt(m.x)
    m.y=BigInt(m.y)
    n.x=BigInt(n.x)
    n.y=BigInt(n.y)
    let pn=this.curve.p
    if (p!=null) pn=BigInt(p)
    if (m.x==n.x && m.y==n.y)
      //lambda = BigInt(this.mod(this.mod(3n*m.x**2n+this.curve.a,pn)*this.invmod(2n*m.y,pn),pn))
      lambda = this.div(this.add(3n*m.x**2n,this.curve.a,pn),2n*m.y,pn)
    else
      //lambda = BigInt(this.mod(this.mod(n.y - m.y,pn)*this.invmod(n.x - m.x,pn),pn))
      lambda = this.div(this.sub(n.y,m.y,pn),this.sub(n.x,m.x,pn),pn)
    //console.log("lambda:",lambda)
    x=this.mod(lambda**2n - m.x - n.x,pn)
    y=this.mod(lambda*(m.x-x)-m.y,pn)
    
    return {x:x,y:y}
  }
  curveSub(m,n,p){
    return this.curveAdd(m,this.curveNav(n,p))
  }
  curveMul(k,g,p,m1=null){
    let gn=this.curve.g
    if (g!=null) gn=g
    let pn = this.curve.p
    if (p!=null) pn=p
    let sign=1
    k=BigInt(k)
    if (k%2n==0) sign=0
    if (k==1n) return gn
    if (k>3n) {
      //console.log("=====>",k)
      let k0=k/2n
      m1 = this.curveMul(k0,gn,pn,m1)
    }
    if (!m1)
      m1=this.curveAdd(gn,gn,pn)
    else {
      m1=this.curveAdd(m1,m1,pn)
    }
    if (sign!=0)
      m1 = this.curveAdd(m1,gn,pn)
    //console.log(k,sign,m1)
    return m1
  }
  polyAdd(p1,p2){return p1^p2}
  polySub(p1,p2){return this.polyAdd(p1,p2)}
  polyMul(u,v) {
    let p = 0;
    for (let i = 0; i < 8; ++i) {
      if (u & 0x01) {
        p ^= v;
      }
      let flag = (v & 0x80);
      v <<= 1;
      if (flag) {
          v ^= 0x1B;  /* P(x) = x^8 + x^4 + x^3 + x + 1 */
      }
      u >>= 1;
    }
    return p;
  }
}
class circuitR1CS{
  constructor(){
    this.gates = []
    this.vars = set()
  }
  
}
/*
class CircuitGenerator:

    def __init__(self):
        self.gates = []
        self.vars = set()

    def _new_var(self, var):
        if var in self.vars:
            raise Exception("'{}' is already set!".format(var))
        self.vars.add(var)

    def mov(self, result, a):
        l = {'1': a} if type(a) is int else {a: 1}
        r = {'1': 1}
        o = {result: 1}
        self._new_var(result)
        self.gates.append((l, r, o))

    def mul(self, result, a, b):
        l = {'1': a} if type(a) is int else {a: 1}
        r = {'1': b} if type(b) is int else {b: 1}
        o = {result: 1}
        self._new_var(result)
        self.gates.append((l, r, o))

    def inv(self, result, a):
        l = {result: 1}
        r = {'1': a} if type(a) is int else {a: 1}
        o = {'1': 1}
        self._new_var(result)
        self.gates.append((l, r, o))

    def neg(self, result, a):
        self.mul(result, '-1', a)

    def add(self, result, a, b):
        if type(a) is int and type(b) is int:
            self.mov(result, a + b)
            return
        if a == b:
            self.mul(result, a, 2)
            return
        l = {'1': a} if type(a) is int else {a: 1}
        l.update({'1': b} if type(b) is int else {b: 1})
        r = {'1': 1}
        o = {result: 1}
        self._new_var(result)
        self.gates.append((l,r,o))

    def compile(self):
        syms = set()
        for gate in self.gates:
            for part in gate:
                syms.update(part.keys())
        syms = {sym: i for i,sym in enumerate(list(syms))}
        LRO = [[[0] * len(syms) for i in range(len(self.gates))] for i in range(3)]
        for i, gate in enumerate(self.gates):
            for j in range(3):
                for k,v in gate[j].items():
                    LRO[j][i][syms[k]] = v
                LRO[j][i] = Vector(LRO[j][i])
        return R1CSCircuit(syms, LRO[0], LRO[1], LRO[2])
*/

exports.gf      = new GF()
