dim =require('./dim.js').dim

global.CONSTANT= []
global.VARIABLE=[]
global.OPERATE=[]

class Autograd{
  constructor(){
    this._expressionStr=null
    this._grads={}
    this._data=null
    this.catch=true
  }
  setCatch(bool=true){
    if (bool){
      this.catch=true
    }else{
      this.catch=false
      this._data=null
      this._grads={}
      this._expressionStr=null
    }
    if (this.left) this.left.setCatch(bool)
    if (this.right) this.right.setCatch(bool)
  }
  clearData(){
    this._data=null
    if (this.left) this.left.clearData()
    if (this.right) this.right.clearData()
  }
  findOp(name){
    if (this.type=='Operate' && this.name==name) return this
    left = this.left &&this.left.findOp(name)
    right = left.right && this.right.findOp(name)
    return left?left:right?right:{}
  }
  shrink(){}
  factor(opStr){return [this]}
}
class Constant extends Autograd{
  constructor(data){
    super()
    if (data.type=="Constant") return this
    data = dim.ensureVector(data)
    let jsonData = data instanceof dim.Vector?JSON.stringify(data.value):JSON.stringify(data)
    let idx=global.CONSTANT.map(x=>x.data).indexOf(jsonData)
    if (idx!=-1) return global.CONSTANT[idx].object
    this.name = "const"+Math.random().toString().slice(-6)
    this.data = data
    this.type = "Constant"
    global.CONSTANT.push({name:this.name,data:jsonData,object:this})
    this._expressionStr=this.name
  }
  partialGradient(partial={},prevOp){
    let rst
    rst = new Constant(0)
    this._grads[this.name]=rst
    return rst
  }
  expression(){
    if (typeof this.data=="number") return this.data
    return JSON.stringify(this.data.value)
  }
  gradExpression(){
    return Object.entries(this._grads).map(entry=>{
        let m={}; 
        m[entry[0]]=entry[1].expression();
        return m
    })
  }
  eval(){return this.data}
  backward(){return this.partialGradient()}
  variables(){return []}
  isSame(a){
    if (!(a instanceof Constant)) return false
    if (this.name==a.name && this.data==a.data) return true
    return fasle
  }
}

class Variable extends Autograd{
  constructor(data,name){
    super()
    if (!name) name = "var"+Math.random().toString().slice(-6)
    //let idx=global.VARIABLE.map(x=>x.name).indexOf(name)
    //if (idx!=-1) return global.VARIABLE[idx]   
    this.name  = name
    this.data = data
    this.type = "Variable"
    global.VARIABLE.push({name:this.name})
  }
  partialGradient(partial={},prevOp){
    let rst
    if (prevOp==undefined) prevOp=new Constant(1)
    if (partial.name == this.name) {
      return prevOp
    }else{
      return new Constant(0)
    }
    if (partial.name == this.name){
      if ((typeof this.data=="number") || this.data.isUnit){ //标量
        if(partial.data.ndim==1) {
          //console.log("标量对标量求导")
          rst= new Constant(dim.ones(partial.data.size))
        }else if (partial.data.ndim==2) {
          //console.log("标量对矩阵求导")
          rst= new Constant(dim.ones(partial.data.shape).T)
        }else rst = new Constant(1)
      }else if (this.data.ndim==1){//向量
        if ((typeof partial.data=="number") || partial.data.isUnit){
          //console.log("向量对标量求导")
        }else if (partial.data.ndim==1){
          //console.log("向量对向量求导，理论应该是返回雅可比矩阵")
          rst = new Constant(dim.ones(this.data.size))
        }else {
          throw new Error("不支持向量关于矩阵的求导运算")
        }
      }else if (this.data.ndim==2){//矩阵
        if ((typeof partial.data=="number") || partial.data.isUnit){
          //console.log("矩阵对标量求导")
          rst = new Constant(dim.ones(this.data.shape))
        }else{ 
          //console.log("矩阵对矩阵求导")
          rst = new Constant(dim.ones(this.data.shape))
          //throw new Error("不支持矩阵关于向量或矩阵的求导运算")
        }
      }else{
        throw new Error("不支持超过两维的高阶求导")
      }
    }else{
      //console.log("对非自身变量求导为0")
      rst = new Constant(0)
    }
    this._grads[this.name]=rst
    return rst
  }
  expression(){
    return this.name
  }
  gradExpression(){
    return Object.entries(this._grads).map(entry=>{
        let m={}; 
        m[entry[0]]=entry[1].expression;
        return m
    })
  }
  eval(){return this.data}
  backward(){return this.partialGradient(this)}
  variables(){return [this]}
  isSame(a){
    if (!(a instanceof Variable)) return false
    if (this.name == a.name) return true
    return false
  }
}
class Operate extends Autograd{
  constructor(left,right=null,operate,name){
    super()
    if (typeof left=="number") this.left = new Constant(left)
    else this.left = left
    if (typeof right == "number") this.right = new Constant(right)
    else this.right = right
    
    if (!name) name = "op"+Math.random().toString().slice(-6)
    this.name=name
    
    this.operate = operate
    this.type = "Operate"
  }
  partialGradient(partial,prevOp){
    if (partial.type!="Variable") throw new Error("partial参数必须是Variable类型")
    if (this.catch && this._grads[partial.name]) return this._grads[partial.name]
    if (prevOp==undefined) prevOp=new Constant(dim.ones(this.eval().shape))
    let rst
    switch (this.operate){
      case "add":
      case "sub":{
        let part1= Operate.wrapper(this.left.partialGradient(partial,prevOp),this.right.partialGradient(partial,prevOp),this.operate)
        rst = part1
        break;
      }case "mul":{
        let part1 = Operate.wrapper(this.right,prevOp,"mul")
        let part2 = this.left.partialGradient(partial,part1)
        let part3 = Operate.wrapper(this.left,prevOp,"mul")
        let part4 = this.right.partialGradient(partial,part3)
        console.log("mul",part2.expression(),part4.expression())
        let part5 = Operate.wrapper(part2,part4,"add")
        rst = part5
        break;
      }case "div":{
        let part1 = Operate.wrapper(this.right,prevOp,"mul")
        let part2 = this.left.partialGradient(partial,part1)
        let part3 = Operate.wrapper(this.left,prevOp,"mul")
        let part4 = this.right.partialGradient(partial,part3)
        let part5 = Operate.wrapper(part2,part4,"sub")
        let part6 = Operate.wrapper(this.right,new Constant(2),"pow")
        let part7 = Operate.wrapper(part3,part4,"div")
        break;
      }case "pow":{
        let c = new Constant(this.right.eval() - 1)
        let part2 = Operate.wrapper(this.left,c,"pow")
        let part3 = Operate.wrapper(this.right,part2,"mul")
        let part4 = Operate.wrapper(part3,prevOp,"mul")
        rst = this.left.partialGradient(partial,part4)
        break;
      }case "square":{
        let part1 = Operate.wrapper(new Constant(2),this.left,"mul")
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        rst = this.left.partialGradient(partial,part2)
        break;
      }case "exp":{
        let part1 = Operate.wrapper(this,prevOp,"mul")
        let part2 = this.left.partialGradient(partial,part1)
        rst = part2
        break;
      }case "log":{
        let part1 = Operate.wrapper(new Constant(1),this.left,"div")
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        let part3 = this.left.partialGradient(partial,part2)
        rst = part3
        break;
      }case "log2":{
        let part1 = Operate.wrapper(new Constant(1/Math.log(2)),this.left,"div")
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        let part3 = this.left.partialGradient(partial,part2)
        rst = part3
        break;
      }case "log10":{
        let part1 = Operate.wrapper(new Constant(1/Math.log(10)),this.left,"div")
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        let part3 = this.left.partialGradient(partial,part2)
        rst = part3
        break;
      }case "abs":{
        let pre1 = Operate.wrapper(this.left,null,"abs")
        let part1 = Operate.wrapper(this.left,pre1,"div")
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        let part3 = this.left.partialGradient(partial,part2)
        rst = part3
        break;
      }case "sin":{
        let part1 = Operate.wrapper(this.left,null,"cos")
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        rst = this.left.partialGradient(partial,part2)
        break;
      }case "cos":{
        let part1 = Operate.wrapper(this.left,null,"sin")
        let part2 = Operate.wrapper(new Constant(-1),part1,"mul")
        let part3 = Operate.wrapper(part2,prevOp,"mul")
        rst = this.left.partialGradient(partial,part3)
        break;
      }case "tan":{
        let part1 = Operate.wrapper(this.left,null,"cos")
        let part2 = Operate.wrapper(part1,2,"pow")
        let part3 = Operate.Wrapper(new Constant(1),part2,"div")
        let part4 = Operate.wrapper(part3,prevOp,"mul")
        rst = this.left.partialGradient(partial,part4)
        break;
      }case "asin":{
        let part1 = Operate.wrapper(this.left,2,"pow")
        let part2 = Operate.wrapper(new Constant(1),part1,"sub")
        let part3 = Operate.wrapper(part1,null,"sqrt")
        let part4 = Operate.wrapper(new Constant(1),part3,"div")
        let part5 = Operate.wrapper(part4,prevOp,"mul")
        rst = this.left.partialGradient(partial,part5)
        break;
      }case "acos":{
        let part1 = Operate.wrapper(this.left,2,"pow")
        let part2 = Operate.wrapper(new Constant(1),part1,"sub")
        let part3 = Operate.wrapper(part1,null,"sqrt")
        let part4 = Operate.wrapper(new Constant(-1),part3,"div")
        let part5 = Operate.wrapper(part4,prevOp,"mul")
        rst = this.left.partialGradient(partial,part5)
        break;
      }case "atan":{
        let part1 = Operate.wrapper(this.left,2,"pow")
        let part2 = Operate.wrapper(new Constant(1),part1,"add")
        let part3 = Operate.wrapper(new Constant(1),part2,"div")
        let part4 = Operate.wrapper(part3,prevOp,"mul")
        rst = this.left.partialGradient(partial,part4)
        break;
      }case "sinh":{
        let part1 = Operate.wrapper(this.left,null,"cosh")
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        rst = this.left.partialGradient(partial,part2)
        break;
      }case "cosh":{
        let part1 = Operate.wrapper(this.left,null,"sinh")
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        rst = this.left.partialGradient(partial,part2)
        break;
      }case "tanh":{
        let part1 = Operate.wrapper(this.left,null,"cosh")
        let part2 = Operate.wrapper(part1,2,"pow")
        let part3 = Operate.wrapper(new Constant(1),part2,"div")
        let part4 = Operate.wrapper(part3,prevOp,"mul")
        rst = this.left.partialGradient(partial,part4)
        break;
      }case "sqrt":{
        let part1 = Operate.wrapper(this.left,new Constant(-0.5),"pow")
        let part2 = Operate.wrapper(part1,new Constant(0.5),"mul")
        let part3 = Operate.wrapper(part2,prevOp,"mul")
        let part4 = this.left.partialGradient(partial,part3)
        rst = part4
        break;
      }case "sum":{
        let part1 =new Constant(dim.fill(1,this.left.eval().shape))
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        let part3 = this.left.partialGradient(partial,part2)
        rst = part3
        break;
      }case "mean":{
        let part1 =new Constant(dim.fill(1/this.left.eval().size,this.left.eval().shape))
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        let part3 = this.left.partialGradient(partial,part2)
        rst = part3
        break;
      }case "max":{
        let zero =new Constant(dim.zero(this.left.eval().shape))
        let part1 =new Constant(zero)
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        let part3 = this.left.partialGradient(partial,part2)
        rst = part3
        break;
      }case "min":{
        let zero =new Constant(dim.zero(this.left.eval().shape))
        let part1 =new Constant(zero)
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        let part3 = this.left.partialGradient(partial,part2)
        rst = part3
        break;
      }case "dot":{
        let part1,part2,part3
        //直接求eval的方式也没有问题，因为上一层的反算结果已经完成了。这种方式的问题是
        //不能在总算式的gradFn.gradExpression()中看到各偏导的dot计算公式
        //let dLeft = new Constant(prevOp.eval().dot(this.right.eval().T))
        //let dRight = new Constant(this.left.eval().T.dot(prevOp.eval()))
        let tRight = Operate.wrapper(this.right,null,"T")
        let dLeft = Operate.wrapper(prevOp,tRight,"dot")
        let tLeft = Operate.wrapper(this.left,null,"T")
        let dRight = Operate.wrapper(tLeft,prevOp,"dot")
        if (this.left.name==partial.name){
          part1 = dLeft
          part2 = this.right.partialGradient(partial,dRight)
        }else if(this.right.name==partial.name){
          part1 = this.left.partialGradient(partial,dLeft)
          part2 = dRight
        }else{
          part1=this.left.partialGradient(partial,dLeft)
          part2=this.right.partialGradient(partial,dRight)
        }
        part3=Operate.wrapper(part1,part2,"add")
        rst = part3
        break;
      }case "T":{
        if (this.left.name==partial.name){
          let part1 = Operate.wrapper(prevOp,null,"T")
          rst = part1
        }else{
          let part1 = Operate.wrapper(prevOp,null,"T")
          let part2 = this.left.partialGradient(partial,part1)
          let part3 = Operate.wrapper(part2,null,"T")
          rst = part3
        }
        break;
      }case "tr":{
        let part1 = Operate.wrapper(this.left.partialGradient(partial,prevOp),null,"tr")
        rst = part1
        break;
      }case "relu":{
        let part1 = Operate.wrapper(this.left,null,"reluDeri")
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        let part3 = this.left.partialGradient(partial,part2)
        rst = part3
        break;
      }case "reluDeri":{
        return new Constant(this.eval())
      }case "sigmoid":{
        let part1 = Operate.wrapper(this.left,null,"sigmoidDeri")
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        let part3 = this.left.partialGradient(partial,part2)
        rst = part3
        break;
      }case "sigmoidDeri":{
        return new Constant(this.eval())
      }case "softmax":{
        let part1 = Operate.wrapper(this,prevOp,"softmaxDeri")
        let part2 = this.left.partialGradient(partial,part1)
        rst = part2        
        break;
      }case "softmaxDeri":{
        return new Constant(this.eval())
      }case "mseLoss":{
        let part1 = Operate.wrapper(this.left,prevOp,"mul")
        let part2 = Operate.wrapper(this.right,prevOp,"mul")
        let part3 = this.left.partialGradient(partial,part1)
        let part4 = this.right.partialGradient(partial,part2)
        rst = Operate.wrapper(part3,part4,"add")        
        break;
      }case "crossEntropy":{
        let part1 = Operate.wrapper(this.left,this.right,"crossEntropyDeri")
        let part2 = Operate.wrapper(part1,prevOp,"mul")
        let part3 = this.left.partialGradient(partial,part2)
        rst = part3        
        break;
      }case "crossEntropyDeri":{
        return new Constant(this.eval())
      }default:{
        throw new Error(`${this.operate}是不正确的operate`)   
      }
    }
    this._grads[partial.name]=rst
    return rst  
  }
  variables(v=[]){
    if (this.left && this.left.type=="Operate") v=this.left.variables(v)
    if (this.right && this.right.type=="Operate") v=this.right.variables(v)
    if (this.left && this.left.type=="Variable") {
      if (v.map(x=>x.name).indexOf(this.left.name)==-1)
        v.push(this.left)
    }
    if (this.right && this.right.type=="Variable") {
      if (v.map(x=>x.name).indexOf(this.right.name)==-1)
      v.push(this.right) 
    }
    return v
  }
  gradExpression(){
    return Object.entries(this._grads).map(entry=>{
        let m={}; 
        m[entry[0]]=entry[1].expression();
        return m
    })
  }
  expression(){
    if (this.catch && this._expressionStr) return this._expressionStr
    let rst
    switch (this.operate){
      case "add":{
        rst = "("+this.left.expression() + "+" + this.right.expression()+")"
        break;
      }case "sub":{
        let part1= this.left.expression()
        let part2= this.right.expression()
        if (part1=='0'){
          if (part2.slice(0,1)=='-') return `${part2.slice(1)}`  
          return `(-${part2})`
        }
        if (part2=='0')  return `${part1}`
        rst = `(${part1}-${part2})`
        break;
      }case "mul":{
        let part1=this.left.expression()
        let part2=this.right.expression()
        if (part1=='-1') return `-${part2}`
        if (part2=='-1') return `-${part1}`
        rst = `(${part1}*${part2})`
        break;
      }case "div":{
        let part1=this.left.expression()
        let part2=this.right.expression()
        if (part2=='-1') return `-${part1}`
        rst = `(${part1}/${part2})`
        break;
      }case "dot":{
        let part1=this.left.expression()
        let part2=this.right.expression()
        rst = `(${part1}×${part2})`
        break;
      }case "pow":{rst = this.left.expression() + "^" + this.right.expression();break;
      }case "square":{rst = this.left.expression() + "^" + 2;break;
      }case "exp":{rst = "e^"+this.left.expression() ;break;
      }case "log":{rst = "ln("+this.left.expression()+")";break;
      }case "log2":{rst = "log2("+this.left.expression()+")";break;
      }case "log10":{rst = "log10("+this.left.expression()+")";break;
      }case "abs":{rst = "|"+this.left.expression()+"|";break;
      }case "sin":{rst = "sin("+this.left.expression()+")";break;
      }case "cos":{rst = "cos("+this.left.expression()+")";break;
      }case "tan":{rst = "tan("+this.left.expression()+")";break;
      }case "asin":{rst = "asin("+this.left.expression()+")";break;
      }case "acos":{rst = "acos("+this.left.expression()+")";break;
      }case "atan":{rst = "atan("+this.left.expression()+")";break;
      }case "sinh":{rst = "sinh("+this.left.expression()+")";break;
      }case "cosh":{rst = "cosh("+this.left.expression()+")";break;
      }case "tanh":{rst = "cosh("+this.left.expression()+")";break;
      }case "asinh":{rst = "asinh("+this.left.expression()+")";break;
      }case "acosh":{rst = "acosh("+this.left.expression()+")";break;
      }case "atanh":{rst = "atanh("+this.left.expression()+")";break;
      }case "sqrt":{rst = "sqrt("+this.left.expression()+")";break;
      }case "sum":{rst = "sum("+this.left.expression()+")";break;
      }case "mean":{rst = "mean("+this.left.expression()+")";break;
      }case "max":{rst = "max("+this.left.expression()+")";break;
      }case "min":{rst = "min("+this.left.expression()+")";break;
      }case "relu":{rst = "relu("+this.left.expression()+")";break;
      }case "reluDeri":{rst = "reluDeri("+this.left.expression()+")";break;
      }case "sigmoid":{rst = "sigmoid("+this.left.expression()+")";break;
      }case "sigmoidDeri":{rst = "sigmoidDeri("+this.left.expression()+")";break;
      }case "softmax":{rst = "softmax("+this.left.expression()+")";break;
      }case "softmaxDeri":{rst = "softmaxDeri("+this.left.expression()+","+this.right.expression()+")";break;
      }case "mseLoss":{rst = "mseLoss("+this.left.expression()+","+this.right.expression()+")";break;
      }case "crossEntropy":{rst = "crossEntropy("+this.left.expression()+","+this.right.expression()+")";break;
      }case "crossEntropyDeri":{rst = "crossEntropyDeri("+this.left.expression()+","+this.right.expression()+")";break;
      }case "T":{rst = "T("+this.left.expression()+")";break;
      }case "tr":{rst = "tr("+this.left.expression()+")";break;
      }default:{
        rst = `${this.operate}算子错误`
      }
    }
    this._expressionStr = rst
    return rst 
  }
  
  static wrapper(left,right,operate){
    if (operate=="mul"){
      if (left.type=="Constant" && left.data==0) return new Constant(0)
      if (right.type=="Constant" && right.data==0) return new Constant(0)
      if (left.type=="Constant" && right.type=="Constant") return new Constant(dim.mul(left.data,right.data))
      if (left.type== "Constant" && left.data==1) return right
      if (right.type=="Constant" && right.data==1) return left
    }
    if (operate=="div"){
      if (left.type=="Constant" && left.data==0) return new Constant(0)
      if (right.type=="Constant" && right.data==0) throw new Error("错误的表达式") 
      if (left.type=="Constant" && right.type=="Constant") return new Constant(dim.div(left.data,right.data))
      if (right.type=="Constant" && right.data==1) return left
      if (left.isSame(right)) return new Constant(1)
    }
    if (operate=="add"){
      if (left.type=="Constant" && left.data==0) return right
      if (right.type=="Constant" && right.data==0) return left
      if (left.type=="Constant" && right.type=="Constant") return new Constant(dim.add(left.data,right.data))
    }
    if (operate=="sub"){
      if (right.type=="Constant" && right.data==0) return left
      if (left.type=="Constant" && right.type=="Constant") return new Constant(dim.sub(left.data,right.data))
      if (left.isSame(right)) return new Constant(0)
    }
    if (operate=="pow"){
      if (right.type=="Constant" && right.data==0) return new Constant(1)
      if (right.type=="Constant" && right.data==1) return left
      if (left.type=="Constant" && left.data==1) return new Constant(1)
    }
    if (operate=="tr"){
      if (left.type=="Operate" && left.left.type=="Operate" && left.left.operate=="T") return new Operate(left.left,null,"tr")
    }
    return new Operate(left,right,operate)
  }
  eval(){
    if (this.catch && this._data !=null) return this._data
    let rst
    switch (this.operate){
      case "add":{rst= dim.add(this.left.eval(),this.right.eval());break}
      case "sub":{rst= dim.sub(this.left.eval(),this.right.eval());break}
      case "mul":{rst= dim.mul(this.left.eval(),this.right.eval());break}
      case "div":{rst= dim.div(this.left.eval(),this.right.eval());break}
      case "pow":{rst= dim.pow(this.left.eval(),this.right.eval());break}
      case "square":{rst= dim.square(this.left.eval());break}
      case "sqrt":{rst= dim.sqrt(this.left.eval());break}
      case "exp":{rst= dim.exp(this.left.eval());break}
      case "log":{rst= dim.log(this.left.eval());break}
      case "log2":{rst= dim.log2(this.left.eval());break}
      case "log10":{rst= dim.log10(this.left.eval());break}
      case "abs":{rst= dim.abs(this.left.eval());break}
      case "sin":{rst= dim.sin(this.left.eval());break}
      case "cos":{rst= dim.cos(this.left.eval());break}
      case "tan":{rst= dim.tan(this.left.eval());break}
      case "asin":{rst= dim.asin(this.left.eval());break}
      case "acos":{rst= dim.acos(this.left.eval());break}
      case "atan":{rst= dim.atan(this.left.eval());break}
      case "sinh":{rst= dim.sinh(this.left.eval());break}
      case "cosh":{rst= dim.cosh(this.left.eval());break}
      case "tanh":{rst= dim.tanh(this.left.eval());break}
      case "asinh":{rst= dim.asinh(this.left.eval());break}
      case "acosh":{rst= dim.acosh(this.left.eval());break}
      case "atanh":{rst= dim.acosh(this.left.eval());break}
      case "sum":{rst=dim.sum(this.left.eval());break}
      case "mean":{rst=dim.mean(this.left.eval());break}
      case "max":{rst=dim.max(this.left.eval());break}
      case "min":{rst=dim.min(this.left.eval());break}
      case "dot":{rst=dim.dot(this.left.eval(),this.right.eval());break}
      case "T":{
        rst=this.left.eval()
        if (typeof rst == "number") break;
        rst = rst.T
        break;}
      case "tr":{rst=dim.trace(this.left.eval());break;}
      case "relu":{rst=dim.nn.relu(this.left.eval());break}
      case "reluDeri":{rst=dim.nn.reluDeri(this.left.eval());break}
      case "sigmoid":{rst=dim.nn.sigmoid(this.left.eval());break}
      case "sigmoidDeri":{rst=dim.nn.sigmoidDeri(this.left.eval());break}
      case "softmax":{rst=dim.nn.softmax(this.left.eval());break}
      case "softmaxDeri":{rst=dim.nn.softmaxDeri(this.left.eval(),this.right.eval());break}
      case "mseLoss":{rst=dim.nn.mseLoss(this.left.eval(),this.right.eval());break}
      case "crossEntropy":{rst=dim.nn.crossEntropy(this.left.eval(),this.right.eval());break}
      case "crossEntropyDeri":{rst=dim.nn.crossEntropyDeri(this.left.eval(),this.right.eval());break}
      default:{
        throw new Error(`unknown operate {${this.operate}}`)  
      }
    }
    this._data = rst
    return rst
  }
  backward(prevOp,partial){
    if (!partial) partial=this.variables()[0]
    if (!partial || partial.type!="Variable") throw new Error('partial参数必须Variable类型')
    return this.partialGradient(partial,prevOp)
  }
  isSame(a){
    let leftEqual,rightEqual
    if (a.type!="Operate") return false
    leftEqual =  this.left.isSame(a.left)
    if (a.right==null && this.right==null){
      rightEqual = true
    }else if (a.right==null || this.right==null){
      rightEqual = false
    }else{
      rightEqual = this.right.isSame(a.right)
    }
    if (leftEqual && rightEqual && this.type=="Operate"){
      if (this.operate == a.operate) return true
    }
    return false
  }
  shrink(){
    let left ,right 
    left = this.left.factor("add")
    right = this.right.factor("add")
    for (let i of left){
      for (let j of right){
        console.log("add:",i.name,"=",j.name)
        if (i.isSame(j)) console.log("shrink",i,j)
      }
    }
    left = this.left.factor("mul")
    right = this.right.factor("mul")
    for (let i of left){
      for (let j of right){
        console.log("mul:",i.name,"=",j.name)
        if (i.isSame(j)) console.log("shrink",i,j)
      }
    }
  }
  factor(opStr,aFactor){
    if (aFactor==undefined) aFactor=[]
    console.log(this.operate)
    if (this.operate!=opStr) return aFactor
    aFactor.push(this.left)
    aFactor.push(this.right)
    this.left.factor(opStr,aFactor)
    this.right.factor(opStr,aFactor)
    return aFactor
  }
}

exports.Constant = Constant
exports.Variable = Variable
exports.Operate = Operate

