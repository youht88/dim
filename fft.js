Complex = require('./complex.js').Complex

class FFT{
  rader(a){ // 二进制平摊反转置换 O(logn)  
    let len = a.length
    let j = len >>1
    for (let i = 1;i < len - 1;i++){
      if (i < j) {
        //swap(a[i], a[j]);
        let temp=a[i]
        a[i]=a[j]
        a[j]=temp
      }
      let k = len>>1;
      while (j >= k){
        j -= k;
        k>>=1;
      }
      if (j < k) j += k;
    }
    return a
  }
  fft(a,on=1){ //FFT:on=1; IFFT:on=-1
    a=this.padZero(a)
    a=this.rader(a);
    let len = a.length
    for (let h = 2;h <= len;h <<= 1){ //计算长度为h的DFT
      let wn = new Complex(Math.cos(-on * 2 * Math.PI / h), Math.sin(-on * 2 * Math.PI / h));//单位复根 e^(2*PI/m),用欧拉公式展开
      for (let j = 0;j < len;j += h){
        let w = new Complex(1, 0); //旋转因子
        for (let k = j;k < j + (h>>1);k++){
          let u = a[k];
          let t = w.mul(a[k + (h>>1)]);
          a[k] = u.add(t);   //蝴蝶合并操作
          a[k + (h>>1)] = u.sub(t);
          w = w.mul(wn);  //更新旋转因子
        }
      }
    }
    if (on == -1){ //如果是傅立叶逆变换
      for (let i = 0;i < len;i++){
        a[i].real /= len;
      }
    }
    return a
  }
  ifft(a){
    return this.fft(a,-1)
  }
  padZero(a){
    let len = 2** (parseInt(Math.log2(a.length - 1))+1)
    let num = len - a.length
    for (let i=0;i<num;i++){
      if (a[0] instanceof Complex){
        a.unshift(new Complex(0,0))
      }else{
        a.unshift(0)
      }
    }
    
    if (a[0] instanceof Complex)
      return a
    let b=new Complex(a)
    return b
  }
  fftshift(){}
  ifftshift(){}
  fftfreq(){}
  //fft(a,n,axis=0){}
}

exports.fft = new FFT()