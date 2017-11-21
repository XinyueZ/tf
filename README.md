# Repo
TF(Tensorflow)

# What 
Here all my TF codes. From "hello,world" to more "complex".

# Not for learning
All codes here are in Python, or Golang even Java(Kotlin), 
I don't use this repo for learning, purpose of this repo is provide significant 
codes for common tasks.


# Files

#### [Training: f(x) = x^2 + 2*x + 1 with model  f(x) =a*x^2 + b*x + c](quadratic.py)

- Training time: 10000
- Random: 1 to 100
- Target: f(x) = x^2 + 2*x + 1
- Model: f(x) =a*x^2 + b*x + c
- Loss(cost): self-defined[MSE](http://img.blog.csdn.net/20170522211318316?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbWFyc2poYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
>cost= 4117.466796875 W= 1.0045 b= 2.00109 c= 1.01033

#### [Training: f(x)=W*sin(x)+b](sin.py)

#### [HelloTensorflow](HelloTensorflow.go)  : Hello,world 

#### [OneDimenCalc](OneDimenCalc.go) : Do multiplication A x B .

#### [multiply](multiply.py): Multiplication sample.

#### [multiop](multiop.py): More than one operators.

#### [SimpleMatrixCalc](SimpleMatrixCalc.go) : Do the fellowing computing on matrix. 
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>A</mi>
  <mo>=</mo>
  <mrow>
    <mo>(</mo>
    <mtable columnalign="center center" rowspacing="4pt" columnspacing="1em">
      <mtr>
        <mtd>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mn>2</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mo>&#x2212;<!-- − --></mo>
          <mn>1</mn>
        </mtd>
        <mtd>
          <mo>&#x2212;<!-- − --></mo>
          <mn>2</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo>)</mo>
  </mrow>
  <mo>,</mo>
  <mi>x</mi>
  <mo>=</mo>
  <mrow>
    <mo>(</mo>
    <mtable rowspacing="4pt" columnspacing="1em">
      <mtr>
        <mtd>
          <mn>10</mn>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mn>100</mn>
        </mtd>
      </mtr>
    </mtable>
    <mo>)</mo>
  </mrow>
</math>
