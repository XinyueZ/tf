# TF(Tensorflow).

Use TF to train different math-models. Use different taste on different Loss(cost) methods.
 
#### [Training: f(x) = x^2 + 2*x + 1 with model  f(x) =a*x^2 + b*x + c](quadratic.py)

- Training time: 10000
- Random: 1 to 100
- Target: f(x) = x^2 + 2*x + 1
- Model: f(x) =a*x^2 + b*x + c
- Loss(cost): self-defined[MSE](http://img.blog.csdn.net/20170522211318316?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbWFyc2poYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
> The loss(cost) is greater! cost= 4117.466796875 W= 1.0045 b= 2.00109 c= 1.01033

#### [Training: f(x)=W*sin(x)+b](sin.py)
- Training time: 5000
- Random: 0 to Pi
- Target: f(x) = 3 * sin(x) + 5
- Model: f(x) = W*sin(x) + b
- Loss(cost): L2 loss + Gradient descent
> The loss(cost) is less! cost= 3.73696e-06 W= 2.99604 b= 4.99967  
