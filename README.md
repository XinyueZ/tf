# TF-Hub

Here I try to summary some useful porches, materials, information about programming with [Tensorflow](https://www.tensorflow.org). 

#### [For beginner](tf_beginner.py) What is training, here clears step-by-step, coding pattern, loss-function, porch of coding with TF.
#### [For beginner+](tf_beginner_plus.py) What is optimizer, here clears what it does, how it works, why we need it.
    - Model: f(x) = a * x + b.
    - Target: Unknown
    - Data:  preparatory data
    - Result: loss= 0.0000157946196850389 a= [-1.00184119] b= [ 40.04899979] 
    - Training time: 400000
    - Rate = 0.0028
    
## Let's train some models

##### [quadratic](quadratic.py)
- Model: f(x) =a*x^2 + b*x + c
- Target: f(x) = x^2 + 2*x + 1
- Training time: 10000
- Rate: 0.5 (You will find the "a, b, c" have a very different change rate.)
- Random: 1 to 100 
- Loss(cost): self-defined[MSE](http://img.blog.csdn.net/20170522211318316?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbWFyc2poYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
> The loss(cost) is greater! 

> cost= 482080.9375000000000000000 a= 0.9484788775444030762 b= 1.2941216230392456055 c= 14.8984098434448242188

If you want to keep result near to target, let rate under 0.01 or less.

> rate: 0.0001

> cost= 156.8805389404296875000 a= 1.0002055168151855469 b= 1.9995691776275634766 c= 1.0115379095077514648


##### [reciprocal](reciprocal.py)
- Model: f(x) = a * (1 / x) + b
- Target: f(x) = 1 / x
- Training time: 5000
- Rate: 0.00015
- Random: 1 to 100
- Loss(cost):  [B-P-F-1](http://upload-images.jianshu.io/upload_images/4593922-4d24d17a6a2d6a8b.jpg?imageMogr2/auto-orient/strip)
> The loss(cost) is less! cost= 0.0000000326241789139 a= 1.0002250671386718750 b= 0.0000251986421062611

##### [sin](sin.py)
- Model: f(x) = W * sin(x) + b
- Target: f(x) = 3 * sin(x) + 5
- Training time: 5000
- Rate:  0.01
- Random: 0 to Pi
- Loss(cost): L2 loss + Gradient descent
> The loss(cost) is less! cost= 0.0000044736493691744 W= 2.9958639144897460938 b= 4.9998445510864257812
