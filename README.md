# Cross Entropy Method

## Introduction

>  The Cross Entropy Method (CEM) deleveloped by Reuven Rubinstein is a general Monte Corlo approach to combinatorial and continuous multi-extremal optimization and importance sampling. 
>
> -- from [Wikipedia Cross-entorpy method](https://en.wikipedia.org/wiki/Cross-entropy_method)

## Generic CE Algorithm

The idea is to random sample the data and iterate for `maxits` to approach the target function $f$.

1. choose initial parameters $v^{(0)},\ \mu^{(0)}$ and $\sigma^{(0)};$ set $t$ = 1

2. generate `N `samples $X_1, X_2, ..., X_n$ from Gaussian distribution base on $\mu^{(t)}, \sigma^{(t)}$

3. solve for $v^{(t)}$, where:

   $$v^{(t)} = argmin_{x\in X}\ f(x)$$

4.  select the best `Ne` samples to update $\mu^{(t)}, \sigma^{(t)}​$

5. If convergence is reached then **stop**; otherwise, increase $t$ by 1 and reiterate from step 2.

## My Implementation

There are two phases in my implementation.

1. To update all dimension, this part is same as the original CEM
2. Update only one dimension at step 2.

## Class Introduction

```python
class CEM()
	def __init__(self, func, d, maxits1=500, maxits2=500, N=100, Ne=10, argmin=True):
        self.func = func            # target function
        self.d = d                  # dimension of function input X
        self.maxits1 = maxits1      # maximum iteration of update all dimension
        self.maxits2 = maxits2      # maximum iteration of update one dimension
        self.N = N                  # sample N examples each iteration
        self.Ne = Ne                # using better Ne examples to update mu and sigma
        self.reverse = not argmin   # try to maximum or minimum the target function
        self.init_coef = 10         # sigma initial value
```

## Usage 

### import class

``` python
from cem import CEM
```

### init class

```python
cem = CEM(my_func, 3)
```

### evalution

```python
cem.eval()
```

## Example

### function without another input

```python
from cem import CEM

def my_func(x):
    return x[0]*x[0] + x[1]*x[1]

if __name__ == '__main__':
    cem = CEM(my_func, 2)
    print(cem.eval())
```

### function with other inputs

```python
from cem import CEM

def my_func(a1, a2, a3):
    c = a1 + a2 - a3
    return x[0]*x[0] + x[1]*x[1]

if __name__ == '__main__':
    cem = CEM(my_func, 2)
    a1 = np.array([1, 2])
    a2 = np.array([2, 3])
    print(cem.eval(a1, a2))
```

