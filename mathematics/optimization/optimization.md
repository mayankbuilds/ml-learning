# Optimization for Machine Learning

## 1. What is optimization?

Optimization means finding parameter values that minimize or maximize an objective. In ML training, we usually try to minimize a loss function.

## 2. Loss function

A loss function measures how wrong a model's prediction is. Consider:

$$
L(w)=(w-3)^2
$$

Here $w$ is a model parameter. The loss is zero at $w=3$.

| $w$ | $L(w)$ |
|---:|---:|
| 0 | 9 |
| 1 | 4 |
| 2 | 1 |
| 3 | 0 |
| 4 | 1 |
| 5 | 4 |

The objective is to find parameter values that produce low loss.

## 3. Gradient

For functions with multiple parameters, the gradient is the vector of partial derivatives. It points in the direction of the steepest local increase of a differentiable function.

For:

$$
L(w,b)=w^2+3b^2
$$

$$
\nabla L(w,b)=
\begin{bmatrix}
2w\\
6b
\end{bmatrix}
$$

At $w=2, b=1$:

$$
\nabla L(2,1)=
\begin{bmatrix}
4\\
6
\end{bmatrix}
$$

To reduce loss using gradient descent, move in the opposite direction to the gradient.

## 4. Gradient descent

Gradient descent repeatedly updates model parameters to reduce loss.

For one parameter $w$:

$$
w_{\text{new}}=w_{\text{old}}-\eta\frac{dL}{dw}
$$

where:
- $w$ is the parameter;
- $\eta$ (eta) is the learning rate;
- $\frac{dL}{dw}$ is the derivative of the loss.

For $L(w)=(w-3)^2$:

$$
\frac{dL}{dw}=2(w-3)
$$

Start with $w=0$ and learning rate $\eta=0.1$:

$$
L'(0)=2(0-3)=-6
$$

$$
w_{\text{new}}=0-0.1(-6)=0.6
$$

Loss decreases from $L(0)=9$ to $L(0.6)=5.76$. Repeating the update moves $w$ closer to 3.

## 5. Learning rate

The learning rate controls the size of each update.

- **Too small:** updates may be very slow.
- **Reasonable:** can make steady progress toward a minimum.
- **Too large:** may overshoot, oscillate, or diverge.

The best learning rate depends on the problem and optimization method.

## 6. NumPy implementation

```python
def loss(w):
    return (w - 3) ** 2

def gradient(w):
    return 2 * (w - 3)

w = 0.0
learning_rate = 0.1

for step in range(20):
    grad = gradient(w)
    w = w - learning_rate * grad
    print(f"Step {step + 1:2d} | w={w:.4f} | loss={loss(w):.6f}")

print("Final parameter:", w)
print("Final loss:", loss(w))
```

### Practice

1. Change the learning rate to `0.01`; observe slower convergence.
2. Try `0.9`; observe how updates behave.
3. Change the target from `3` to `-2`, updating both loss and gradient consistently.

## 7. Numerical derivative check

A numerical approximation to a derivative is:

$$
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}
$$

```python
def f(x):
    return x**2

x = 3.0
h = 1e-5

numerical = (f(x + h) - f(x - h)) / (2 * h)
analytical = 2 * x

print(numerical)   # approximately 6
print(analytical)  # 6
```

This helps check whether a manually derived derivative or gradient implementation is plausible.

## 8. Common mistakes

- Forgetting the minus sign in the gradient-descent update.
- Confusing the loss (a value) with the gradient (a vector of derivatives).
- Choosing an excessively large learning rate.
- Assuming gradient descent always finds the global minimum for every possible loss function.

## 9. Quick formula sheet

$$
\nabla L=
\begin{bmatrix}
\partial L/\partial w_1\\
\partial L/\partial w_2\\
\vdots
\end{bmatrix}
$$

$$
\theta_{\text{new}}=\theta_{\text{old}}-\eta\nabla L(\theta)
$$

## 10. ML connection summary

- Loss functions quantify prediction error or the training objective.
- Gradients indicate how loss changes with model parameters.
- Gradient descent updates parameters to reduce the objective.
- Learning rate controls the size of each update.
