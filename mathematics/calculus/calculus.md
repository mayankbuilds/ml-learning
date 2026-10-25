# Calculus for Machine Learning

## 1. Functions

A function maps an input to an output.

$$
f(x)=x^2
$$

Examples: $f(2)=4$, $f(3)=9$, and $f(-2)=4$.

In ML, a model maps input features to a prediction:

$$
\hat y=f(x)
$$

Model parameters are adjusted during training to improve predictions.

## 2. Derivatives: rate of change

A derivative tells us how quickly a function's output changes as its input changes.

$$
f(x)=x^2 \qquad \frac{df}{dx}=2x
$$

At $x=3$, $f'(3)=6$. For a small input change $\Delta x$, the output change is approximately $6\Delta x$ near that point.

### Basic derivative rules

| Function | Derivative |
|---|---|
| $f(x)=c$ | $0$ |
| $f(x)=x$ | $1$ |
| $f(x)=x^2$ | $2x$ |
| $f(x)=x^3$ | $3x^2$ |
| $f(x)=ax+b$ | $a$ |
| $f(x)=ax^n$ | $anx^{n-1}$ |

Power rule:

$$
\frac{d}{dx}x^n=nx^{n-1}
$$

A positive derivative means the function is increasing locally; a negative derivative means it is decreasing locally. A zero derivative means the local slope is zero, but the point is not necessarily a minimum.

## 3. Partial derivatives

When a function depends on multiple variables, a partial derivative measures change with respect to one variable while holding the other variables fixed.

For:

$$
f(w,b)=w^2+3b^2
$$

$$
\frac{\partial f}{\partial w}=2w,\qquad
\frac{\partial f}{\partial b}=6b
$$

In a simple linear model, $w$ may be a weight and $b$ a bias. Partial derivatives tell us how a function changes when each parameter changes.

## 4. Chain rule

The chain rule differentiates a composition of functions. If $u=g(x)$ and $y=f(u)$, then:

$$
\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}
$$

Example:

$$
y=(3x+2)^2
$$

Let $u=3x+2$, so $y=u^2$.

$$
\frac{dy}{dx}=2u\cdot3=6(3x+2)
$$

**ML connection:** a model produces a prediction, and a loss is calculated from that prediction. The chain rule helps compute how each parameter affects the final loss. Neural networks use this principle in backpropagation.

## 5. Common mistakes

- Applying the power rule incorrectly: $\frac{d}{dx}x^n=nx^{n-1}$.
- Forgetting to multiply by the inner derivative when using the chain rule.
- Assuming a zero derivative always means a minimum.
- Confusing a derivative (rate of change) with the function's value.

## 6. Quick formula sheet

$$
\frac{d}{dx}x^n=nx^{n-1}
$$

$$
\frac{d}{dx}(ax+b)=a
$$

$$
\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}
$$

## 7. Quick practice

1. If $f(x)=x^3$, find $f'(x)$.
2. If $f(x)=5x^2$, find $f'(x)$.
3. If $y=(2x+1)^2$, find $dy/dx$.
4. For $g(w,b)=w^2+4b^2$, find both partial derivatives.

## 8. ML connection summary

- Derivatives measure how a function changes as an input changes.
- Partial derivatives measure change with respect to one parameter at a time.
- The chain rule handles nested computations and supports backpropagation.
