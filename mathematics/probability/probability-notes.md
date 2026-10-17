# Probability for Machine Learning

## 1. Probability Basics

Probability measures how likely an event is to occur. For equally likely outcomes:

\[
P(A)=\frac{\text{Favorable outcomes}}{\text{Total possible outcomes}}
\]

\[
0 \leq P(A) \leq 1
\]

- \(P(A)=0\): impossible event
- \(P(A)=1\): certain event

### Complement Rule

\[
P(A^c)=1-P(A)
\]

If \(P(A)=0.7\), then \(P(A^c)=0.3\).

## 2. Conditional Probability

Conditional probability measures the probability of event \(A\) when event \(B\) is already known to have occurred.

\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)}, \qquad P(B)>0
\]

Example: 100 emails are examined. 40 contain “free prize”; 30 of those 40 are spam.

\[
P(\text{Spam}\mid\text{Phrase})=\frac{30}{40}=0.75
\]

The probability is 75%. The denominator is 40 because only emails containing the phrase are considered.

```python
spam_and_phrase = 30
emails_with_phrase = 40
print(spam_and_phrase / emails_with_phrase)  # 0.75
```

## 3. Independent and Dependent Events

Events \(A\) and \(B\) are independent when knowing one occurred does not change the probability of the other.

\[
P(A\mid B)=P(A)
\]

Equivalently:

\[
P(A\cap B)=P(A)P(B)
\]

Separate tosses of a fair coin are independent. Drawing cards without replacement creates dependent events.

**Important:** Association or correlation does not, by itself, prove causation.

## 4. Bayes' Theorem

Bayes' theorem updates a probability after new evidence is observed.

\[
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}
\]

- **Prior** \(P(A)\): probability before evidence
- **Likelihood** \(P(B\mid A)\): probability of evidence if \(A\) is true
- **Evidence** \(P(B)\): overall probability of observing the evidence
- **Posterior** \(P(A\mid B)\): updated probability after evidence

For a disease \(D\) and positive test \(+\):

\[
P(D\mid +)=\frac{P(+\mid D)P(D)}{P(+)}
\]

If a test can be positive without disease:

\[
P(+)=P(+\mid D)P(D)+P(+\mid\neg D)P(\neg D)
\]

Example: \(P(D)=0.01\), \(P(+\mid D)=0.90\), and \(P(+\mid\neg D)=0.05\).

\[
P(+)=0.90(0.01)+0.05(0.99)=0.0585
\]

\[
P(D\mid +)=\frac{0.90(0.01)}{0.0585}\approx0.1538
\]

The probability of disease given a positive result is about 15.38%, not 90%. The prior probability and false-positive rate matter.

```python
p_disease = 0.01
p_positive_given_disease = 0.90
p_positive_given_no_disease = 0.05

p_positive = (
    p_positive_given_disease * p_disease
    + p_positive_given_no_disease * (1 - p_disease)
)
posterior = p_positive_given_disease * p_disease / p_positive
print(posterior)  # approximately 0.1538
```

## 5. Random Variables

A random variable maps a random outcome to a numerical value.

Example: Toss a coin three times and let \(X\) be the number of heads.

\[
X\in\{0,1,2,3\}
\]

A probability distribution describes the possible values of a random variable and their probabilities.

- **Discrete:** countable values, such as number of heads or defective products.
- **Continuous:** values across a continuous range, such as height, temperature or time.

## 6. Expected Value

Expected value is the probability-weighted average of a random variable's possible values.

\[
E[X]=\sum_x xP(X=x)
\]

For a fair six-sided die:

\[
E[X]=\frac{1+2+3+4+5+6}{6}=3.5
\]

A single roll cannot be 3.5. Expected value is the theoretical long-run average over repeated rolls.

## 7. Variance and Standard Deviation

Variance measures the expected squared distance from the mean.

\[
\operatorname{Var}(X)=E[(X-E[X])^2]
\]

For a discrete random variable:

\[
\operatorname{Var}(X)=\sum_x P(X=x)(x-E[X])^2
\]

Equivalent identity:

\[
\operatorname{Var}(X)=E[X^2]-(E[X])^2
\]

Standard deviation is the square root of variance:

\[
\sigma=\sqrt{\operatorname{Var}(X)}
\]

Lower variance means values are more concentrated around the mean; higher variance means they are more spread out.

## 8. Bernoulli Distribution

A Bernoulli random variable models one trial with two possible outcomes, conventionally 0 and 1. Let \(p=P(X=1)\).

\[
P(X=1)=p,\qquad P(X=0)=1-p
\]

\[
E[X]=p,\qquad \operatorname{Var}(X)=p(1-p)
\]

Example: a binary label such as spam (1) or not spam (0).

## 9. Binomial Distribution

A binomial random variable counts successes in a fixed number \(n\) of independent Bernoulli trials, each with the same success probability \(p\).

\[
X\sim\operatorname{Binomial}(n,p)
\]

\[
E[X]=np,\qquad \operatorname{Var}(X)=np(1-p)
\]

Example: count spam emails among 10 independent emails with \(p=0.3\).

\[
E[X]=10(0.3)=3
\]

\[
\operatorname{Var}(X)=10(0.3)(0.7)=2.1
\]

Expected count is 3; an individual batch is not guaranteed to contain exactly 3 successes.

**Bernoulli vs. Binomial:** Bernoulli models one binary trial; Binomial counts successes across a fixed number of independent trials with the same success probability.

## 10. Normal Distribution

The normal distribution is a continuous, symmetric, bell-shaped distribution described by:

- \(\mu\): mean, the center
- \(\sigma\): standard deviation, the spread

Approximately:
- 68% of observations fall within \(\mu\pm\sigma\)
- 95% fall within \(\mu\pm2\sigma\)
- 99.7% fall within \(\mu\pm3\sigma\)

These are approximate rules for normally distributed data.

Example: if \(\mu=100\) and \(\sigma=10\), approximately 68% of observations lie between 90 and 110. Not every real-world dataset is normally distributed.

## 11. Probability in Machine Learning

- **Classification:** models may output class probabilities.
- **Naive Bayes:** applies Bayes' theorem using a conditional-independence assumption between features given the class.
- **Data analysis:** distributions describe how values occur.
- **Simulation:** random sampling can approximate theoretical quantities.

### Probability, accuracy and calibration

- **Prediction probability:** estimated probability assigned to a particular outcome.
- **Accuracy:** fraction of predictions that are correct on an evaluated dataset.
- **Calibration:** among predictions assigned a given probability, how often the outcome occurs in practice.

A model outputting 0.9 for one example does not mean it is 90% accurate overall.

## 12. NumPy Practice

```python
import numpy as np

# Conditional probability
print("P(Spam | Phrase):", 30 / 40)

# Bayes' theorem
p_disease = 0.01
p_positive_given_disease = 0.90
p_positive_given_no_disease = 0.05

p_positive = (
    p_positive_given_disease * p_disease
    + p_positive_given_no_disease * (1 - p_disease)
)
print("P(Disease | Positive):",
      p_positive_given_disease * p_disease / p_positive)

rng = np.random.default_rng(42)

# 1,000 experiments, each containing 10 trials with p=0.3.
# Each output is the number of successes in one experiment.
successes = rng.binomial(n=10, p=0.3, size=1000)
print("Sample mean:", successes.mean())      # theoretical mean = 3
print("Sample variance:", successes.var())  # theoretical variance = 2.1

# Samples from a normal distribution with mean 50 and SD 5
measurements = rng.normal(loc=50, scale=5, size=1000)
print("Sample mean:", measurements.mean())  # approximately 50
print("Sample SD:", measurements.std())     # approximately 5
```

Simulation outputs are sample approximations, not exact guarantees. A fixed random seed makes the pseudorandom sequence reproducible.

## 13. Common Mistakes

1. In conditional probability, use the condition's group as the denominator.
2. Do not confuse \(P(A\mid B)\) with \(P(B\mid A)\).
3. Independence is a mathematical condition, not simply a lack of obvious connection.
4. A positive test does not automatically imply a high posterior probability.
5. Expected value is a theoretical long-run average, not a guaranteed outcome.
6. Binomial mean is \(np\); binomial variance is \(np(1-p)\).
7. Predicted probability is not the same as overall model accuracy.
8. Normal-distribution percentage rules apply approximately when data follows a normal distribution.

## Quick Formula Reference

| Concept | Formula |
|---|---|
| Complement | \(P(A^c)=1-P(A)\) |
| Conditional probability | \(P(A\mid B)=P(A\cap B)/P(B)\) |
| Independent events | \(P(A\cap B)=P(A)P(B)\) |
| Bayes' theorem | \(P(A\mid B)=P(B\mid A)P(A)/P(B)\) |
| Expected value | \(E[X]=\sum_x xP(X=x)\) |
| Variance | \(\operatorname{Var}(X)=E[(X-E[X])^2]\) |
| Bernoulli mean | \(p\) |
| Bernoulli variance | \(p(1-p)\) |
| Binomial mean | \(np\) |
| Binomial variance | \(np(1-p)\) |
| Normal 68% interval | Approximately \(\mu\pm\sigma\) |
