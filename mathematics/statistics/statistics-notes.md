# Statistics for Machine Learning

## 1. Why Statistics Matters in ML

Statistics helps us understand data, summarize its behavior, estimate unknown quantities, compare groups/models, and decide whether an observed result may be meaningful or just random variation.

ML connections:
- EDA: understand feature distributions and outliers.
- Preprocessing: detect skew and scale differences.
- Evaluation: interpret variation in model scores.
- Experiments: compare model or product changes carefully.

## 2. Mean, Median, and Mode

For values \(x_1,\ldots,x_n\):

### Mean
\[
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
\]

Mean uses every value, so extreme values can pull it strongly.

### Median
Sort the values and take the middle value. If there are an even number of values, average the two middle values.

Median is more robust to outliers.

### Mode
The most frequently occurring value. A dataset may have more than one mode or no uniquely useful mode.

Example: data = [2, 3, 3, 4, 100]
- Mean = 22.4
- Median = 3
- Mode = 3

The outlier 100 pulls the mean upward, while the median remains representative of the central values.

```python
import numpy as np

x = np.array([2, 3, 3, 4, 100])
print("Mean:", np.mean(x))       # 22.4
print("Median:", np.median(x))   # 3.0
values, counts = np.unique(x, return_counts=True)
print("Mode:", values[np.argmax(counts)])  # 3
```

## 3. Variance and Standard Deviation

Variance measures spread around the mean; standard deviation is its square root and is expressed in the same units as the data.

For a population:
\[
\sigma^2=\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2
\]

For a sample, the common unbiased variance estimator uses \(n-1\):
\[
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2
\]

\[
\text{Standard deviation}=\sqrt{\text{variance}}
\]

NumPy defaults to population variance (`ddof=0`). Use `ddof=1` for sample variance.

```python
x = np.array([2, 4, 6, 8])
print(np.var(x))          # population variance: 5.0
print(np.var(x, ddof=1))  # sample variance: 6.666...
print(np.std(x))          # population SD: about 2.236
```

Higher variance/SD means more spread, not necessarily worse data. Variance is in squared units; SD returns to the original units.

## 4. Percentiles and Outliers

A percentile tells you the value below which a given percentage of observations falls. The median is the 50th percentile.

The interquartile range (IQR) is:
\[
IQR=Q_3-Q_1
\]

A common boxplot rule flags values below \(Q_1-1.5IQR\) or above \(Q_3+1.5IQR\) as potential outliers. These are flags for investigation, not automatic proof that data is wrong.

```python
q1, q3 = np.percentile(x, [25, 75])
iqr = q3 - q1
```

## 5. Covariance and Correlation

Covariance describes whether two variables tend to move together. Its magnitude depends on the units, so it is hard to compare across datasets.

Pearson correlation standardizes the relationship:
\[
r=\frac{\operatorname{cov}(X,Y)}{\sigma_X\sigma_Y}
\]

- \(r\) is between -1 and +1.
- Near +1: strong positive linear association.
- Near -1: strong negative linear association.
- Near 0: little or no linear association (a nonlinear relationship may still exist).

Correlation does not prove causation. Outliers can strongly affect Pearson correlation, and correlation can miss nonlinear patterns.

```python
hours = np.array([1, 2, 3, 4, 5])
scores = np.array([50, 55, 65, 70, 80])
print(np.corrcoef(hours, scores)[0, 1])
```

In a DataFrame:
```python
# df.corr(numeric_only=True)
```

## 6. Population, Sample, and Sampling

- **Population:** the full group you want to understand.
- **Sample:** the subset actually observed.
- **Sampling:** how that subset is selected.

A sample should represent the population. Biased sampling can produce misleading estimates even with a large sample. Random sampling helps, but does not automatically remove every source of bias.

In ML, train/validation/test splits estimate performance on unseen examples. Data leakage or a split that does not match the real deployment setting can make evaluation overoptimistic.

## 7. Estimation and Uncertainty

An estimate uses sample data to infer a population quantity.

- **Point estimate:** one number, such as sample mean.
- **Interval estimate:** a range intended to express uncertainty, such as a confidence interval.

A 95% confidence interval is often described informally as a plausible range for the population parameter. In the frequentist interpretation, across repeated samples, 95% of intervals constructed by the procedure would contain the true parameter. It does not mean that 95% of the observed data lies inside the interval.

Larger representative samples generally reduce sampling uncertainty, but do not fix systematic bias.

## 8. Hypothesis Testing — Practical Level

Hypothesis testing helps assess whether sample evidence is inconsistent with a specified null hypothesis.

- **Null hypothesis \(H_0\):** a baseline claim, often “no difference.”
- **Alternative hypothesis \(H_1\):** the difference/effect being investigated.
- **Significance level \(\alpha\):** chosen threshold, often 0.05.
- **p-value:** assuming \(H_0\) and the test assumptions are true, the probability of results at least as extreme as those observed.

Decision rule (simple version):
- If \(p \leq \alpha\), reject \(H_0\).
- If \(p > \alpha\), fail to reject \(H_0\); this does not prove \(H_0\) true.

A p-value is **not** the probability that \(H_0\) is true, and statistical significance does not guarantee practical importance.

### Errors and power
- **Type I error:** reject a true null hypothesis (false positive).
- **Type II error:** fail to reject a false null hypothesis (false negative).
- **Power:** probability of detecting a specified effect when it exists.

Practical checks:
1. Define the question and hypotheses before inspecting results.
2. Choose a suitable test for the data and design.
3. Check assumptions and sample independence.
4. Report effect size and uncertainty, not only the p-value.
5. Avoid repeatedly testing until a small p-value appears.

## 9. Statistics in ML

- Use mean/median and distributions to understand features.
- Use variance/SD to understand spread and scaling.
- Use correlation as a diagnostic, not proof of causation or a guarantee of predictive value.
- Use representative samples and correct splits to estimate generalization.
- Use uncertainty and careful comparisons when comparing model performance.
- Statistical significance and real-world usefulness are different questions.

## 10. NumPy Practice

```python
import numpy as np

scores = np.array([60, 65, 70, 75, 100])

print("Mean:", scores.mean())
print("Median:", np.median(scores))
print("Population variance:", scores.var())
print("Population SD:", scores.std())
print("Sample variance:", scores.var(ddof=1))
print("25th, 50th, 75th percentiles:",
      np.percentile(scores, [25, 50, 75]))

hours = np.array([1, 2, 3, 4, 5])
marks = np.array([50, 55, 65, 70, 80])
print("Correlation:", np.corrcoef(hours, marks)[0, 1])
```

## 11. Common Mistakes

1. Using mean automatically when data has strong outliers; compare with median.
2. Confusing population variance (`ddof=0`) with sample variance (`ddof=1`).
3. Interpreting correlation as causation.
4. Assuming zero correlation means no relationship of any kind.
5. Assuming a large sample removes selection bias.
6. Saying a p-value is the probability that the null hypothesis is true.
7. Treating “not statistically significant” as proof that there is no effect.
8. Reporting statistical significance without effect size or practical context.
9. Treating every flagged outlier as an error to delete.

## Quick Formula Reference

| Concept | Formula / interpretation |
|---|---|
| Mean | \(\bar{x}=\sum x_i/n\) |
| Population variance | \(\sigma^2=\sum(x_i-\mu)^2/N\) |
| Sample variance | \(s^2=\sum(x_i-\bar{x})^2/(n-1)\) |
| Standard deviation | Square root of variance |
| IQR | \(Q_3-Q_1\) |
| Pearson correlation | \(\operatorname{cov}(X,Y)/(\sigma_X\sigma_Y)\) |
| Type I error | Reject a true null |
| Type II error | Fail to reject a false null |
| p-value | Probability of equally/more extreme data, assuming \(H_0\) and assumptions |
