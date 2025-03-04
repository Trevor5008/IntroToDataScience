"""
Normal distribution:
- It is symmetric about the mean (the left and right tails are mirror images).
- The mean, median, and mode are all equal (center of symmetry).
- It is unimodal (single peak at the mean).
- It follows the 68-95-99.7 rule for spread (approximate probability within 1, 2, 3 standard deviations of μ)
- It is defined over the entire real line (tails extend to infinity in both directions, approaching zero probability density as $x$ moves far from the mean).
"""

# One Sample Z Test - One-sided
# H0 - Null Hypothesis: the average is 29, and the difference between 29 and 28.3 is small and due to chance
# Ha - Alternative Hypothesis: the average is lower than 29, and the difference is significant

from scipy.stats import norm
import math

n = 50
mu1 = 29
sigma1 = 4

z_score1 = (28.3-mu1)/(sigma1/math.sqrt(n))
print(f"Z-score: {z_score1}")

p_value1 = norm.cdf(z_score1)
print(f"p-value: {p_value1}")

if p_value1 < 0.05:
    print("We can reject the null hypothesis. The same mean is significantly "
          "different from the population mean.")
else:
    print("There's not enough evidence, therefore we cannot reject the null hypothesis."
          " No significant difference between the sample mean and the population mean.")

# One Sample Z Test - Two-sided
# Null hypothesis: the average is 30, and the difference between 30 and 28.3 is small and due to change
# Alternative hypothesis: the true average ACT score of all freshman is difference than 30 (less or greater than 30)

mu2 = 30
sigma2 = 4

z_score2 = (28.3 - mu2)/(sigma2 / math.sqrt(n))
print(f"Z-score: {z_score2}")

p_value2 = 2*norm.cdf(z_score2)
print(f"p-value: {p_value2}")

if p_value2 < 0.05:
    print("We can reject the null hypothesis. The same mean is significantly "
          "different from the population mean.")
else:
    print("There's not enough evidence, therefore we cannot reject the null hypothesis."
          " No significant difference between the sample mean and the population mean.")

# Two Sample Z Test
# Null hypothesis: the difference between 12.9% and 6.7% is likely due to chance
# Alternative hypothesis: the average of the keto is higher than the low calorie diet, indicating that keto may be better for weight loss

obs_diff = 12.9 - 6.7
exp_diff = 0 # assuming that the null hypothesis is true
stdev_keto = 4
stdev_low = 3.7
n3 = 60

standard_error = math.sqrt((stdev_keto**2)/n3+(stdev_low**2)/n3)
z_score3 = (obs_diff+exp_diff)/standard_error
print(f"z_score3: {z_score3}")

p_value3 = 1 - norm.cdf(z_score3)
print(f"p-value: {p_value3}")

if p_value3 < 0.05:
    print("We can reject the null hypothesis.")
else:
    print("We fail to reject the null hypothesis. There is significant difference"
          "between keto and low calorie diet protocols.")

# One-Sample T-Test
"""
Example: One-sample t-test in Python – Suppose the average height of adult 
men in a certain country is known to be $175.3$ cm. We have a sample of 10 
adult men from a particular town, and their heights are recorded. We want to 
test if the mean height in this town is different from $175.3$ cm (two-tailed
test).
"""

import numpy as np
from scipy import stats

one_sample_data = [177.3, 182.7, 169.6, 176.3, 180.3, 179.4, 178.5, 177.2, 181.8, 176.5]

result = stats.ttest_1samp(one_sample_data, 175.3)
print(f"t-statistic: {result.statistic} and p-value: {result.pvalue}")

if result.pvalue < 0.05:
    print("We can reject the null hypothesis.")
else:
    print("We fail to reject the null hypothesis. There is significant difference.")

# Two-Sample T-Test
"""
Example: Two-sample t-test (Welch's) – We will use the popular "tips" dataset which 
contains records of restaurant bills and tips, along with attributes like whether 
the patron was a smoker or not. Let's test if smokers and non-smokers have 
different average tip amounts. This is a two-sample independent t-test 
(each bill is from a different party of customers, and we consider smokers vs 
non-smokers as two groups).
"""
import plotly.express as px
import pandas as pd

df =  px.data.tips()
print(df.head())
print(df.groupby("smoker")["tip"].mean())

fig = px.histogram(df, x="tip", color="smoker", barmode="overlay",
                   histnorm='probability',title="Tip Distribution by Smoking Status")
# fig.show()

smokers_tip = df[df["smoker"]=="Yes"]["tip"]
non_smokers_tip = df[df["smoker"]=="No"]["tip"]

result2 = stats.ttest_ind(smokers_tip, non_smokers_tip, equal_var=False)
print(f"t-statistic: {result2.statistic} and p-value: {result2.pvalue}")

if result2.pvalue < 0.05:
    print("We can reject the null hypothesis.")
else:
    print("We fail to reject the null hypothesis. There is significant difference.")

"""
Example: Paired t-test in Python – Let's simulate a scenario. Suppose we have 
15 patients' blood pressure measured before and after taking a new drug. We want 
to test if the drug has changed blood pressure on average. (A decrease would be a 
negative difference if we do after - before.)
"""

n6 = 15

before = np.random.normal(loc=150, scale=10, size=n6)
after = before - np.random.normal(loc=5, scale=5, size=n6)

diff = after - before

print(f"Mean difference = {diff.mean():.2f} (after - before).")

result3 = stats.ttest_rel(before, after)
print(f"t-statistic: {result3.statistic} and p-value: {result3.pvalue}")

if result3.pvalue < 0.05:
    print("We can reject the null hypothesis.")
else:
    print("We fail to reject the null hypothesis. There is significant difference.")