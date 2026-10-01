# Outlier Detection on California Housing

**Objective:** Compare three outlier detection methods on the California Housing data and decide which one a policy team should use before modeling.

## Methodology
- Found and fixed three bugs in an outlier pipeline: a Z-score that used the mean and std instead of the median and MAD, a Tukey multiplier of 1.0 instead of 1.5, and an Isolation Forest contamination of 0.5 instead of 0.05.
- Used an `OutlierDetector` class (modified Z-score, Tukey fences, Isolation Forest) that checks its settings and returns a summary, saved as `outlier_detector.py`.
- Ran the modified Z-score and Tukey on median income (MedInc), and Isolation Forest on all 9 columns, then compared the flagged rows with a Venn diagram.
- Built an interactive explorer with ipywidgets and plotly to change each method's parameter and look through the flagged rows.

## Key Findings
- The modified Z-score flagged 400 rows, Tukey flagged 681, and Isolation Forest flagged 1,032. All three agreed on 322.
- Every row the Z-score flagged was also flagged by Tukey, because Tukey's upper fence is lower on this right-skewed column.
- Isolation Forest flagged many rows the other two missed, since it looks at combinations of columns, not just income.
- I recommended the modified Z-score as the main screen with Isolation Forest as a second check, and reviewing flagged rows instead of deleting them.
