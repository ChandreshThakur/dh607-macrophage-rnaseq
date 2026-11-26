# Phase 4 statistical design (pre-specified)

Recorded before inspecting the Phase 4 genome-wide interaction results.

## Experimental units and condition coding

The released sample metadata contain 336 unique RNA-seq profiles from 84 donors, with exactly one profile from every donor in each condition. The observed metadata mapping is:

| Condition | `condition_name` | `IFNg` | `SL1344` | Binary IFNg | Binary Salmonella |
|---|---|---|---|---:|---:|
| A | naive | naive | control | 0 | 0 |
| B | IFNg | primed | control | 1 | 0 |
| C | SL1344 | naive | infected | 0 | 1 |
| D | IFNg_SL1344 | primed | infected | 1 | 1 |

The biological replicate is the donor. Sample IDs identify measurements, not independent replicates. Donor is categorical. The model matrix is defined as an intercept, 83 treatment-coded donor indicators, binary IFNg, binary Salmonella, and their product. It therefore has 87 columns. Its rank and coefficient names must be checked programmatically before any fit.

The primary interaction estimand is

`I = D - B - C + A = (D - B) - (C - A)`.

It asks whether the Salmonella response after IFN-gamma priming differs from the Salmonella response without priming. Positive values mean that the combined response is more positive on the stated scale than additivity predicts; negative values mean it is less positive. A negative interaction does not necessarily mean that expression in D is below expression in A, B, or C.

## Part A: primary course analysis

Raw integer counts are normalized once by the Phase 3 DESeq-style median-of-ratios factors. Genes with count >=10 in at least 84 samples form the primary testing family. The analyzed measurement is `log2(normalized count + 0.5)`.

For every retained gene, the four measurements are aligned within each donor and the 84 donor interaction contrasts are calculated. The output reports the mean interaction, standard deviation, standard error, two-sided one-sample t statistic, 95% t confidence interval, raw P value, and Benjamini-Hochberg adjusted P value across the complete genome-wide testing family. No donor is silently excluded.

Diagnostics include donor-level distributions, maximum studentized donor deviations, sign consistency, and leave-one-donor-out effects for the pre-specified panel and representative genome-wide extremes. These are sensitivity analyses, not independent validation.

## Part B: count-aware supporting analysis

The supporting model uses unmodified integer counts and the formula

`~ donor + IFNg_binary + Salmonella_binary + IFNg_binary:Salmonella_binary`.

The reference state is one named donor, IFNg=0, Salmonella=0. The tested coefficient must be the product term and must have the same `D-B-C+A` sign as Part A. Before a full fit, the implementation must pass (1) a synthetic known-interaction check and (2) a balanced real-data pilot. Package version, coefficient names, convergence status, warnings, Cook's filtering, and any independent filtering must be saved. CQN values must never enter this negative-binomial model.

The count-aware model is complementary: its effect estimator, dispersion model, outlier handling, and standard errors differ from the donor-level t analysis. Agreement is assessed using effect correlation, direction agreement, and overlap of FDR classifications; the methods are not described as mathematically identical.

## Sensitivity analyses

- Authors' continuous CQN matrix, using the same within-donor `D-B-C+A` t analysis. Its numerical values are treated as an author-normalized log-like expression scale, not counts.
- Alternate prevalence filters: count >=5 in >=42 samples and count >=20 in >=84 samples, while preserving the primary family.
- Effect concordance and FDR classifications across normalizations and filters.
- Leave-one-donor-out effects for the frozen candidate panel and selected positive, negative, and ambiguous genome-wide examples.
- Candidate results reported both with genome-wide BH values and a clearly separate candidate-panel BH adjustment.

## Technical structure and limitations

Library pool, chemistry, RNA automation, replicate selection, and sex are exactly balanced across the four conditions within their categories. They therefore do not confound the within-donor factorial contrasts. Adding donor-constant technical variables to a donor-fixed-effect model would be redundant. Date variables are not added to the primary model because the paired four-condition design already blocks donor and treatment is balanced within the recorded library strata.

Median-ratio size factors are condition-associated, so absolute interaction magnitudes on that scale may be composition-sensitive. CQN sensitivity can establish robustness to another normalization, but neither normalization identifies absolute RNA molecules per cell without external spike-ins.

## Decision thresholds

Genome-wide statistical significance is BH FDR <0.05. Effect size, confidence interval, donor consistency, normalization sensitivity, expression adequacy, and biological plausibility are reported alongside significance. No minimum effect threshold is used to define the primary testing family after seeing results.

