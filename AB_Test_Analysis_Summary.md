# A/B Test Analysis Summary

**Student:** Advik Singh<br>
**Registration No.:** RA2411056030023<br>
**Branch:** DS-A

## Decision

Do **not** roll out the new landing page as a conversion improvement. Its estimated
conversion effect is **-0.158 pp** (treatment minus control), with
a two-sided two-proportion z-test p-value of **0.1899**. At
alpha = 0.05, this is not statistically significant.

## Primary result

| Arm | Users | Conversions | Conversion rate |
| --- | ---: | ---: | ---: |
| Control (old page) | 145,274 | 17,489 | 12.039% |
| Treatment (new page) | 145,310 | 17,264 | 11.881% |

- **Effect (new − old):** -0.158 pp (-1.31% relative)
- **95% CI:** [-0.394 pp, +0.078 pp]
- **z statistic / p-value:** -1.311 / 0.1899
- **Bootstrap 95% interval:** [-0.393 pp, +0.080 pp]
- **Sample-ratio-mismatch p-value:** 0.9468

## Data quality

- Raw records: 294,478
- Assignment/page mismatches excluded: 3,893
- Repeated-user records removed: 1
- Final population: 290,584 unique users
- CSV SHA-256: `d56e2accec25e99ac21cb3d76c5df516dd19cc7a77c14c9014f94e1ea1301beb`

## Important scope limitation

The supplied CSV has no price, revenue, margin or cost field. The report title follows
the requested filename, but the valid inference is about **landing-page conversion only**,
not pricing or profitability.

See `AB_Test_Pricing_Report_CORRECTED.pdf` for the complete six-page report and
`AB_Test_Analysis.py` to reproduce it.
