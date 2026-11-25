# Figures 5, 9 and 10

The three CSV files contain the frozen numerical inputs used to draw these figures. From the LaTeX project root, run the following command to recreate their vector PDFs in the figures directory.

```sh
python artwork_source/plot_figures.py figures
```

This edition used Matplotlib 3.10.8, NumPy and the Liberation Serif font. The chart text falls back to Times New Roman or DejaVu Serif if Liberation Serif is unavailable. LaTeX compilation uses the included PDFs directly.

To extract the same plotting inputs again, run the following command with the original experiment archive. It requires pandas and writes the CSVs next to the script.

```sh
python artwork_source/extract_values.py /path/to/MSA-DTI-experiment-data-and-scripts.zip
python artwork_source/collect_adaptation_pairs.py /path/to/MSA-DTI-experiment-data-and-scripts.zip
```

The archive used for verification has SHA-256 `4044eb777260fa55967547a9d9a90a1c047d464c369878a11ca7aeb988e235e4`. The source CSV and cohort are recorded for each row. Figure 5 scales all four metrics by 1000 for display. Figures 9 and 10 use the cohort-specific normalized risk defined in Section 5.1 of the paper. The percentage in Figure 9(b) is 100 times one minus the ratio of the full-adaptation risk to the standard fine-tuning risk for the same selector. Figure 10 fixes the MSA-DTI selector. For each subscription or project, the source table first averages its 12 normalized risks within each adaptation setting. Its plotted reduction is 100 times one minus the ratio of full adaptation to standard fine-tuning for that unit. Circles indicate lower risk and triangles indicate higher risk. Boxes give the 25th, 50th and 75th percentiles by linear interpolation. Whiskers extend to the most extreme observations within 1.5 interquartile ranges of the corresponding box edge.

Supplementary_Data_S1.csv in the project root gives the 40 cohort-level normalized risks formerly plotted in the cross-platform comparison. They are the Azure and UFCG entries in normalized_risk.csv arranged by selector and adaptation setting.
