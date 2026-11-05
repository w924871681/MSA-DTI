# Reproduction checks

The published PDFs are the author's review copies. The main manuscript has 17 pages and the supplement has 5. The archive is the editable source used to generate both.

| Check | Evidence |
| --- | --- |
| Main comparison | 5,760 method–budget–task records. Recomputed three metrics from per-window forecasts and independently reconstructed future labels. All architecture IDs matched. Maximum absolute difference across the two datasets and three metrics was below 1.2e-16 when read as CSV doubles. |
| Supplemental and ablation results | Verified 660 source checkpoint digests, 3,240 supplemental choices, 76 latency conditions and 24 target timing conditions. Checked 360 tight and 360 medium Azure search results and all 120 Bitbrains preselection-ablation tasks. |
| Figures and tables | Regenerated the nine experimental PDF figures and 17 LaTeX table files/text fragments from a clean output directory. The two descriptive tables and four fixed illustrations are supplied in the paper archive. |
| Original inputs | Provider fastStorage.zip matched the frozen SHA-256. Rebuilt and checked the source tensors and all five Bitbrains A–E group input carriers. For Azure, checked the released series against the hash of every published target input. The complete 75-shard rescan was outside this package check. |
| New target adaptation | Used the public runner on all eight methods and all three budgets for the first task in each dataset with frozen source weights. All 48 selections matched the published architecture IDs on the test RTX 3060 Laptop GPU. Largest absolute differences across the three metrics were 4.83e-9 (Bitbrains) and 1.45e-7 (Azure) on this hardware and its newer PyTorch runtime. |
| Source training and ablation smoke tests | Prepared source data from the provider ZIP and trained the first Bitbrains architecture for 50 epochs with PyTorch 2.5.1. Its six state tensors differed from the archived checkpoint by at most 6.0e-6. Refit the four medium-budget ablation arms on the first task. The full method had 17 eligible candidates, removing preselection gave 48, removing the deployment filter gave 19, and removing the margin gave 17. |
| Full retraining | The source-training, reference fitting and baseline routines and CLI are provided. A fresh all-method source-to-target run on a second Linux/CUDA machine is outstanding. The replay and the two target smoke tests have different coverage. |

Elapsed times in the manuscript refer to the recorded original equipment and execution schedule. Replay of timestamp records can validate arithmetic; a new device provides its own timings.

