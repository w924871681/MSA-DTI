# Bitbrains experiment figures 6–8

From any working directory, run `python /path/to/plot_bitbrains.py OUTPUT_DIR`. The script writes the three manuscript PDFs into that directory. All numerical inputs are read directly from the four files in `data/`. They are byte-identical to the corresponding files under `code/data/` in the public v1.1 tag.

| Figure | Function | Frozen inputs |
| --- | --- | --- |
| 6, paired task MSE | `build_fig6` | `deployment_events_720.csv.gz`, `primary_task_scores.csv` |
| 7, robustness and intervals | `build_fig7_forest` | `deployment_events_720.csv.gz`, `primary_task_scores.csv` |
| 8, feasibility and selection frequencies | `build_fig8_budget_feasibility` | `budget_resource_profiles.json`, `selection_decisions.csv.gz` |

## Origin

The functions, fonts and frozen inputs come from [MSA-DTI v1.1 at commit 1438a1d](https://github.com/w924871681/MSA-DTI/tree/1438a1d4ca245f05eda1ef04f67b69b714deebbe). The original figures are drawn by `code/plotting/generate_paper_assets.py` (`build_fig6`, `build_fig7_forest`, `build_fig8_budget_feasibility`). This standalone excerpt retains the original data selection, aggregation, bootstrap, drawing and panel layout. Its command-line wrapper supplies the destination and writes the PDFs. The project license is included in `LICENSE`. Tinos fonts and their license notices are in `fonts/`.

Figure 6 uses original errors for the win counts and clips tiny values only for display. Figure 7 uses a paired cluster bootstrap over VMs with 4,000 draws and seed 20261123. The two panels use PT-FT and Full-Val, respectively. Figure 8 reads the frozen H=1 resource profiles and the 360 deployment selections for Bitbrains E.

## Rendering

The published tag lists Matplotlib 3.10.0, pandas 2.2.3, NumPy 2.1.3 and fonttools 4.55.3. The three PDFs in the manuscript record Matplotlib 3.10.0. This excerpt was also run with Matplotlib 3.10.8. Page sizes, all extracted figure labels and reported data agree with the manuscript PDF assets. Rendering can differ in font spacing, rasterization and PDF encoding across environments. Use the manuscript reference PDFs if an exact copy of a submitted file is required.
