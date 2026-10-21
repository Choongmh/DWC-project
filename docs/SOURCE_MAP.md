# Source map

| Local source | Destination | Role |
|---|---|---|
| `submission_ready/INTERN'S LEVEL_DWC.zip` | `src/`, `tests/`, `models/`, `outputs/` | Final integrated T1–T3 source, frozen model, results and development notes |
| `dwc2026-internlvl_t2proxy/data/raw` | `data/raw` | Nine original input workbooks |
| `dwc2026-internlvl_t2proxy/data/processed` | `data/processed` | Existing cleaned T1 tables and T2 evaluation cache |
| `interns level ppt.pptx` | `presentations/` | Original team slides, preserved |
| `exon_recovered.pbix` | `dashboard/` | Recovered original Power BI report; native open/refresh still to verify |
| `exon.pbix/Report` | `dashboard/report_definition/` | Inspectable report visual definitions |
| `t2 exon.mp4` | `video/` | T2 video; not asserted to be the complete seven-minute submission |
| Question folder and supplied reports | `docs/original/` | Source briefing, hand-off and presentation PDFs |

The recovered dashboard has a hidden Page 1 and visible Final page. Its visual fields refer to `Master Production` and `t3_top10_params`. Cards use maxima; they should not be presented as medians without changing their calculation.

Some slides contain images rather than editable text. The original deck is retained rather than rewritten. The workbook uses source code and final submission notes where older slides disagree.

The source repository's current branches and remote-only files remain unverified. No final full-team video or public YouTube URL has been identified locally. Optional processed reservoir-grid files are not required to reproduce the core pipeline. Generated T3 parquet files can be rebuilt from the source and saved CSVs.

`audit/assembly_manifest.json` records the initial assembled files and hashes. Original local folders were left unchanged.
