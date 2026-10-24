# Searching Harder Looks Better

Code and research materials for *Searching Harder Looks Better: A Budget-Resolved Study of Evaluation Inflation in Evolutionary Feature Selection*.

This repository contains the experimental records and analysis inputs used to study the predictive return on additional feature-selection search and the difference between an algorithm's inner and outer advantage over random search. It is a private preparation repository. A public release, software license, and permanent archival DOI have not yet been finalized.

## Research materials

| Collection | Completed jobs used in the analyses | Purpose |
|---|---:|---|
| R22 | 43,710 | Original benchmark, auxiliary controls, and strict inner-fold scaling analysis |
| X11c | 4,000 | Eight additional non-biomedical data sets, five searches, two learners, and three fixed contrasts |
| X12m | 800 | Size-matched random-search follow-up on the additional data sets |
| Total | **48,510** | Collections are analysed according to their own protocols and comparison families |

The additional-benchmark contrasts were fixed internally before execution. The materials do not establish an external preregistration. The estimated effects belong to the specified data sets, search configurations, preprocessing protocols, and random references.

## Start here

Download the complete repository through GitHub's **Code > Download ZIP**, then extract it. Use Python 3.9 or later for the standard-library integrity and restoration utility:

```sh
python restore_materials.py --verify-only
python restore_materials.py --output restored
```

The second command reconstructs and verifies the scientific archives, extracts the collections separately under `restored/R22`, `restored/X11c`, and `restored/X12m`, and assembles `restored/project` in the documented R22-to-X11c-to-X12m order. It restores saved material; it does not launch performance experiments.

See [REPRODUCE.md](REPRODUCE.md) for analysis entry points and environment requirements, [DATA_SOURCES.md](DATA_SOURCES.md) for data attribution, and [RIGHTS.md](RIGHTS.md) for the current licensing status.

## Repository layout

- `R22_science.zip.part*`, `X11c_science.zip.part*`, and `X12m_science.zip`: cleaned scientific collections, with unchanged bytes for each retained source member. Large archives are split to accommodate the browser upload limit.
- `integrity_manifest.json`: source-archive identities, retained-member hashes, exclusions, collection counts, and distribution-file checksums.
- `restore_materials.py`: verifies the supplied files and restores the original project-relative layout.
- `source_tree.zip`: a compact source and methods snapshot with its original directory structure.
- `src/efsbench/`, `analysis/`, `experiments/`, `extras/`, and `tests/`: browsable copies of frozen implementation, analysis, orchestration, and test files. Refer to the collection-specific source and freeze records for the version used in each analysis.

Raw result records, protocol freezes, analysis outputs, data provenance, and environment records are retained in the scientific collections. Draft manuscripts, submission letters, word-processing verification artifacts, private human-coding sheets, and third-party full-text articles are outside this repository's distribution scope. An exclusion inventory records the files removed from the original local delivery packages. Archived documentation can describe its own earlier collection; this page identifies the combined repository.

## Interpretation

The paper distinguishes three quantities: the gap between inner and outer gains when the budget increases, the paired difference between inner and outer algorithm advantages, and the outer-performance advantage itself. Positive advantage discrepancy does not by itself establish worse outer performance or a ranking reversal. Request-matched, distinct-mask, and size-matched random search define different comparisons.

The original and additional benchmarks remain separate in the reported analyses. The same-size follow-up retains its own exploratory comparison family. The repository preserves uncertain results as well as supported effects.

## Publication status

This repository is being prepared for a submission to Applied Soft Computing. Repository creation is not journal submission, acceptance, public preregistration, or permanent DOI archiving. The current manuscript and submission metadata are maintained separately while author and declaration information is finalized.
