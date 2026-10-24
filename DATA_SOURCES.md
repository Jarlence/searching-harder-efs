# Data sources and attribution

The eight additional benchmark sources are third-party UCI Machine Learning Repository data sets. Their official pages were checked on **2026-10-24** and each identifies its data as licensed under **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**. The repository account is not the creator of these data sets. Upstream creators are credited below, with the dataset DOI and source page as the authoritative attribution record.

The supplied processed matrices are derivatives prepared for this study. They are not new UCI releases. Preserve the source credit, DOI, license link, and the description of changes when redistributing them. These dataset licenses do not license the project software or other repository content; see [RIGHTS.md](RIGHTS.md).

## Eight additional data sets

Citation years below follow the UCI recommended dataset citations; they need not equal the page's donation date. Matrix dimensions are the study's frozen processed dimensions, not a claim about the complete upstream collection.

| Study ID | Official dataset and source | Upstream creator attribution | Dataset DOI | Processed matrix |
|---|---|---|---|---|
| `nc_climate` | [Climate Model Simulation Crashes, UCI 252](https://archive.ics.uci.edu/dataset/252/climate+model+simulation+crashes) | D. Lucas and co-creators listed in the UCI record (2013); the source lists R. Klein, J. Tannahill, D. Ivanova, S. Brandon, D. Domyancic, and Y. Zhang as well as two abbreviated `D. D.` entries | [10.24432/C5HG71](https://doi.org/10.24432/C5HG71) | 540 rows, 18 predictors |
| `nc_credit` | [Statlog (German Credit Data), UCI 144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | Hans Hofmann (1994) | [10.24432/C5NC77](https://doi.org/10.24432/C5NC77) | 1,000 rows, 24 predictors |
| `nc_dbworld` | [DBWorld e-mails, UCI 219](https://archive.ics.uci.edu/dataset/219/dbworld+e+mails) | Michele Filannino (2011) | [10.24432/C5589M](https://doi.org/10.24432/C5589M) | 62 rows, 4,702 predictors |
| `nc_dexter` | [Dexter, UCI 168](https://archive.ics.uci.edu/dataset/168/dexter) | Isabelle Guyon, Steve Gunn, Asa Ben-Hur, and Gideon Dror (2004) | [10.24432/C5P898](https://doi.org/10.24432/C5P898) | 600 rows, 20,000 predictors |
| `nc_gisette` | [Gisette, UCI 170](https://archive.ics.uci.edu/dataset/170/gisette) | Isabelle Guyon, Steve Gunn, Asa Ben-Hur, and Gideon Dror (2004) | [10.24432/C5HP5B](https://doi.org/10.24432/C5HP5B) | 1,200 rows, 5,000 predictors |
| `nc_ionosphere` | [Ionosphere, UCI 52](https://archive.ics.uci.edu/dataset/52/ionosphere) | V. Sigillito, S. Wing, L. Hutton, and K. Baker (1989) | [10.24432/C5W01B](https://doi.org/10.24432/C5W01B) | 350 rows, 34 predictors |
| `nc_qsar` | [QSAR biodegradation, UCI 254](https://archive.ics.uci.edu/dataset/254/qsar+biodegradation) | Kamel Mansouri, Tine Ringsted, Davide Ballabio, Roberto Todeschini, and Viviana Consonni (2013) | [10.24432/C5H60M](https://doi.org/10.24432/C5H60M) | 1,052 rows, 41 predictors |
| `nc_spambase` | [Spambase, UCI 94](https://archive.ics.uci.edu/dataset/94/spambase) | Mark Hopkins, Erik Reeber, George Forman, and Jaap Suermondt (1999) | [10.24432/C53G6X](https://doi.org/10.24432/C53G6X) | 1,200 rows, 57 predictors |

The abbreviated entries on the Climate source page are retained as an upstream metadata ambiguity. No identities have been inferred or invented. The linked record supplies the complete current recommended citation.

## Copies and transformations included in the study

After running `restore_materials.py`, the X11c collection provides the following original project-relative paths:

- `docs/NEW_DATA_CONFIRMATION_PLAN.md`: source selection, representations, deterministic cohort construction, and interpretation boundaries.
- `data/new_confirmation/registry.json`: source IDs, dimensions, label mappings, selected source indices, provenance, and raw/processed checksums.
- `data/new_confirmation/raw/nc_*.zip` and `nc_*.access.json`: downloaded upstream archives and access records.
- `data/new_confirmation/processed/nc_*.npz` and `nc_*.json`: processed numerical arrays and their data cards.
- `extras/confirmation_data.py`: deterministic parsing and cohort-construction implementation.

The plan is a preserved historical protocol. Its initial proposed public-registration requirement is not evidence that external registration occurred before execution. The repository README describes the actual release status.

The processing preserves published numerical predictor representations, stores values as contiguous float64, and canonicalizes negative zero. It retains the first source row for each exactly equal predictor vector in source order, independently of the label. The archived registry records conflicting-label groups and retained source identities. This is feature-vector deduplication, not a claim that equal vectors necessarily represent the same physical measurement. Cohorts exceeding 1,200 feature-unique rows use the frozen stratified sampling rule and salt `X11c-fixed-cohort-20261014-v1`. Selected rows retain ascending source order. Binary labels are encoded using the recorded mapping. Learned vocabulary construction, imputation, and new supervised feature screening are not part of this adapter.

| Study ID | Published representation and study-specific changes |
|---|---|
| `nc_climate` | `pop_failures.dat`; exclude the prespecified study/simulation identifier columns and retain the 18 simulation parameters. |
| `nc_credit` | Use the official `german.data-numeric` file with 24 predictors, rather than reconstructing the categorical representation. |
| `nc_dbworld` | Use the official unstemmed email-body binary bag-of-words matrix `WEKA/dbworld_bodies.arff`; first-row retention removes two repeated feature vectors from the 64-row collection. |
| `nc_dexter` | Concatenate the publicly labelled training and validation sets in that order; losslessly expand sparse indices to the 20,000-column numerical matrix. Exclude the unlabelled test set. |
| `nc_gisette` | Concatenate the publicly labelled training and validation sets, then apply the fixed 1,200-row cohort selection to the 7,000 labelled rows. Exclude the unlabelled test set. |
| `nc_ionosphere` | Use the official numerical radar predictors, retaining the published constant column; first-row retention reduces 351 rows to 350. |
| `nc_qsar` | Use the fixed descriptors in `biodeg.csv`; first-row retention reduces 1,055 rows to 1,052. |
| `nc_spambase` | Use the official 57 numerical email predictors; first-row retention reduces 4,601 rows to 4,207 feature-unique rows, followed by the fixed 1,200-row cohort selection. |

These transformations are documented as changes made for this study. The source authors are not represented as endorsing this study or its derived cohorts. The frozen plan and registry are authoritative for exact row selection and class encoding; this summary does not replace them.

## Integrity of the data copies

The following SHA-256 values come from the frozen X11c registry. The raw hash identifies the upstream ZIP retained by the study; the processed hash identifies the frozen numerical artifact recorded there.

| Study ID | Raw archive SHA-256 | Processed SHA-256 |
|---|---|---|
| `nc_climate` | `fe8c0c2b79d1beedf31fbca2db8a97cec3c1b773ee29ed3357a2ff1dc8dd12d0` | `f970973531ae7a837cb7443e6741d97cfafb27360044b4b070ac40dd44ae591c` |
| `nc_credit` | `e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b` | `60f675848419938f2035d80dc724479c3c0acf0e993221502aaf421ab83607aa` |
| `nc_dbworld` | `8fff11481345c8e0682800b4bcbdc7031028fec06a6c89bf723e09cab2395151` | `810f09236f44b1657a0020b6d886446a67311fa6639b89ff9803e261b4659bcf` |
| `nc_dexter` | `70400df292178d8d3627411a5631a4cb13fb6b19da4a200b9a206f25255cddcc` | `0570688f33647f56af88ae9830cad95268e1308559f3edb056eefa5931a4609b` |
| `nc_gisette` | `430bb631b3e494659f13763c117f06f5aa8a269b0d8e13ccf5b71d95303f3808` | `c868e8ad314996abb51a747448dc9197708dae5883384b91d98b18071659a51d` |
| `nc_ionosphere` | `4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860` | `7d8f8e557c7ac5280c836e4f5521e57db25433987a318630400d000c789aba74` |
| `nc_qsar` | `b5f7cf1b05564d94d65265a9a4aeae5fffcd15b8d3bcafe4225a8827d7ae08ac` | `0cb17acfcd8973f9ff28dff9e0625ae51d36be0d01d79b2062256620c3084c66` |
| `nc_spambase` | `813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636` | `e01d89fd0e8f392086471260878093d6bcce37a302999848892cbd25098fa6d7` |

## Original benchmark sources

R22 contains `data/data_cards.json` and acquisition/verification scripts, including `experiments/get_data.py` and `experiments/verify_data.py`. Its original third-party raw/processed datasets are not redistributed in this release. Retrieve them from the recorded upstream sources and observe those sources' applicable terms. This release check did not independently re-verify the licenses of all original thirteen sources; the eight-source CC BY 4.0 finding must not be extended to them.

