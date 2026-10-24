"""Write results/MANIFEST.json: SHA-256 of every code file, document, result table, figure and the SUMMARY, plus job counts per suite.
Run LAST (after run_all.py and repro_check.py).  The manifest is what you would archive (e.g. on Zenodo) next to the raw job files."""
import glob
import hashlib
import json
import os
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


groups = {
    "code": ["src/efsbench/*.py", "experiments/*.py", "extras/*.py", "extras/complete_r22.ps1", "analysis/*.py", "tests/*.py", "examples/*.py", "audit/*.py"],
    "docs": ["docs/*.md", "docs/*.json", "*.md", "audit/*.md", "novelty/NOVELTY_AUDIT.md", "src/efsbench/README.md"],
    "results": ["results/SUMMARY.md", "results/REPRODUCIBILITY.md", "results/REVISION2_VALIDATION.md", "results/tables/*", "results/figures/*", "results/r22_strict_scaling/*"],
    "audit_data": ["audit/coding_v1.jsonl", "audit/recode/*.txt", "audit/extracts.jsonl", "audit/audit_summary.json",
                   "audit/recheck_overrides.jsonl", "audit/recall_check.jsonl", "audit/recall_broad.jsonl"],
    "environment": ["requirements-lock.txt"],
    "data_cards": ["data/data_cards.json"],
    "closest_work_provenance": ["novelty/external/li2026/provenance.md", "novelty/external/li2026/record_metadata.json", "novelty/external/li2026/extraction_manifest.json"],
    "paper": ["paper/main.tex", "paper/sections/*.tex", "paper/refs.bib", "paper/generated/*",
              "paper/readable/*", "paper/*.py", "paper/*.md",
              "paper/main.pdf", "paper/supplement_audit.tex", "paper/supplement_audit.pdf",
              "paper/supplement_protocols.tex", "paper/supplement_protocols.pdf", "paper/supplement_strict_scaling.tex", "paper/supplement_strict_scaling.pdf", "paper/figures/*.pdf", "paper/source_assets/*.pdf"],
    # Preparation inputs only. The release manifest binds completed ZIPs separately
    # to avoid hashing a release into itself or including private author metadata.
    "submission_preparation": ["submission/asc/build_materials.py", "submission/asc/build_materials_r22.py", "submission/asc/build_manuscript_source.py",
                               "submission/asc/make_submission_assets.py", "submission/asc/check_release.py", "submission/asc/check_release_r22.py", "submission/asc/README.md",
                               "submission/asc/highlights.txt", "submission/asc/cover_letter.md",
                               "submission/asc/journal_requirements.md", "submission/asc/graphical_abstract_caption.txt",
                               "submission/asc/graphical_abstract_provenance.json",
                               "submission/asc/mean_k_validation.json", "submission/asc/mean_k_validation.md",
                               "submission/asc/generated_proofread_validation.json",
                               "submission/asc/proofread_pdf_validation.json", "submission/asc/proof_check.md",
                               "submission/asc/visual_qa.json", "submission/asc/validate_focused_pdf.py",
                               "submission/asc/focused_stats_validation.json",
                               "submission/asc/focused_revision_validation.json",
                               "submission/asc/validate_r17_kendall.py", "submission/asc/r17_kendall_validation.json",
                               "submission/asc/validate_r17_pdf.py", "submission/asc/r17_revision_validation.json",
                               "submission/asc/r17_supplement_validation.json",
                               "submission/asc/validate_r18_pdf.py", "submission/asc/r18_revision_validation.json",
                               "submission/asc/validate_r19_pdf.py", "submission/asc/r19_revision_validation.json",
                               "submission/asc/validate_r20_pdf.py", "submission/asc/r20_revision_validation.json",
                               "submission/asc/validate_r22_pdf.py", "submission/asc/r22_revision_validation.json", "submission/asc/r22_assets_validation.json",
                               "submission/asc/r22_core_figures_validation.json",
                               "submission/asc/qa/r22_evidence_order/text_consistency.json",
                               "submission/asc/build_submission_bundle_r22.py",
                               "submission/asc/qa/r22_final_visual/text_consistency.json",
                               "submission/asc/r18_readable_tables_validation.json",
                               "submission/asc/qa/r18_readable_tables_validation.json",
                               "submission/asc/qa/save_r18_snapshot.py",
                               "submission/asc/qa/save_r19_snapshot.py",
                               "submission/asc/qa/save_r19_baseline.py",
                               "submission/asc/qa/save_r20_baseline.py", "submission/asc/qa/save_r20_snapshot.py",
                               "submission/asc/qa/make_r20_x1_table.py",
                               "submission/asc/qa/make_r20_s2_readable.py",
                               "submission/asc/qa/r20_s2_readable_validation.json",
                               "submission/asc/qa/r20_x1_core/*.csv", "submission/asc/qa/r20_x1_core/*.json",
                               "submission/asc/qa/r20_x1_core/*.py", "submission/asc/qa/r20_x1_core/*.md",
                               "submission/asc/qa/r20_literature/RESEARCH_NOTE.md",
                               "submission/asc/qa/r20_literature/provenance.json",
                               "submission/asc/qa/r19_final_visual/text_consistency.json",
                               "submission/asc/qa/r20_final_visual/text_consistency.json",
                               "submission/asc/qa/r18_rs_advantage/*.csv",
                               "submission/asc/qa/r18_rs_advantage/*.json",
                               "submission/asc/qa/r18_rs_advantage/*.py",
                               "submission/asc/qa/r18_rs_advantage/README.md"],
}
out = {"created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "files": {}, "job_counts": {}}
for g, pats in groups.items():
    for pat in pats:
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            # Filter by pathname before opening any file. Human verification
            # sheets and blind keys are excluded from the current manifest.
            excluded = ("DO_NOT_OPEN" in f.upper() or "verification_sheet" in f.lower()
                        or "verification_key" in f.lower() or "__pycache__" in f)
            if os.path.isfile(f) and "_fixture" not in f and "_partial" not in f and not excluded:
                out["files"][os.path.relpath(f, ROOT).replace("\\", "/")] = sha(f)
for base in ("results/jobs", "results/jobs_extra"):
    for d in sorted(glob.glob(os.path.join(ROOT, base, "*"))):
        if os.path.isdir(d):
            out["job_counts"]["%s/%s" % (base, os.path.basename(d))] = dict(
                done=len(glob.glob(os.path.join(d, "*.json"))), errors=len(glob.glob(os.path.join(d, "errors", "*.json"))))
# one hash over all job files (names + content hashes) so that the raw results can be checked as a whole
h = hashlib.sha256()
n = 0
for base in ("results/jobs", "results/jobs_extra"):
    for f in sorted(glob.glob(os.path.join(ROOT, base, "*", "*.json"))):
        h.update(os.path.relpath(f, ROOT).replace("\\", "/").encode()); h.update(sha(f).encode()); n += 1
out["all_job_files"] = dict(n=n, sha256=h.hexdigest())
json.dump(out, open(os.path.join(ROOT, "results", "MANIFEST.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("manifest: %d files, %d job files, job-set sha256 %s" % (len(out["files"]), n, h.hexdigest()[:16]))
