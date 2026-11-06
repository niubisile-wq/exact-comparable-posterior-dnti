# Code Repository DOI And Public Link Guide

This file explains how to create a public code/data link and a citable DOI for the IJEPES submission.

## Recommendation

Best for reviewer confidence: prepare the repository now and, if the authors are comfortable releasing the materials before peer review, make it public before submission and archive a release with Zenodo.

Best compromise if early public release is undesirable: keep the GitHub repository private for the first submission, keep the current "available upon reasonable request / public repository upon acceptance" wording, and make the repository public with a Zenodo DOI at revision, acceptance, or before publication. This is weaker than a public DOI at submission, but it avoids premature release.

For this manuscript, my practical recommendation is:

1. Prepare the repository immediately.
2. Do not enter a fake DOI or fake URL.
3. If there is no confidentiality or third-party licence risk, make the repository public before submission and use the public DOI wording in `DATA_CODE_AVAILABILITY_TEXT_20260706.md`.
4. If there is any concern about releasing code/data early, submit with the current request-based wording and publish the Zenodo DOI at revision/acceptance.

## Why GitHub Plus Zenodo

GitHub URL: living repository link, useful for browsing code and issues.

Zenodo DOI: frozen, citable archive of a specific release, better for journal records.

Example public links after completion:

```text
GitHub repository:
https://github.com/<github-user-or-org>/<repository-name>

Zenodo DOI:
https://doi.org/10.5281/zenodo.<record-number>
```

## Repository Name

Use a concise, paper-specific name. Suggested names:

```text
exact-comparable-posterior-dnti
distribution-topology-posterior-contract
ijepes-posterior-topology-identification
```

## Minimum Repository Structure

Use this structure before creating a release:

```text
README.md
LICENSE
CITATION.cff
requirements.txt or environment.yml
scripts/
experiments/
figures/
manifests/
source_data/
external_data_notes/
docs/
```

Minimum contents:

- `README.md`: paper title, repository purpose, exact commands to regenerate figures/results, headline-result verification table, external-data notes.
- `LICENSE`: code licence, usually MIT or BSD-3-Clause if the authors agree.
- `CITATION.cff`: author and citation metadata for GitHub/Zenodo.
- `requirements.txt` or `environment.yml`: reproducible Python environment.
- `scripts/`: code for topology libraries, posterior training/evaluation, baselines, ablations, stress tests, and plotting.
- `manifests/`: frozen result manifests and traceability maps.
- `source_data/`: figure source data and derived tables that the authors are allowed to share.
- `external_data_notes/`: access instructions for reused public or third-party data that cannot be redistributed.
- `docs/`: reviewer reproduction guide and data dictionary.

## Do Not Upload

Do not upload:

- passwords, API keys, tokens, `.env` files, browser cookies, SSH keys, or account files
- private utility/field data that the authors are not allowed to redistribute
- personal information or confidential metadata
- copyrighted third-party datasets unless redistribution is explicitly allowed
- LaTeX temporary files, old draft figures, cache folders, model checkpoints that are not needed, or absolute local paths
- files that make numerical claims inconsistent with the manuscript

## GitHub Steps

1. Log in to GitHub.
2. Create a new repository using the selected repository name.
3. Choose visibility:
   - `Private` if you want to prepare first and publish later.
   - `Public` if you want a public link and Zenodo DOI now.
4. Upload the repository files or push them with Git.
5. Check that the README explains exactly how to reproduce the main figures and headline numbers.
6. Check that the licence is present.
7. Create a release:
   - tag: `v1.0.0-submission` or `v1.0.0`
   - release title: `Submission archive for exact-comparable posterior topology identification`
   - release notes: briefly state that this is the frozen code/data package supporting the IJEPES submission.

## Zenodo DOI Steps

Zenodo can archive a GitHub release and mint a DOI only after the target repository is public and connected.

1. Log in to Zenodo.
2. Connect Zenodo to GitHub.
3. On Zenodo's GitHub page, click `Sync now`.
4. Find the public GitHub repository and toggle it on.
5. Create a GitHub release if you have not already done so.
6. Wait for Zenodo to process the release.
7. Open the Zenodo record and copy the DOI.
8. Replace `[DOI]` and `[URL]` in `DATA_CODE_AVAILABILITY_TEXT_20260706.md`.
9. Update the manuscript and submission-system boxes only after the DOI resolves to the intended landing page.

## Manuscript Text After DOI Exists

Use the public DOI wording in `DATA_CODE_AVAILABILITY_TEXT_20260706.md`. Do not keep "upon acceptance" wording if a real DOI already exists.

## Manuscript Text Before DOI Exists

Use the current request-based wording in `DATA_CODE_AVAILABILITY_TEXT_20260706.md`. Do not claim that data/code are public unless the link works outside the author account.

## Final Checks Before Public Release

- The GitHub repository opens in a private/incognito browser window.
- The Zenodo DOI resolves to the correct record.
- The Zenodo file list matches the GitHub release.
- The README contains clear reproduction commands.
- The source data map names every figure/table that depends on deposited files.
- The licence is correct for code and shareable derived data.
- Reused third-party data are cited and access-noted, not illegally redistributed.
- The manuscript Data Availability and Code Availability statements match the repository.

## Official Guidance Checked On 2026-07-06

- IJEPES/Elsevier guide: the journal applies research-data Option C, requiring repository deposit and article citation/linking where possible, or a statement explaining why data cannot be shared.
- GitHub Docs: Zenodo can archive a public GitHub repository and issue a DOI for the archive; Zenodo can only access public repositories.
- Zenodo Docs: after a repository is enabled, new releases are ingested and archived; the release record DOI is available after processing.
- GitHub Docs: making a repository public means the code is visible to everyone and can be forked; check secrets and private data before changing visibility.
