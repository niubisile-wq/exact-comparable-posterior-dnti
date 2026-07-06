# Release Checklist

## Must Complete Before Public Release

- Add final public repository URL.
- Add DOI or other persistent identifier after archival upload.
- Select and add a license.
- Remove local absolute paths or replace them with environment variables.
- Confirm that third-party raw data can be redistributed; otherwise publish access notes only.
- Decide how to handle model checkpoint files larger than normal Git limits.
- Add checksums for frozen result manifests and final figure source data.
- Verify that all manuscript headline numbers can be traced to frozen files.
- Keep `RESULT_TRACEABILITY.md` synchronized with the final manuscript tables, figures, and captions.
- Compile the manuscript from the public repository layout.
- Search for `TODO`, `TBD`, `PLACEHOLDER`, local user paths, and desktop paths; resolve all findings before public release.

## Optional But Strong

- Create a Zenodo draft release linked to GitHub.
- Add a small smoke-test script that verifies the most important frozen numbers without rerunning training.
