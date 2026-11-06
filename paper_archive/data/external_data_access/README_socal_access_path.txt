SoCal full-dataset access path
Date: 2026-07-02

Why this exists
- The current public sample is scientifically useful, but it still lacks a synchronized
  topology + measurement overlap window for a true real-data posterior benchmark.
- The official full dataset exists and is the most natural path to close that gap.

Official source confirmed
- GitHub repository:
  https://github.com/caltech-netlab/digital-twin-dataset
- Dataset portal:
  https://socal28bus.caltech.edu/

What the official README says
- The full dataset is hosted at the portal and accessed via a REST API.
- Users authenticate with GitHub.
- Access requires submitting a ticket so that the GitHub account is added to the
  allowlist.
- For best quality / coverage, use data between September 2024 and September 2025.

What this means operationally
- The blocker is no longer discovery.
- The blocker is now authentication / allowlisting.

Shortest path to unblock
1. Have a usable GitHub account.
2. Submit an access ticket to the official form.
3. Once allowlisted, use the repository's DatasetApiClient or notebook examples
   to download synchronized topology + measurement windows.
4. Place the downloaded data locally and continue the real-data posterior benchmark.

Files prepared in this directory
- socal_access_request_draft_email.txt
- socal_access_request_short_form.txt
- socal_data_request_checklist.txt

Current paper-safe fallback if access is not granted in time
- Keep the current bounded claim:
  real-network topology-state compatibility, measurement-ingest compatibility,
  measurement-only regime library, and explicit diagnosis of the missing
  synchronized public overlap window.
