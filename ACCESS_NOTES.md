# ACCESS_NOTES.md — what we know, and do not know, about reading CassavaBase

## SweetPotatoBase (added 2026-09-28 by Bidhan)
CassavaBase was not reachable on 2026-09-28, so the Step 6.3 demonstration uses SweetPotatoBase
(https://sweetpotatobase.org/brapi/v2), a sister Breedbase server. Its Data Usage Policy
(https://sweetpotatobase.org/usage_policy) follows the same Toronto Agreement terms: cite the source
and version, respect the producers' first global analyses and publication plans, expect possible
quality issues in prepublication data, and contact producers about overlapping publication plans.
Public GET only, no credentials, one study and one variable, bounded attempts; real bytes stay under
git-ignored folders (cache/live, snapshots/) until the redistribution question is answered.

Reviewed by Bidhan on: 2026-09-28

#CassavaBase’s Data Usage Policy follows the Toronto Agreement. It encourages analysis of released prepublication data, while asking users to cite sources and relevant versions, respect producers’ initial global analyses, read project documentation, and discuss overlapping publication plans with the producers.The website’s “Cite CassavaBase” menu points to its Help page, which requests the following paper when CassavaBase contributes to research. CassavaBase
Fernandez-Pozo, N., et al. (2015). The Sol Genomics Network (SGN): from genotype to phenotype to breeding. Nucleic Acids Research, 43(Database issue), D1036–D1041. DOI: 10.1093/nar/gku1195.  

## Why this file exists

A GET request asks to read. It is not proof that automated access, caching or
redistribution is permitted. This project is public-data-only and educational.
Before any live request, the person — not the program and not the AI helper —
decides whether access is appropriate. Nothing below invents a permission or
quotes a term of service that has not been read.

## Known (facts we can point at)

- Target server: `https://cassavabase.org/brapi/v2` (from `.env.example`).
- `brapi_ping.py` (repo root, Bidhan's file) is designed to call three public
  read endpoints (`/serverinfo`, `/commoncropnames`, `/studies`) with **no
  token**. Whether it passed is Bidhan's own record; it has not been run in
  the part2 work.
- part2 sends only GET requests, only to allowlisted BrAPI paths, with explicit
  timeouts, a byte cap, a bounded attempt budget and Retry-After honoured.
- part2 stores each reply with its hash so a result can cite the exact bytes
  it used; it never modifies anything on the server.

## Unresolved (questions only the data owner or their published terms can answer)

| Question | Status | Where to look / whom to ask |
| --- | --- | --- |
| Is automated read access allowed without registration or an account? | unresolved | the site's terms/usage page; the operators' contact address |
| Are there rate limits or request quotas we must respect? | unresolved | same; also any `Retry-After` behaviour observed during the approved probe |
| May replies be cached locally for repeated offline use? | unresolved | terms of use |
| Is attribution required when results are shown or published, and in what form? | unresolved | terms of use / citation guidance on the site |
| May downloaded records be redistributed (for example in a public repository or a snapshot)? | unresolved | terms of use; until answered, real snapshots stay git-ignored |
| Who operates the server and how do we reach them? | unresolved | record the contact you find on the site here |

## How to resolve (Bidhan's steps; record what you find here with a date)

1. Open cassavabase.org and look for terms of use, data-use or citation
   pages. Paste the link(s) and the date you read them below.
2. If the terms do not answer a question, write to the operators using the
   contact on the site. Paste the question you asked and any reply, with dates.
3. Only after that, decide whether the live probe (one `GET /serverinfo`) is
   appropriate, and update the "Reviewed by Bidhan on" line.

### Findings log

- (empty — add dated entries here)

## Rules the code follows regardless of the answers

- Offline is the default. A live run needs `BRAPI_MODE=live` plus a
  `FetchApproval` created by controller code for that specific run.
- The first live action is a single `GET /serverinfo`; nothing else is called
  until the person reads what the server advertises and approves the next step.
- Any real data fetched later stays under git-ignored folders until the
  redistribution question is answered.



