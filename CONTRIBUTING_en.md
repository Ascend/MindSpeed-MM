# Developer Contribution Guide

## Prerequisites

The following must be completed before review; otherwise the review will not proceed.

1. Compilation tests must pass, and Clean Code issues must be resolved. If tests fail, the reason must be noted. For non‑Clean Code issues, a waiver request must be submitted.

2. The PR title must be concise and complete following the format: `[header](backend/ops): description` (English is recommended for the description). The available headers are listed in the table below.

    | Header    | Content Involved                   |
    | ------       | -------------------------- |
    | feat    | New features, modules, or model integrations  |
    | fix      | Bug fixes                   |
    | docs        |Documentation additions or updates             |
    | style       | Code changes to comply with Clean Code standards |
    | adaptor   | Model source code integration               |
    | chore   | Standalone test case submissions         |

    For the `backend/ops` field, you may specify `torch` or `triton`. If omitted, it defaults to `torch`.

    Examples of PR titles:
    - `feat(triton): optimize solve_tril of GDN`: indicates a performance optimization for a Triton operator.
    - `docs: Add FSDP2 Muon optimizer feature guide`: indicates documentation for the Muon optimizer in the FSDP2 backend.

3. Fill in the PR description according to the template in `.gitcode/PULL_REQUEST_TEMPLATE.md`. The template will be automatically generated when a PR is created. Do not remove any sections; if a section does not apply, clearly state why.

4. Code must be thoroughly self‑tested and self‑reviewed before a review is requested.

5. The CLA must be signed, and the PR must display the `CLA yes` label.

6. Submitted code must be associated with an issue. Model code submissions and performance optimization contributions that are part of the version roadmap may be linked directly to the current Roadmap issue. For open‑source contributors who are not project members and lack permission to set issue associations, you may copy the issue link directly into the PR description. After the code is merged, the corresponding issue should be closed promptly.

## Commit Requirements

1. A PR should focus on a single logical change. Changes for different purposes should be submitted as separate PRs.

2. Multiple commits within a single PR must be squashed; at most two commits are allowed.

3. Commit messages must clearly describe the code changes. Vague descriptions such as "fix bug" or "add adaptor file" will not be accepted.

4. Regular expressions must undergo security scanning. Any public network addresses included must be explicitly declared.

5. Code submissions that introduce new features or new models must include test cases. If the test cases are not included in the current PR or if existing test cases are used, please provide the associated PR link or test case path in the `How was this patch tested?` section of the PR description.

## Review Requirements

1. Reviewers must conduct thorough reviews, provide meaningful feedback, and must not approve PRs without proper review. PRs must not be forcibly merged due to business urgency.

2. Review comments should be as detailed as possible, preferably with suggestions for improvement.

3. All review comments must be resolved. Project members must check the "Resolved" part to confirm that all open issues have been addressed before merging. For community developers who are not project members and lack resolution permissions, each review comment must be replied to individually.

4. PRs should be closed promptly. After review comments have been addressed, reviewers must add the appropriate labels.

5. For code merges that lack test cases (excluding documentation changes), the committer must provide an explanation in the comments along with validation conclusions before merging.

## Commit Message and Changelog Writing Guide

For details, click [https://www.ruanyifeng.com/blog/2016/01/commit_message_change_log.html](https://gitee.com/link?target=https%3A%2F%2Fwww.ruanyifeng.com%2Fblog%2F2016%2F01%2Fcommit_message_change_log.html).
