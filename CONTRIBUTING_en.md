# Developer Contribution Guide

## Prerequisites (Must Be Completed Before Review)

1. The compilation and tests pass, and Clean Code issues are resolved. If a test fails, the reason must be noted; non-Clean Code issues require applying for suppression.

2. The PR title is concise and complete (the title follows the format "header name(backend/ops): + description", and an English description is recommended). All headers are shown in the following table:

    | Header name | Involved content |
    | ------       | -------------------------- |
    | feat    | Merging of new features, modules, and models |
    | fix      | Defect fixes |
    | docs        | Adding or modifying documentation |
    | style       | Modifying code to comply with Clean Code standards |
    | adaptor   | Merging of model source code |
    | chore   | Test cases committed separately |

    The `backend/ops` item in parentheses can be filled with `torch` or `triton`; if left blank, it defaults to `torch`.

    The following are two examples of PR titles:
    - `feat(triton)`: optimize `solve_tril` of GDN (indicates a performance optimization for the triton operator)
    - `docs`: Add FSDP2 Muon optimizer feature guide (indicates documentation for the FSDP2 Muon optimizer)

3. Fill in the PR description according to the template in `.gitcode/PULL_REQUEST_TEMPLATE.md`. The template is generated automatically after a PR is created. Do not delete relevant content arbitrarily; if a section is not applicable, directly state the reason why it is not applicable.

4. The code must be fully self-verified and self-checked, and there must be no obvious issues before requesting review;

5. Complete the CLA signing, and the PR should display the `CLA yes` label.

6. The submitted code must be associated with an issue. Model code submissions and performance optimization submissions within the version plan can be directly associated with the current version's Roadmap. Open-source community contributors who are not project members and do not have permission to associate an issue can directly copy the issue link in the PR description. After the code is merged, the issue should be closed in a timely manner.

## Commits Requirements

1. A PR must serve a single purpose. Modifications for different purposes should be split into multiple PRs.

2. Multiple commit records within a single PR must be squashed, with at most two commits.

3. Commit messages must clearly describe the code functionality. Vague descriptions such as "fix bug" or "add adapter files" will not be accepted.

4. Regular expressions must undergo security scanning, and public network addresses must be declared.

5. Code submissions involving new features or new models must be covered by test cases. If the test cases are not submitted in this PR, or if test cases already exist, provide the related PR link or the test case path in the `How was this patch tested?` section of the PR description.

## Review Requirements

1. Reviewers must review strictly and provide effective review comments. They must not approve directly, nor force a merge on the grounds of business urgency.

2. Review comments should be as detailed as possible, preferably with suggested modifications.

3. All review comments must be closed. Members of this project need to check "Resolved" to ensure that all unresolved issues have been addressed before merging. Community developers who are not members of this project and do not have the permission to resolve must reply to each review comment.

4. PRs should be closed as soon as possible. Reviewers need to approve after the review comments are closed.

5. For code merged without test cases (except documentation changes), the committer must explain this in the comments and provide relevant verification conclusions before merging.

## Commit Message and Changelog Writing Guide

For details, see [https://www.ruanyifeng.com/blog/2016/01/commit_message_change_log.html](https://gitee.com/link?target=https%3A%2F%2Fwww.ruanyifeng.com%2Fblog%2F2016%2F01%2Fcommit_message_change_log.html).
