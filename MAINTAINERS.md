# Definitions of Roles

## Maintainer

- A committer that is also an arbiter for other committers in the Github/Git repositories.
- May override merging criteria to merge/close a pull-request or to push/revert a commit.

## Committer (previously Reviewer)

- May reject a pull-request, which prohibits merging. "Veto"
- May vote for an approval.
- (Requires merge privilege) May merge a pull-request if it has enough number of approval-votes from committers (2 or more).
- May propose new committers among well-known contributors and vote for new committers
- Vote for a committer removal.
- The ability of reject and merge may be limited to a few subdirectories.
- The list of reviewers and their subdirectories is at [CODEOWNERS].

## Contributor

- Anyone who has sent a pull-request, which is accepted and merged with the full procedures.

## Note

We allow anyone to send and review pull-requests (although may be not allowed to vote), or to write an issue as long as they do no harm or break [CODE_OF_CONDUCT.md]