# Credentials

## Rotate the StageIT API token

A StageIT API token was committed to `staging/.stagingrc` in commit `c0da3f9`
and remains in the history of `origin/main`, `origin/euroflash-port` and
`origin/lofar-dm-matching` on `github.com/DirkKuiper/lotaas_pipeline`.

The working copy no longer holds it and `.gitignore` now blocks the path, but
**neither revokes the token**. Anyone who has cloned, forked or mirrored the
repository still has it, and GitHub keeps unreachable objects served by SHA
for some time after a rewrite. Rotation at the LTA is the only fix:

1. Issue a new StageIT API token and revoke the old one.
2. Put the new token in `~/.config/lotaas/stagingrc`, mode 0600, as
   `api_token = ...`, or point `LOTAAS_STAGING_CONFIG` at another private file.
3. Confirm nothing in the working tree carries it:
   `git grep -nI -e 'api_token *=' -e 'xox[baprs]-' -- . ':!tests'`

Purging history is secondary, does not substitute for rotation, and rewrites
published commits that other clones and branches point at. If you want it
anyway, after rotating:

```bash
git clone --mirror git@github.com:DirkKuiper/lotaas_pipeline.git
cd lotaas_pipeline.git
git filter-repo --invert-paths --path staging/.stagingrc   # pip install git-filter-repo
git push --force --mirror
```

Then ask GitHub Support to expire the cached unreachable objects, and have
everyone re-clone: a `git pull` into an existing clone keeps the old commits.

## Where credentials belong

Both credential files live outside the repository, at mode 0600, and neither
is copied into the container image or the source snapshot sent to compute
nodes:

| Purpose | Default path | Override | Contents |
| --- | --- | --- | --- |
| StageIT / LTA | `~/.config/lotaas/stagingrc` | `LOTAAS_STAGING_CONFIG` | `api_token = ...` |
| Slack notifier | `~/.config/lotaas/slackrc` | `LOTAAS_SLACK_CONFIG` | `bot_token = ...`, `channel_id = ...` |

`SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID` override the Slack file. Download
macaroons are fetched per request at run time and are never written to the
repository; `euroflash.access_check` reports their path caveats and HTTP
status without printing any token.

Notifications are sent only from the head node. The classifier on a compute
node logs instead of posting, so no Slack token needs to exist there.
