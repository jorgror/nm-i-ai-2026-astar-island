# NM i AI Rules (Summary)

Source: [app.ainm.no/rules](https://app.ainm.no/rules)  
Fetched: March 20, 2026 (Europe/Oslo)

This file is a practical summary for our team workflow. The official rules page is authoritative.

## What Applies Right Now

- The page contains both pre-competition rules and main competition rules.
- For this weekend challenge window, the **Main Competition Rules** are the relevant section.

## Main Competition Rules (Practical Summary)

## 1) Schedule

- Start: Thursday, March 19, 2026 at 18:00 CET.
- Deadline: Sunday, March 22, 2026 at 15:00 CET.
- No late submissions count, except in-flight submissions already processing at deadline.

## 2) Team and Eligibility

- Participants must be at least 15 years old.
- Team size is limited.
- A person can only be on one team.
- Team roster locks after first submission in any main task.

## 3) Scoring and Tasks

- Three independent tasks contribute to total ranking.
- Overall ranking is based on normalized per-task scores.
- Missing a task effectively gives zero contribution for that task.
- Task-specific formats/rate limits are separate and must also be followed.

## 4) Prize Eligibility Requirements

- All members must complete Vipps identity verification before deadline.
- Team must submit a public code repository URL before deadline.
- Non-eligible teams may still appear on leaderboard but are not paid prizes.

## 5) Code and License Requirement

- Prize-eligible code must be open-sourced in a public repository.
- Required license is MIT or equivalent permissive license.

## 6) Fair Play and Prohibited Conduct

- AI assistants and open-source tools are explicitly allowed.
- Strictly prohibited:
  - collusion and solution sharing between teams
  - account/team manipulation for extra attempts
  - bypassing limits or attacking platform infrastructure
  - extracting hidden data/eval internals from platform
  - score manipulation via non-genuine hardcoded behavior

## 7) Monitoring and Enforcement

- Organizers monitor submissions, logs, code similarity, and communication channels.
- Possible outcomes include warning, prize ineligibility, score removal, or ban.
- Jury decisions are binding; no formal appeals process is guaranteed during the short main event window.

## 8) Platform, Conduct, and Updates

- No guaranteed compensation for downtime.
- Code of conduct applies across channels (including Slack).
- Rules can be updated during competition; teams must monitor announcements.

## Do We Follow The Rules?

## Current status: **Mostly yes for tooling behavior**, with team-level items to confirm.

What our current scripts already align with:

- We do not attempt to bypass platform limits.
- Query and submit flow is designed for normal API usage.
- We only log our own team interactions.
- We do not include any mechanism for scraping other teams or reverse-engineering internals.

What still needs explicit team action outside code:

- Ensure all members complete Vipps verification before deadline.
- Ensure repository is public and license is MIT (or equivalent permissive) before prize cutoff.
- Ensure no cross-team sharing/collusion in discussions or code.
- Keep watching official Slack/platform announcements for rule amendments.

## Competition Safety Checklist

- [ ] Vipps verification completed for all team members.
- [ ] Repository is public.
- [ ] `LICENSE` file exists and is MIT/permissive.
- [ ] Team roster finalized before first submission.
- [ ] No cross-team sharing of competition-specific observations.
- [ ] Follow task-specific limits and submission formats.

