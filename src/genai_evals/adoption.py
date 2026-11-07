"""AI-assisted SDLC adoption metrics and merge quality gates.

Teams adopt coding agents (Claude Code, Cursor, Aider, Copilot, Continue,
Codeium) without instrumenting the adoption, so nobody can tell whether
agent-written code actually helps or hurts. This module ingests agent
session traces and computes per-team adoption and quality metrics, then
applies merge quality gates:

- adoption_share: AI-authored lines / total lines authored
- acceptance_rate: accepted AI PRs / closed AI PRs
- revert_rate: reverted AI PRs / total AI PRs
- eval_pass_rate: AI PRs with passing evals / evaluated AI PRs
- trend: growing | flat | declining (first half vs second half)

Gates block merging when eval_pass_rate < threshold or revert_rate >
threshold, and escalate to human review when adoption collapses while
reverts spike. Deterministic, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class SessionTrace:
    team: str
    agent: str
    pr_id: str | None = None
    lines_total: int = 0
    lines_ai: int = 0
    merged: bool = False
    reverted: bool = False
    eval_passed: bool | None = None
    at_seq: int = 0


@dataclass
class TeamAdoption:
    team: str
    sessions: int
    adoption_share: float
    acceptance_rate: float
    revert_rate: float
    eval_pass_rate: float
    first_half_share: float
    second_half_share: float
    trend: str


@dataclass
class GateDecision:
    team: str
    allowed: bool
    blocked_reason: str


@dataclass
class AdoptionReport:
    teams: list[TeamAdoption]
    gates: list[GateDecision]
    overall_adoption_share: float
    overall_acceptance_rate: float
    overall_eval_pass_rate: float


DEFAULT_THRESHOLDS = {
    "min_eval_pass_rate": 0.80,
    "max_revert_rate": 0.15,
    "trend_alert_drop": 0.10,
}


def _share(rows: list[SessionTrace]) -> float:
    total = sum(r.lines_total for r in rows)
    if total <= 0:
        return 0.0
    return sum(r.lines_ai for r in rows) / total


def _acceptance(rows: list[SessionTrace]) -> float:
    ai = [r for r in rows if r.lines_ai > 0]
    closed = [r for r in ai if r.merged or r.reverted or r.pr_id is not None]
    if not closed:
        return 0.0
    return sum(1 for r in closed if r.merged) / len(closed)


def _eval_rate(rows: list[SessionTrace]) -> float:
    evaled = [r for r in rows if r.eval_passed is not None]
    if not evaled:
        return 0.0
    return sum(1 for r in evaled if r.eval_passed) / len(evaled)


def team_metrics(rows: list[SessionTrace]) -> TeamAdoption:
    rows = sorted(rows, key=lambda r: r.at_seq)
    half = max(1, len(rows) // 2)
    first, second = rows[:half], rows[half:]
    share_first = _share(first)
    share_second = _share(second)
    if share_second > share_first * 1.05:
        trend = "growing"
    elif share_second < share_first * 0.95:
        trend = "declining"
    else:
        trend = "flat"
    ai = [r for r in rows if r.lines_ai > 0]
    closed = [r for r in ai if r.merged or r.reverted or r.pr_id is not None]
    return TeamAdoption(
        team=rows[0].team if rows else "unknown",
        sessions=len(rows),
        adoption_share=round(_share(rows), 4),
        acceptance_rate=round(_acceptance(rows), 4),
        revert_rate=round(
            sum(1 for r in ai if r.reverted) / max(1, len(ai)), 4
        ),
        eval_pass_rate=round(_eval_rate(rows), 4),
        first_half_share=round(share_first, 4),
        second_half_share=round(share_second, 4),
        trend=trend,
    )


def evaluate_gate(metrics: TeamAdoption, thresholds: dict | None = None) -> GateDecision:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if metrics.eval_pass_rate < t["min_eval_pass_rate"]:
        return GateDecision(
            team=metrics.team,
            allowed=False,
            blocked_reason=(
                f"eval pass rate {metrics.eval_pass_rate:.2f} below "
                f"threshold {t['min_eval_pass_rate']:.2f}; needs human review"
            ),
        )
    if metrics.revert_rate > t["max_revert_rate"]:
        return GateDecision(
            team=metrics.team,
            allowed=False,
            blocked_reason=(
                f"revert rate {metrics.revert_rate:.2f} above "
                f"threshold {t['max_revert_rate']:.2f}; needs human review"
            ),
        )
    if metrics.trend == "declining" and (metrics.first_half_share - metrics.second_half_share) > t["trend_alert_drop"]:
        return GateDecision(
            team=metrics.team,
            allowed=False,
            blocked_reason=(
                f"adoption declining ({metrics.first_half_share:.2f} -> "
                f"{metrics.second_half_share:.2f}); coaching review"
            ),
        )
    return GateDecision(team=metrics.team, allowed=True, blocked_reason="")


def adoption_report(
    traces: list[SessionTrace], thresholds: dict | None = None
) -> AdoptionReport:
    by_team: dict[str, list[SessionTrace]] = defaultdict(list)
    for t in traces:
        by_team[t.team].append(t)
    teams = [team_metrics(rows) for _, rows in sorted(by_team.items())]
    gates = [evaluate_gate(m, thresholds) for m in teams]
    all_rows = list(traces)
    return AdoptionReport(
        teams=teams,
        gates=gates,
        overall_adoption_share=round(_share(all_rows), 4),
        overall_acceptance_rate=round(_acceptance(all_rows), 4),
        overall_eval_pass_rate=round(_eval_rate(all_rows), 4),
    )


def report_markdown(report: AdoptionReport) -> str:
    lines = [
        "# AI SDLC adoption report",
        "",
        f"- Overall adoption share: {report.overall_adoption_share:.1%} of lines",
        f"- Overall acceptance rate: {report.overall_acceptance_rate:.1%} of AI PRs",
        f"- Overall eval pass rate: {report.overall_eval_pass_rate:.1%}",
        "",
        "## Per team",
    ]
    for m in report.teams:
        lines.append(
            f"- **{m.team}**: {m.sessions} sessions, adoption "
            f"{m.adoption_share:.1%}, acceptance {m.acceptance_rate:.1%}, "
            f"eval pass {m.eval_pass_rate:.1%}, revert {m.revert_rate:.1%}, "
            f"trend {m.trend}"
        )
    lines += ["", "## Gates"]
    for g in report.gates:
        verdict = "allow" if g.allowed else f"BLOCKED: {g.blocked_reason}"
        lines.append(f"- {g.team}: {verdict}")
    return "\n".join(lines) + "\n"
