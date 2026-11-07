import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genai_evals.adoption import (
    SessionTrace,
    adoption_report,
    evaluate_gate,
    report_markdown,
    team_metrics,
)


def _trace(team, agent="claude-code", pr_id=None, lines_total=100, lines_ai=50,
           merged=False, reverted=False, eval_passed=None, at_seq=0):
    return SessionTrace(
        team=team, agent=agent, pr_id=pr_id, lines_total=lines_total,
        lines_ai=lines_ai, merged=merged, reverted=reverted,
        eval_passed=eval_passed, at_seq=at_seq,
    )


def test_team_metrics_basic():
    rows = [
        _trace("backend", lines_total=100, lines_ai=80, merged=True),       # seq 0: ai 80%
        _trace("backend", lines_total=100, lines_ai=20, merged=True),       # seq 1: ai 20%
    ]
    m = team_metrics(rows)
    assert m.team == "backend"
    assert m.sessions == 2
    assert m.adoption_share == 0.5
    assert m.acceptance_rate == 1.0
    assert m.revert_rate == 0.0
    assert m.eval_pass_rate == 0.0


def test_trend_growing_when_second_half_outpaces():
    rows = [
        _trace("t", lines_total=100, lines_ai=10, at_seq=0),
        _trace("t", lines_total=100, lines_ai=20, at_seq=1),
        _trace("t", lines_total=100, lines_ai=80, at_seq=2),
        _trace("t", lines_total=100, lines_ai=90, at_seq=3),
    ]
    m = team_metrics(rows)
    assert m.first_half_share == 0.15
    assert m.second_half_share == 0.85
    assert m.trend == "growing"


def test_trend_declining():
    rows = [
        _trace("t", lines_total=100, lines_ai=90, at_seq=0),
        _trace("t", lines_total=100, lines_ai=80, at_seq=1),
        _trace("t", lines_total=100, lines_ai=10, at_seq=2),
        _trace("t", lines_total=100, lines_ai=20, at_seq=3),
    ]
    m = team_metrics(rows)
    assert m.trend == "declining"


def test_evals_rate_counts_evaluated_only():
    rows = [
        _trace("t", lines_ai=50, eval_passed=True),
        _trace("t", lines_ai=50, eval_passed=True),
        _trace("t", lines_ai=50, eval_passed=False),
        _trace("t", lines_ai=50, eval_passed=None),  # not evaluated
    ]
    assert team_metrics(rows).eval_pass_rate == 0.6667


def test_gate_blocks_low_eval_pass_rate():
    m = team_metrics([
        _trace("t", lines_ai=50, eval_passed=True),
        _trace("t", lines_ai=50, eval_passed=False),
        _trace("t", lines_ai=50, eval_passed=False),
        _trace("t", lines_ai=50, eval_passed=False),
    ])
    g = evaluate_gate(m)
    assert not g.allowed
    assert "eval pass rate" in g.blocked_reason


def test_gate_blocks_high_revert_rate():
    rows = [
        _trace("t", lines_ai=50, merged=True, reverted=True, eval_passed=True),
        _trace("t", lines_ai=50, merged=True, reverted=True, eval_passed=True),
        _trace("t", lines_ai=50, merged=True, reverted=True, eval_passed=True),
        _trace("t", lines_ai=50, merged=True, reverted=False, eval_passed=True),
    ]
    g = evaluate_gate(team_metrics(rows))
    assert not g.allowed
    assert "revert rate" in g.blocked_reason


def test_gate_allows_healthy_team():
    rows = [
        _trace("t", lines_total=100, lines_ai=100, merged=True, eval_passed=True, at_seq=i)
        for i in range(5)
    ]
    g = evaluate_gate(team_metrics(rows))
    assert g.allowed
    assert g.blocked_reason == ""


def test_gate_trend_alert_blocks_collapse():
    rows = [
        _trace("t", lines_total=100, lines_ai=90, eval_passed=True, at_seq=0),
        _trace("t", lines_total=100, lines_ai=90, eval_passed=True, at_seq=1),
        _trace("t", lines_total=100, lines_ai=5, eval_passed=True, at_seq=2),
        _trace("t", lines_total=100, lines_ai=5, eval_passed=True, at_seq=3),
    ]
    g = evaluate_gate(team_metrics(rows))
    assert not g.allowed
    assert "declining" in g.blocked_reason


def test_report_markdown_lists_teams_and_verdicts():
    rows = [
        _trace("backend", lines_total=100, lines_ai=80, merged=True, eval_passed=True, at_seq=0),
        _trace("mobile", lines_total=100, lines_ai=10, reverted=True, eval_passed=False, at_seq=1),
    ]
    md = report_markdown(adoption_report(rows))
    assert "# AI SDLC adoption report" in md
    assert "backend" in md
    assert "mobile" in md
    assert "BLOCKED" in md


def test_empty_traces_no_division_errors():
    m = team_metrics([])
    assert m.adoption_share == 0.0
    assert m.sessions == 0
    assert m.trend == "flat"
