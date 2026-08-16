import React, { useMemo, useState } from "react";
import "./ExplainabilityCard.css";

function compact(value) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toFixed(1) : "0.0";
}

function prettyTag(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function buildFeatureRows(row) {
  return [
    {
      label: "ML Suitability",
      value: Number(row.ml_suitability_score || 0),
      note: "Model confidence from the production ensemble",
    },
    {
      label: "Quality Score",
      value: Number(row.quality_score || 0),
      note: "Overall cricket-performance quality",
    },
    {
      label: "Recent Form",
      value: Number(row.recent_form_score || 0),
      note: "Recent performance trend strength",
    },
    {
      label: "Reliability",
      value: Number(row.reliability_score || 0),
      note: "How trustworthy the sample size is",
    },
    {
      label: "Similarity",
      value: Number(row.similarity_score || 0),
      note: "How close the player is to the IPL benchmark profile",
    },
  ].filter((item) => item.value > 0);
}

const ExplainabilityCard = ({ row, player, team, score, role }) => {
  const [isOpen, setIsOpen] = useState(false);

  const summaryText = useMemo(() => {
    const deficit = Number(row?.role_deficit || 0);
    const current = Number(row?.current_role_count || 0);
    const target = Number(row?.target_role_count || 0);
    const ml = Number(row?.ml_suitability_score || 0);
    const benchmark = row?.closest_ipl_benchmark;

    const gapSentence =
      deficit > 0
        ? `${team} currently has ${current} of the target ${target} ${prettyTag(role)} profiles.`
        : `${team} still shortlisted this profile as added depth for ${prettyTag(role)}.`;

    const benchmarkSentence = benchmark
      ? `Closest IPL benchmark: ${benchmark}.`
      : "No direct IPL benchmark match is stored for this player.";

    return `${gapSentence} ${player} is recommended because the model scores this profile at ${compact(
      ml
    )}/100 for ${prettyTag(role)} suitability. ${benchmarkSentence}`;
  }, [player, role, row, team]);

  const featureRows = useMemo(() => buildFeatureRows(row || {}), [row]);
  const strongestReasons = useMemo(() => {
    const reasons = [];
    if (Number(row?.role_deficit || 0) > 0) {
      reasons.push(`${team} needs ${Number(row.role_deficit)} more ${prettyTag(role)} option${Number(row.role_deficit) > 1 ? "s" : ""}`);
    }
    if (Number(row?.ml_suitability_score || 0) > 0) {
      reasons.push(`Ensemble ML suitability is ${compact(row.ml_suitability_score)}/100`);
    }
    if (Number(row?.recent_form_score || 0) >= 60) {
      reasons.push(`Recent form is strong at ${compact(row.recent_form_score)}`);
    }
    if (Number(row?.reliability_score || 0) >= 70) {
      reasons.push(`Reliability is high at ${compact(row.reliability_score)}`);
    }
    if (row?.closest_ipl_benchmark) {
      reasons.push(`IPL profile match is closest to ${row.closest_ipl_benchmark}`);
    }
    return reasons.slice(0, 4);
  }, [row, role, team]);

  return (
    <div className="explainability-wrapper">
      <div className="recommendation-header">
        <div className="player-info">
          <span className="player-name">{player}</span>
          <span className="recommendation-arrow">→</span>
          <span className="team-name">{team}</span>
        </div>
        <div className="score-section">
          <span className="score">{score.toFixed(1)}/100</span>
          <button
            className={`explain-btn ${isOpen ? "active" : ""}`}
            onClick={() => setIsOpen((value) => !value)}
          >
            {isOpen ? "Hide Details" : "[Explain]"}
          </button>
        </div>
      </div>

      <div className="recommendation-summary">
        <div className="summary-role-row">
          <span className="summary-role-tag">{prettyTag(role)}</span>
          {row?.domestic_team ? <span className="summary-origin-tag">{row.domestic_team}</span> : null}
        </div>
        <p className="recommendation-reason">{summaryText}</p>
        {strongestReasons.length ? (
          <div className="reason-chip-row">
            {strongestReasons.map((item) => (
              <span key={item} className="reason-chip">
                {item}
              </span>
            ))}
          </div>
        ) : null}
      </div>

      {isOpen ? (
        <div className="explanation-panel">
          <h4 className="explanation-title">Why {player} is recommended</h4>

          <div className="explanation-metric-grid">
            <div className="explanation-metric-card">
              <span>Team Need</span>
              <strong>
                {Number(row?.current_role_count || 0)} / {Number(row?.target_role_count || 0)} {prettyTag(role)}
              </strong>
              <small>{Number(row?.role_deficit || 0)} missing slot{Number(row?.role_deficit || 0) === 1 ? "" : "s"}</small>
            </div>
            <div className="explanation-metric-card">
              <span>Benchmark Match</span>
              <strong>{row?.closest_ipl_benchmark || "N/A"}</strong>
              <small>{row?.closest_ipl_benchmark_team || "IPL reference"}</small>
            </div>
            <div className="explanation-metric-card">
              <span>Archetype</span>
              <strong>{prettyTag(row?.archetype_role || role)}</strong>
              <small>{prettyTag(row?.cluster_label || "cluster not available")}</small>
            </div>
          </div>

          <div className="features-breakdown">
            {featureRows.map((item) => {
              const barWidth = Math.min(Math.max(item.value, 0), 100);
              return (
                <div key={item.label} className="feature-row">
                  <div className="feature-name-block">
                    <div className="feature-name">{item.label}</div>
                    <div className="feature-note">{item.note}</div>
                  </div>
                  <div className="feature-bar-container">
                    <div className="feature-bar-track">
                      <div className="feature-bar-fill" style={{ width: `${barWidth}%` }} />
                    </div>
                  </div>
                  <div className="feature-contribution">{compact(item.value)}</div>
                </div>
              );
            })}
          </div>

          <div className="explanation-footer">
            <p className="role-info">
              Recommended for <strong>{prettyTag(role)}</strong> because the team still has a shortage in that role and this player combines model suitability, form, and role-profile alignment.
            </p>
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default ExplainabilityCard;
