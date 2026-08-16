import { useState } from "react";

const BOWLING_TEMPLATE = {
  economy_rate: 7.2,
  bowling_average: 20,
  bowling_strike_rate: 15,
  avg_wickets_per_match: 1.4,
  bowling_dot_rate: 0.42,
  two_plus_haul_rate: 0.3,
  pp_economy: 6.9,
  mid_economy: 7.1,
  death_economy: 8.4,
  pp_workload_pct: 0.25,
  mid_workload_pct: 0.45,
  death_workload_pct: 0.3,
  sample_reliability_score: 0.8,
};

export default function InferencePanel({ domains, onInferExisting, onInferCustom, inferenceResult }) {
  const [domain, setDomain] = useState("batting");
  const [player, setPlayer] = useState("RM Patidar");

  return (
    <section className="panel-grid two-column">
      <div className="glass-card">
        <h3>Live Model Inference</h3>
        <div className="field-grid">
          <label>
            Domain
            <select value={domain} onChange={(e) => setDomain(e.target.value)}>
              {domains.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label>
            Player Name
            <input value={player} onChange={(e) => setPlayer(e.target.value)} placeholder="Enter player" />
          </label>
        </div>
        <div className="action-row">
          <button className="primary-button" onClick={() => onInferExisting(domain, player)}>
            Score Existing Player
          </button>
          <button className="secondary-button" onClick={() => onInferCustom("bowling", BOWLING_TEMPLATE)}>
            Run Custom Bowling Sample
          </button>
        </div>
      </div>

      <div className="glass-card spotlight-card">
        <h3>Inference Result</h3>
        {inferenceResult ? (
          <>
            <div className="spotlight-score">{Number(inferenceResult.suitability_score).toFixed(2)}</div>
            <div className="spotlight-label">
              {inferenceResult.player || "Custom Input"} • {inferenceResult.domain}
            </div>
            <div className="spotlight-grid">
              <div>
                <span className="mini-label">Probability</span>
                <strong>{(Number(inferenceResult.predicted_probability) * 100).toFixed(2)}%</strong>
              </div>
              <div>
                <span className="mini-label">Archetype</span>
                <strong>{inferenceResult.archetype_role || "Unassigned"}</strong>
              </div>
              <div>
                <span className="mini-label">Cluster</span>
                <strong>{inferenceResult.cluster_label || "N/A"}</strong>
              </div>
            </div>
          </>
        ) : (
          <p className="muted-copy">Run a live inference call to see model scoring and archetype output.</p>
        )}
      </div>
    </section>
  );
}
