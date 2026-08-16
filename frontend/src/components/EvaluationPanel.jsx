function pct(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function Bar({ label, value, tone = "default" }) {
  return (
    <div className="bar-row">
      <div className="bar-label-row">
        <span>{label}</span>
        <strong>{pct(value)}</strong>
      </div>
      <div className="bar-track">
        <div className={`bar-fill tone-${tone}`} style={{ width: `${Math.max(4, Number(value) * 100)}%` }} />
      </div>
    </div>
  );
}

export default function EvaluationPanel({ comparisonRows, timeRows }) {
  return (
    <section className="panel-grid two-column">
      <div className="glass-card">
        <h3>Random Split Evaluation</h3>
        {comparisonRows.map((row) => (
          <div key={row.domain} className="evaluation-block">
            <div className="evaluation-title">{row.domain}</div>
            <Bar label="Accuracy" value={row.accuracy_random} tone="gold" />
            <Bar label="F1 Score" value={row.f1_score_random} tone="green" />
            <Bar label="ROC-AUC" value={row.roc_auc_random} tone="ink" />
          </div>
        ))}
      </div>

      <div className="glass-card">
        <h3>Time-Based Validation</h3>
        {timeRows.map((row) => (
          <div key={row.domain} className="evaluation-block">
            <div className="evaluation-title-row">
              <span className="evaluation-title">{row.domain}</span>
              <span className="subtle-tag">{row.test_seasons}</span>
            </div>
            <Bar label="Accuracy" value={row.accuracy} tone="gold" />
            <Bar label="F1 Score" value={row.f1_score} tone="green" />
            <Bar label="ROC-AUC" value={row.roc_auc} tone="ink" />
          </div>
        ))}
      </div>
    </section>
  );
}
