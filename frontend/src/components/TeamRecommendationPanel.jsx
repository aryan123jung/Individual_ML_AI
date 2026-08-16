function scoreColor(score) {
  if (score >= 90) return "elite";
  if (score >= 80) return "strong";
  if (score >= 70) return "good";
  return "watch";
}

export default function TeamRecommendationPanel({
  teams,
  selectedTeam,
  onSelectTeam,
  onGenerate,
  recommendations,
  hasRequested,
  isLoading,
}) {
  return (
    <section className="panel-grid two-column">
      <div className="glass-card">
        <div className="card-title-row">
          <h3>Franchise Focus</h3>
          <span className="subtle-tag">Step 1</span>
        </div>
        <p className="panel-copy">
          Choose a franchise first. The system will only generate domestic recommendations after you ask for them.
        </p>
        <div className="team-chip-grid">
          {teams.map((team) => (
            <button
              key={team}
              className={`team-chip ${selectedTeam === team ? "active" : ""}`}
              onClick={() => onSelectTeam(team)}
            >
              {team}
            </button>
          ))}
        </div>
        <div className="action-row top-gap">
          <button
            className="primary-button"
            onClick={onGenerate}
            disabled={!selectedTeam || isLoading}
          >
            {isLoading ? "Loading..." : "Generate Recommendations"}
          </button>
        </div>
      </div>

      <div className="glass-card">
        <div className="card-title-row">
          <h3>{selectedTeam || "Team"} Recommendation Board</h3>
          <span className="subtle-tag">{hasRequested ? `${recommendations.length} shown` : "Step 2"}</span>
        </div>
        {!hasRequested ? (
          <div className="empty-state">
            <strong>No recommendations shown yet</strong>
            <p>
              Select a team on the left and click <em>Generate Recommendations</em> to see the domestic player shortlist.
            </p>
          </div>
        ) : (
          <div className="recommendation-list">
            {recommendations.map((item) => (
              <article key={`${item.domestic_player}-${item.target_role}`} className="recommendation-row">
                <div>
                  <div className="player-name">{item.domestic_player}</div>
                  <div className="player-meta">
                    {item.target_role} • {item.domestic_team || "Domestic squad"}
                  </div>
                </div>
                <div className="recommendation-aside">
                  <span className={`score-pill ${scoreColor(item.final_recommendation_score)}`}>
                    {Number(item.final_recommendation_score).toFixed(2)}
                  </span>
                  <span className="mini-note">{item.archetype_role || item.recommendation_type}</span>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
