const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }

  return response.json();
}

export const api = {
  getTeams: () => request("/teams"),
  getTeamSummary: () => request("/recommendations/teams/summary"),
  getRecommendationHealth: () => request("/recommendations/health"),
  getSystemSummary: () => request("/system/summary"),
  getTeamDetail: (team) => request(`/teams/${encodeURIComponent(team)}`),
  getTeamRecommendations: (team, recommendationType, limit = 12) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (recommendationType) {
      params.set("recommendation_type", recommendationType);
    }
    return request(`/recommendations/teams/${encodeURIComponent(team)}?${params.toString()}`);
  },
  getPlayerProfile: (player) => request(`/players/${encodeURIComponent(player)}`),
  comparePlayers: (playerA, playerB) => {
    const params = new URLSearchParams({
      player_a: playerA,
      player_b: playerB,
    });
    return request(`/players/compare/?${params.toString()}`);
  },
  getEvaluationMetrics: () => request("/evaluation/metrics"),
  getInferenceDomains: () => request("/inference/domains"),
  inferExistingPlayer: (domain, player) =>
    request(`/inference/${encodeURIComponent(domain)}/players/${encodeURIComponent(player)}`),
  inferCustomPlayer: (domain, featureValues) =>
    request(`/inference/${encodeURIComponent(domain)}/custom`, {
      method: "POST",
      body: JSON.stringify({ feature_values: featureValues }),
    }),
};
