import { useEffect, useMemo, useState } from "react";
import { api } from "./services/api";
import ExplainabilityCard from "./components/ExplainabilityCard";

const NAV_ITEMS = [
  { id: "home", label: "Home" },
  { id: "teamlab", label: "Team Lab" },
  { id: "players", label: "Player Profiles" },
  { id: "compare", label: "Player Compare" },
  { id: "testing", label: "Model Testing" },
];

const RECOMMENDATION_TYPE_OPTIONS = [
  { id: "batting", label: "Batters" },
  { id: "bowling", label: "Bowlers" },
  { id: "allrounder", label: "Allrounders" },
];

const PLAYER_SEARCH_MODES = [
  { id: "domestic", label: "Domestic Player" },
  { id: "ipl", label: "IPL Player" },
];

const UNDERPERFORMANCE_VIEWS = [
  { id: "combined", label: "Combined View" },
  { id: "current", label: "Current Season" },
  { id: "trend", label: "Multi-Season Trend" },
];

const VISUAL_BREAKDOWN_LENSES = [
  { id: "career", label: "Overall IPL Career" },
  { id: "season", label: "This Season" },
];

function pct(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function compact(value) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toFixed(2) : "0.00";
}

function prettyLabel(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function renderStatBlock(title, record) {
  if (!record) return null;
  const entries = Object.entries(record).filter(([, value]) => value !== null && value !== undefined && value !== "");
  if (!entries.length) return null;

  return (
    <div className="detail-block" key={title}>
      <div className="subsection-title">{title}</div>
      <div className="detail-grid">
        {entries.map(([key, value]) => (
          <div key={key} className="detail-item">
            <span>{prettyLabel(key)}</span>
            <strong>{typeof value === "number" ? compact(value) : String(value)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function buildStatBlocksForEntity(entity, profile) {
  if (!entity) return [];

  if (entity.kind === "ipl_squad") {
    return [
      { title: "Current Season IPL Stats", record: profile?.current_squad_profile?.[0] },
      { title: "Historical IPL Batting Features", record: profile?.ipl_historical_profiles?.batting },
      { title: "Historical IPL Bowling Features", record: profile?.ipl_historical_profiles?.bowling },
      { title: "Historical IPL Allrounder Features", record: profile?.ipl_historical_profiles?.allrounder },
      { title: "Player Bio", record: profile?.player_master },
    ].filter((item) => item.record);
  }

  return [
    { title: "Domestic Batting Features", record: profile?.domestic_profiles?.batting },
    { title: "Domestic Bowling Features", record: profile?.domestic_profiles?.bowling },
    { title: "Domestic Allrounder Features", record: profile?.domestic_profiles?.allrounder },
    { title: "Similar IPL Comparison", record: profile?.similarity_matches?.[0] },
  ].filter((item) => item.record);
}

function buildPlayerProfileBlocks(profile) {
  if (!profile) return [];
  return [
    { title: "Player Bio", record: profile.player_master },
    { title: "Current IPL Season Profile", record: profile.current_squad_profile?.[0] },
    { title: "Domestic Batting Features", record: profile.domestic_profiles?.batting },
    { title: "Domestic Bowling Features", record: profile.domestic_profiles?.bowling },
    { title: "Domestic Allrounder Features", record: profile.domestic_profiles?.allrounder },
    { title: "Historical IPL Batting Features", record: profile.ipl_historical_profiles?.batting },
    { title: "Historical IPL Bowling Features", record: profile.ipl_historical_profiles?.bowling },
    { title: "Historical IPL Allrounder Features", record: profile.ipl_historical_profiles?.allrounder },
  ].filter((item) => item.record);
}

function prettyTag(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function asNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function firstNonZero(...values) {
  for (const value of values) {
    const number = Number(value);
    if (Number.isFinite(number) && number !== 0) return number;
  }
  return asNumber(values[0]);
}

function buildProfileChartMetrics(profile, fallbackRow = null) {
  if (!profile && !fallbackRow) return [];
  const squad = profile?.current_squad_profile?.[0] || {};
  const domesticBat = profile?.domestic_profiles?.batting || {};
  const domesticBowl = profile?.domestic_profiles?.bowling || {};
  const domesticAll = profile?.domestic_profiles?.allrounder || {};
  const iplBat = profile?.ipl_historical_profiles?.batting || {};
  const iplBowl = profile?.ipl_historical_profiles?.bowling || {};
  const recommendation = fallbackRow || profile?.recommendations?.[0] || {};

  return [
    ["Runs", firstNonZero(squad.runs, domesticBat.total_runs, iplBat.total_runs, fallbackRow?.runs)],
    ["Avg Runs", firstNonZero(domesticBat.avg_runs, iplBat.avg_runs, squad.runs && squad.matches_played ? squad.runs / squad.matches_played : 0)],
    ["Strike Rate", firstNonZero(squad.batting_strike_rate, domesticBat.strike_rate, iplBat.strike_rate, fallbackRow?.strike_rate)],
    ["Wickets", firstNonZero(squad.wickets, domesticBowl.total_wickets, iplBowl.total_wickets, domesticAll.total_wickets, fallbackRow?.wickets)],
    ["Economy", firstNonZero(squad.bowling_economy, domesticBowl.economy_rate, iplBowl.economy_rate, domesticAll.economy_rate, fallbackRow?.economy)],
    ["Reliability", firstNonZero(domesticBat.sample_reliability_score ? domesticBat.sample_reliability_score * 100 : 0, domesticBowl.sample_reliability_score ? domesticBowl.sample_reliability_score * 100 : 0, fallbackRow?.reliability_score, fallbackRow?.sample_reliability_score ? fallbackRow.sample_reliability_score * 100 : 0)],
    ["Quality", firstNonZero(recommendation.quality_score, fallbackRow?.quality_score)],
    ["Recent Form", firstNonZero(recommendation.recent_form_score, fallbackRow?.recent_form_score)],
    ["ML Fit", firstNonZero(recommendation.ml_suitability_score, fallbackRow?.ml_suitability_score)],
  ];
}

function sumHistoryValues(rows, key) {
  return (rows || []).reduce((sum, row) => sum + asNumber(row?.[key]), 0);
}

function safeRate(numerator, denominator, multiplier = 1) {
  const den = asNumber(denominator);
  if (!den) return null;
  return Number(((asNumber(numerator) * multiplier) / den).toFixed(2));
}

function aggregateIplCareerProfile(profile) {
  const battingRows = profile?.ipl_historical_history?.batting || [];
  const bowlingRows = profile?.ipl_historical_history?.bowling || [];

  const batting = battingRows.length
    ? {
        total_runs: sumHistoryValues(battingRows, "total_runs"),
        total_matches: sumHistoryValues(battingRows, "total_matches"),
        total_balls_faced: sumHistoryValues(battingRows, "total_balls_faced"),
        avg_runs: safeRate(sumHistoryValues(battingRows, "total_runs"), sumHistoryValues(battingRows, "total_matches")),
        strike_rate: safeRate(sumHistoryValues(battingRows, "total_runs"), sumHistoryValues(battingRows, "total_balls_faced"), 100),
        pp_strike_rate: safeRate(sumHistoryValues(battingRows, "pp_runs"), sumHistoryValues(battingRows, "pp_balls"), 100),
        mid_strike_rate: safeRate(sumHistoryValues(battingRows, "mid_runs"), sumHistoryValues(battingRows, "mid_balls"), 100),
        death_strike_rate: safeRate(sumHistoryValues(battingRows, "death_runs"), sumHistoryValues(battingRows, "death_balls"), 100),
      }
    : {};

  const bowling = bowlingRows.length
    ? {
        total_wickets: sumHistoryValues(bowlingRows, "total_wickets"),
        total_matches_bowled: sumHistoryValues(bowlingRows, "total_matches_bowled"),
        total_balls_bowled: sumHistoryValues(bowlingRows, "total_balls_bowled"),
        total_runs_conceded: sumHistoryValues(bowlingRows, "total_runs_conceded"),
        avg_wickets_per_match: safeRate(sumHistoryValues(bowlingRows, "total_wickets"), sumHistoryValues(bowlingRows, "total_matches_bowled")),
        economy_rate: safeRate(sumHistoryValues(bowlingRows, "total_runs_conceded"), sumHistoryValues(bowlingRows, "total_balls_bowled"), 6),
        pp_economy: safeRate(sumHistoryValues(bowlingRows, "pp_runs_conceded"), sumHistoryValues(bowlingRows, "pp_balls_bowled"), 6),
        mid_economy: safeRate(sumHistoryValues(bowlingRows, "mid_runs_conceded"), sumHistoryValues(bowlingRows, "mid_balls_bowled"), 6),
        death_economy: safeRate(sumHistoryValues(bowlingRows, "death_runs_conceded"), sumHistoryValues(bowlingRows, "death_balls_bowled"), 6),
      }
    : {};

  return {
    ...profile,
    current_squad_profile: [],
    domestic_profiles: {},
    domestic_history: {},
    ipl_historical_profiles: {
      batting,
      bowling,
      allrounder: {},
    },
  };
}

function buildPhaseMetrics(profile, fallbackRow = null) {
  const squad = profile?.current_squad_profile?.[0] || {};
  const domesticBat = profile?.domestic_profiles?.batting || {};
  const domesticBowl = profile?.domestic_profiles?.bowling || {};
  const iplBat = profile?.ipl_historical_profiles?.batting || {};
  const iplBowl = profile?.ipl_historical_profiles?.bowling || {};
  const recommendation = fallbackRow || {};
  return {
    batting: [
      ["Powerplay", firstNonZero(domesticBat.pp_strike_rate, iplBat.pp_strike_rate, squad.batting_strike_rate, recommendation.strike_rate)],
      ["Middle", firstNonZero(domesticBat.mid_strike_rate, iplBat.mid_strike_rate, domesticBat.strike_rate, iplBat.strike_rate, squad.batting_strike_rate, recommendation.strike_rate)],
      ["Death", firstNonZero(domesticBat.death_strike_rate, iplBat.death_strike_rate, domesticBat.strike_rate, iplBat.strike_rate, squad.batting_strike_rate, recommendation.strike_rate)],
    ],
    bowling: [
      ["Powerplay", firstNonZero(domesticBowl.pp_economy, iplBowl.pp_economy, squad.bowling_economy, fallbackRow?.economy)],
      ["Middle", firstNonZero(domesticBowl.mid_economy, iplBowl.mid_economy, domesticBowl.economy_rate, iplBowl.economy_rate, squad.bowling_economy, fallbackRow?.economy)],
      ["Death", firstNonZero(domesticBowl.death_economy, iplBowl.death_economy, domesticBowl.economy_rate, iplBowl.economy_rate, squad.bowling_economy, fallbackRow?.economy)],
    ],
  };
}

function MetricBars({ title, metrics, inverseKeys = [] }) {
  const filtered = metrics.filter(([, value]) => Number.isFinite(Number(value)) && Number(value) > 0);
  if (!filtered.length) return null;
  const maxValue = Math.max(...filtered.map(([, value]) => Math.abs(Number(value))), 1);
  return (
    <div className="detail-block chart-block">
      <div className="subsection-title">{title}</div>
      <div className="mini-chart-stack">
        {filtered.map(([label, value]) => {
          const number = Number(value);
          const width = `${Math.max((Math.abs(number) / maxValue) * 100, 6)}%`;
          const isInverse = inverseKeys.includes(label);
          return (
            <div key={label} className="mini-chart-row">
              <div className="mini-chart-head">
                <span>{label}</span>
                <strong>{compact(number)}</strong>
              </div>
              <div className="mini-chart-lane">
                <div className={`mini-chart-fill ${isInverse ? "inverse" : ""}`} style={{ width }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TrendColumns({ title, metrics, inverse = false }) {
  const filtered = metrics.filter(([, value]) => Number.isFinite(Number(value)) && Number(value) > 0);
  if (!filtered.length) return null;
  const maxValue = Math.max(...filtered.map(([, value]) => Math.abs(Number(value))), 1);
  return (
    <div className="detail-block chart-block">
      <div className="subsection-title">{title}</div>
      <div className="phase-columns">
        {filtered.map(([label, value]) => {
          const number = Number(value);
          const height = `${Math.max((Math.abs(number) / maxValue) * 100, 12)}%`;
          return (
            <div key={label} className="phase-column-card">
              <div className={`phase-column-fill ${inverse ? "inverse" : ""}`} style={{ height }} />
              <strong>{compact(number)}</strong>
              <span>{label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function buildSeasonTrendSeries(profile) {
  const sources = [
    {
      title: "IPL Batting Trend",
      rows: profile?.ipl_historical_history?.batting || [],
      valueKey: "avg_runs",
      secondaryKey: "strike_rate",
      valueLabel: "Avg Runs",
      secondaryLabel: "Strike Rate",
      color: "green",
    },
    {
      title: "Domestic Batting Trend",
      rows: profile?.domestic_history?.batting || [],
      valueKey: "avg_runs",
      secondaryKey: "strike_rate",
      valueLabel: "Avg Runs",
      secondaryLabel: "Strike Rate",
      color: "green",
    },
    {
      title: "IPL Bowling Trend",
      rows: profile?.ipl_historical_history?.bowling || [],
      valueKey: "avg_wickets_per_match",
      secondaryKey: "economy_rate",
      valueLabel: "Wkts/Match",
      secondaryLabel: "Economy",
      color: "gold",
    },
    {
      title: "Domestic Bowling Trend",
      rows: profile?.domestic_history?.bowling || [],
      valueKey: "avg_wickets_per_match",
      secondaryKey: "economy_rate",
      valueLabel: "Wkts/Match",
      secondaryLabel: "Economy",
      color: "gold",
    },
  ];

  for (const source of sources) {
    const points = source.rows
      .filter((row) => row.season_start_year)
      .map((row) => ({
        label: String(row.season_start_year),
        value: asNumber(row[source.valueKey]),
        secondary: asNumber(row[source.secondaryKey]),
      }))
      .filter((point) => point.value > 0 || point.secondary > 0);
    if (points.length >= 2) {
      return { ...source, points };
    }
  }
  return null;
}

function buildBenchmarkBars(profile, fallbackRow = null) {
  const roleBenchmarks = profile?.role_benchmarks || {};
  const domesticBat = profile?.domestic_profiles?.batting || {};
  const domesticBowl = profile?.domestic_profiles?.bowling || {};
  const iplBat = profile?.ipl_historical_profiles?.batting || {};
  const iplBowl = profile?.ipl_historical_profiles?.bowling || {};
  const row = fallbackRow || {};
  const bars = [];

  if (roleBenchmarks.batting) {
    bars.push(
      ["Avg Runs", firstNonZero(domesticBat.avg_runs, iplBat.avg_runs, row.avg_runs), asNumber(roleBenchmarks.batting.avg_runs_median)],
      ["Strike Rate", firstNonZero(domesticBat.strike_rate, iplBat.strike_rate, row.strike_rate), asNumber(roleBenchmarks.batting.strike_rate_median)],
      ["Boundary Rate", firstNonZero(domesticBat.boundary_rate, iplBat.boundary_rate), asNumber(roleBenchmarks.batting.boundary_rate_median)],
    );
  }
  if (roleBenchmarks.bowling) {
    bars.push(
      ["Economy", firstNonZero(domesticBowl.economy_rate, iplBowl.economy_rate, row.economy), asNumber(roleBenchmarks.bowling.economy_rate_median)],
      ["Wkts/Match", firstNonZero(domesticBowl.avg_wickets_per_match, iplBowl.avg_wickets_per_match, row.avg_wickets_per_match), asNumber(roleBenchmarks.bowling.avg_wickets_per_match_median)],
      ["Dot Rate", firstNonZero(domesticBowl.bowling_dot_rate, iplBowl.bowling_dot_rate), asNumber(roleBenchmarks.bowling.bowling_dot_rate_median)],
    );
  }
  return bars.filter(([, playerValue, benchmarkValue]) => playerValue > 0 || benchmarkValue > 0);
}

function TrendLineChart({ series, sharedLabels = null }) {
  if (!series?.points?.length) return null;
  const width = 560;
  const height = 240;
  const padding = 34;
  const labelStride = Math.max(1, Math.ceil(series.points.length / 8));
  const values = series.points.map((point) => point.value);
  const maxValue = Math.max(...values, 1);
  const minValue = Math.min(...values, 0);
  const range = Math.max(maxValue - minValue, 1);
  const polyline = series.points
    .map((point, index) => {
      const x = padding + (index * (width - padding * 2)) / (series.points.length - 1);
      const y = height - padding - ((point.value - minValue) / range) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="detail-block chart-block">
      <div className="subsection-title">{series.title}</div>
      <svg viewBox={`0 0 ${width} ${height}`} className="line-chart-svg" role="img" aria-label={series.title}>
        {[0, 0.5, 1].map((ratio) => {
          const y = height - padding - ratio * (height - padding * 2);
          return <line key={ratio} x1={padding} y1={y} x2={width - padding} y2={y} className="line-chart-grid" />;
        })}
        <polyline points={polyline} className={`line-chart-path ${series.color || "green"}`} />
        {series.points.map((point, index) => {
          const x = padding + (index * (width - padding * 2)) / (series.points.length - 1);
          const y = height - padding - ((point.value - minValue) / range) * (height - padding * 2);
          const showLabel = sharedLabels
            ? sharedLabels.has(String(point.label))
            : index === 0 ||
              index === series.points.length - 1 ||
              index % labelStride === 0;
          return (
            <g key={`${point.label}-${index}`}>
              <circle cx={x} cy={y} r="4.5" className={`line-chart-point ${series.color || "green"}`} />
              {showLabel ? (
                <>
                  <line x1={x} y1={height - padding + 4} x2={x} y2={height - padding + 10} className="line-chart-tick" />
                  <text x={x} y={height - 6} className="line-chart-label" textAnchor="middle">{point.label}</text>
                </>
              ) : null}
            </g>
          );
        })}
      </svg>
      <div className="chart-footnote">
        <span>{series.valueLabel}</span>
        <strong>with {series.secondaryLabel}</strong>
      </div>
    </div>
  );
}

function BenchmarkComparisonChart({ rows }) {
  if (!rows?.length) return null;
  return (
    <div className="detail-block chart-block">
      <div className="subsection-title">Player vs Role Benchmark</div>
      <div className="benchmark-stack">
        {rows.map(([label, playerValue, benchmarkValue]) => {
          const maxValue = Math.max(playerValue, benchmarkValue, 1);
          return (
            <div key={label} className="benchmark-row">
              <div className="benchmark-head">
                <strong>{label}</strong>
                <span>{compact(playerValue)} vs {compact(benchmarkValue)}</span>
              </div>
              <div className="benchmark-bars">
                <div className="benchmark-bar-group">
                  <div className="benchmark-label">Player</div>
                  <div className="comparison-lane"><div className="comparison-fill left" style={{ width: `${(playerValue / maxValue) * 100}%` }} /></div>
                </div>
                <div className="benchmark-bar-group">
                  <div className="benchmark-label">Benchmark</div>
                  <div className="comparison-lane"><div className="comparison-fill right" style={{ width: `${(benchmarkValue / maxValue) * 100}%` }} /></div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ProfileCharts({ profile, fallbackRow = null, title = "Performance Charts" }) {
  const metrics = buildProfileChartMetrics(profile, fallbackRow);
  const phaseMetrics = buildPhaseMetrics(profile, fallbackRow);
  const keyMetrics = metrics.slice(0, 6);
  const modelMetrics = metrics.slice(6);
  const seasonSeries = buildSeasonTrendSeries(profile);
  const benchmarkBars = buildBenchmarkBars(profile, fallbackRow);
  const hasAny = [...keyMetrics, ...modelMetrics, ...phaseMetrics.batting, ...phaseMetrics.bowling].some(([, value]) => Number(value) > 0);
  if (!hasAny) return null;
  return (
    <div className="profile-chart-section">
      <div className="panel-head">
        <h3>{title}</h3>
        <span className="status-chip">Visual analysis</span>
      </div>
      <div className="profile-chart-grid">
        <MetricBars title="Core Metrics" metrics={keyMetrics} inverseKeys={["Economy"]} />
        <MetricBars title="Model Metrics" metrics={modelMetrics} />
        <TrendColumns title="Batting Phase Impact" metrics={phaseMetrics.batting} />
        <TrendColumns title="Bowling Phase Control" metrics={phaseMetrics.bowling} inverse />
        <TrendLineChart series={seasonSeries} />
        <BenchmarkComparisonChart rows={benchmarkBars} />
      </div>
    </div>
  );
}

function buildVisualLensProfile(profile, mode) {
  if (!profile) return null;
  if (mode === "season") {
    return {
      ...profile,
      domestic_profiles: {},
      domestic_history: {},
      ipl_historical_profiles: {},
      ipl_historical_history: {},
    };
  }
  return aggregateIplCareerProfile(profile);
}

function buildVisualLensFallback(profile, summary, mode) {
  if (mode === "season") {
    const squad = profile?.current_squad_profile?.[0] || {};
    return {
      runs: squad.runs,
      strike_rate: squad.batting_strike_rate,
      wickets: squad.wickets,
      economy: squad.bowling_economy,
      quality_score: summary?.quality_score,
      recent_form_score: summary?.recent_form_score,
      ml_suitability_score: summary?.ml_suitability_score,
      reliability_score: summary?.sample_reliability_score ? Number(summary.sample_reliability_score) * 100 : 0,
    };
  }
  const iplBat = aggregateIplCareerProfile(profile)?.ipl_historical_profiles?.batting || {};
  const iplBowl = aggregateIplCareerProfile(profile)?.ipl_historical_profiles?.bowling || {};
  return {
    runs: iplBat.total_runs,
    avg_runs: iplBat.avg_runs,
    strike_rate: iplBat.strike_rate,
    wickets: iplBowl.total_wickets,
    avg_wickets_per_match: iplBowl.avg_wickets_per_match,
    economy: iplBowl.economy_rate,
    quality_score: summary?.quality_score,
    recent_form_score: summary?.recent_form_score,
    ml_suitability_score: summary?.ml_suitability_score,
    reliability_score: summary?.sample_reliability_score ? Number(summary.sample_reliability_score) * 100 : 0,
  };
}

function VisualBreakdownCompare({ leftProfile, rightProfile, leftSummary, rightSummary }) {
  const [visualLens, setVisualLens] = useState("career");
  const leftLensProfile = buildVisualLensProfile(leftProfile, visualLens);
  const rightLensProfile = buildVisualLensProfile(rightProfile, visualLens);
  const leftLensFallback = buildVisualLensFallback(leftProfile, leftSummary, visualLens);
  const rightLensFallback = buildVisualLensFallback(rightProfile, rightSummary, visualLens);

  const leftPhase = buildPhaseMetrics(leftLensProfile, leftLensFallback);
  const rightPhase = buildPhaseMetrics(rightLensProfile, rightLensFallback);
  const leftSeries = visualLens === "career" ? buildSeasonTrendSeries(leftLensProfile) : null;
  const rightSeries = visualLens === "career" ? buildSeasonTrendSeries(rightLensProfile) : null;
  const sharedTrendLabels = useMemo(() => {
    if (!leftSeries?.points?.length) return null;
    const points = leftSeries.points;
    const stride = Math.max(1, Math.ceil(points.length / 8));
    return new Set(
      points
        .filter((point, index) => index === 0 || index === points.length - 1 || index % stride === 0)
        .map((point) => String(point.label))
    );
  }, [leftSeries]);
  const leftBars = buildBenchmarkBars(leftLensProfile, leftLensFallback);
  const rightBars = buildBenchmarkBars(rightLensProfile, rightLensFallback);

  const leftMetrics = buildProfileChartMetrics(leftLensProfile, leftLensFallback);
  const rightMetrics = buildProfileChartMetrics(rightLensProfile, rightLensFallback);
  const leftCore = leftMetrics.slice(0, 6);
  const rightCore = rightMetrics.slice(0, 6);
  const leftModel = leftMetrics.slice(6);
  const rightModel = rightMetrics.slice(6);

  return (
    <div className="plain-card comparison-section-card">
      <div className="panel-head">
        <h3>Visual Breakdown</h3>
        <span className="status-chip">{visualLens === "season" ? "Current season" : "IPL career"}</span>
      </div>

      <div className="type-switcher">
        {VISUAL_BREAKDOWN_LENSES.map((option) => (
          <button
            key={option.id}
            className={`type-switch ${visualLens === option.id ? "active" : ""}`}
            onClick={() => setVisualLens(option.id)}
          >
            {option.label}
          </button>
        ))}
      </div>

      <div className="comparison-chart-players">
        <strong>{leftSummary.player}</strong>
        <strong>{rightSummary.player}</strong>
      </div>

      <div className="comparison-chart-grid">
        <MetricBars title="Core Metrics" metrics={leftCore} inverseKeys={["Economy"]} />
        <MetricBars title="Core Metrics" metrics={rightCore} inverseKeys={["Economy"]} />

        <MetricBars title="Model Metrics" metrics={leftModel} />
        <MetricBars title="Model Metrics" metrics={rightModel} />

        <TrendColumns title="Batting Phase Impact" metrics={leftPhase.batting} />
        <TrendColumns title="Batting Phase Impact" metrics={rightPhase.batting} />

        <TrendColumns title="Bowling Phase Control" metrics={leftPhase.bowling} inverse />
        <TrendColumns title="Bowling Phase Control" metrics={rightPhase.bowling} inverse />

        {leftSeries ? <TrendLineChart series={leftSeries} sharedLabels={sharedTrendLabels} /> : <div className="detail-block chart-block chart-empty-state"><div className="subsection-title">IPL Trend</div><p>This view uses current-season IPL output only.</p></div>}
        {rightSeries ? <TrendLineChart series={rightSeries} sharedLabels={sharedTrendLabels} /> : <div className="detail-block chart-block chart-empty-state"><div className="subsection-title">IPL Trend</div><p>This view uses current-season IPL output only.</p></div>}

        <BenchmarkComparisonChart rows={leftBars} />
        <BenchmarkComparisonChart rows={rightBars} />
      </div>
    </div>
  );
}

function normalizeRadarValue(key, value) {
  const number = Number(value || 0);
  const caps = {
    batting_impact: 100,
    scoring_speed: 220,
    bowling_impact: 100,
    control: 100,
    reliability: 100,
    ml_fit: 100,
  };
  const cap = caps[key] || 100;
  return Math.max(0, Math.min(number / cap, 1));
}

function RadarComparison({ left, right }) {
  const metricEntries = Object.entries(left?.radar_metrics || {});
  if (!metricEntries.length || !right?.radar_metrics) return null;
  const size = 300;
  const center = size / 2;
  const radius = 102;
  const levels = [0.25, 0.5, 0.75, 1];

  const pointsFor = (metrics) =>
    metricEntries
      .map(([key], index) => {
        const angle = (-Math.PI / 2) + (index / metricEntries.length) * Math.PI * 2;
        const value = normalizeRadarValue(key, metrics[key]);
        const x = center + Math.cos(angle) * radius * value;
        const y = center + Math.sin(angle) * radius * value;
        return `${x},${y}`;
      })
      .join(" ");

  return (
    <div className="radar-card">
      <div className="subsection-title">Radar Comparison</div>
      <svg viewBox={`0 0 ${size} ${size}`} className="radar-svg" role="img" aria-label="Player radar comparison">
        {levels.map((level) => (
          <polygon
            key={level}
            points={metricEntries
              .map((_, index) => {
                const angle = (-Math.PI / 2) + (index / metricEntries.length) * Math.PI * 2;
                const x = center + Math.cos(angle) * radius * level;
                const y = center + Math.sin(angle) * radius * level;
                return `${x},${y}`;
              })
              .join(" ")}
            className="radar-grid"
          />
        ))}
        {metricEntries.map(([key], index) => {
          const angle = (-Math.PI / 2) + (index / metricEntries.length) * Math.PI * 2;
          const x = center + Math.cos(angle) * radius;
          const y = center + Math.sin(angle) * radius;
          const labelX = center + Math.cos(angle) * (radius + 26);
          const labelY = center + Math.sin(angle) * (radius + 26);
          return (
            <g key={key}>
              <line x1={center} y1={center} x2={x} y2={y} className="radar-axis" />
              <text x={labelX} y={labelY} className="radar-label" textAnchor="middle">
                {prettyTag(key)}
              </text>
            </g>
          );
        })}
        <polygon points={pointsFor(left.radar_metrics)} className="radar-area radar-left" />
        <polygon points={pointsFor(right.radar_metrics)} className="radar-area radar-right" />
      </svg>
      <div className="radar-legend">
        <span><i className="legend-swatch left" />{left.player}</span>
        <span><i className="legend-swatch right" />{right.player}</span>
      </div>
    </div>
  );
}

function ComparisonBars({ left, right }) {
  const metrics = [
    ["Quality Score", "quality_score"],
    ["Recent Form", "recent_form_score"],
    ["ML Suitability", "ml_suitability_score"],
    ["Similarity", "similarity_score"],
    ["Strike Rate", "strike_rate"],
    ["Economy", "economy"],
  ];

  return (
    <div className="plain-card comparison-bars-card">
      <div className="panel-head">
        <h3>Metric Bars</h3>
        <span className="status-chip">Side-by-side</span>
      </div>
      <div className="comparison-bars">
        {metrics.map(([label, key]) => {
          const leftValue = Number(left?.[key] || 0);
          const rightValue = Number(right?.[key] || 0);
          const denominator = Math.max(Math.abs(leftValue), Math.abs(rightValue), 1);
          const leftWidth = `${(Math.abs(leftValue) / denominator) * 100}%`;
          const rightWidth = `${(Math.abs(rightValue) / denominator) * 100}%`;
          return (
            <div className="comparison-bar-row" key={key}>
              <div className="comparison-bar-head">
                <strong>{label}</strong>
                <span>{compact(leftValue)} vs {compact(rightValue)}</span>
              </div>
              <div className="comparison-lanes">
                <div className="comparison-lane">
                  <div className="comparison-fill left" style={{ width: leftWidth }} />
                </div>
                <div className="comparison-lane">
                  <div className="comparison-fill right" style={{ width: rightWidth }} />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ComparisonStatTable({ title, leftLabel, rightLabel, rows }) {
  const filteredRows = rows.filter((row) => row.left !== undefined || row.right !== undefined || row.leftText || row.rightText);
  if (!filteredRows.length) return null;

  return (
    <div className="plain-card comparison-stat-table">
      <div className="panel-head">
        <h3>{title}</h3>
        <span className="status-chip">A vs B</span>
      </div>
      <div className="comparison-stat-header">
        <span>Metric</span>
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
      <div className="comparison-stat-body">
        {filteredRows.map((row) => (
          <div key={row.label} className="comparison-stat-row">
            <span>{row.label}</span>
            <strong>{row.leftText ?? compact(row.left)}</strong>
            <strong>{row.rightText ?? compact(row.right)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function App() {
  const [activeView, setActiveView] = useState("home");
  const [teams, setTeams] = useState([]);
  const [selectedTeam, setSelectedTeam] = useState("");
  const [teamDetail, setTeamDetail] = useState(null);
  const [teamRecommendations, setTeamRecommendations] = useState([]);
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [selectedEntityProfile, setSelectedEntityProfile] = useState(null);
  const [selectedEntityLoading, setSelectedEntityLoading] = useState(false);
  const [showSelectedEntityModal, setShowSelectedEntityModal] = useState(false);
  const [recommendationRequested, setRecommendationRequested] = useState(false);
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [teamDetailLoading, setTeamDetailLoading] = useState(false);
  const [recommendationType, setRecommendationType] = useState("batting");
  const [underperformanceView, setUnderperformanceView] = useState("combined");

  const [domains, setDomains] = useState([]);
  const [playerSearchMode, setPlayerSearchMode] = useState("domestic");
  const [playerDomain, setPlayerDomain] = useState("batting");
  const [playerQuery, setPlayerQuery] = useState("RM Patidar");
  const [playerInference, setPlayerInference] = useState(null);
  const [playerProfile, setPlayerProfile] = useState(null);
  const [playerLoading, setPlayerLoading] = useState(false);
  const [showPlayerProfileModal, setShowPlayerProfileModal] = useState(false);
  const [comparePlayerA, setComparePlayerA] = useState("Rohit Sharma");
  const [comparePlayerB, setComparePlayerB] = useState("Suryakumar Yadav");
  const [comparisonData, setComparisonData] = useState(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);

  const [evaluation, setEvaluation] = useState(null);
  const [recommendationHealth, setRecommendationHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function initialize() {
      try {
        setLoading(true);
        const [teamsResponse, evaluationResponse, domainsResponse, healthResponse] = await Promise.all([
          api.getTeams(),
          api.getEvaluationMetrics(),
          api.getInferenceDomains(),
          api.getRecommendationHealth(),
        ]);
        setTeams(teamsResponse);
        const firstTeam = teamsResponse[0] || "";
        setSelectedTeam(firstTeam);
        setEvaluation(evaluationResponse);
        setDomains(domainsResponse.domains || []);
        setRecommendationHealth(healthResponse);
        if (firstTeam) {
          const detail = await api.getTeamDetail(firstTeam);
          setTeamDetail(detail);
        }
      } catch (err) {
        setError(String(err.message || err));
      } finally {
        setLoading(false);
      }
    }

    initialize();
  }, []);

  const testingComparison = useMemo(
    () => evaluation?.evaluation_mode_comparison || [],
    [evaluation]
  );
  const timeBasedMetrics = useMemo(
    () => evaluation?.time_based_metrics || [],
    [evaluation]
  );
  const recommendationChecks = useMemo(
    () => recommendationHealth?.checks || [],
    [recommendationHealth]
  );
  const recommendationSummary = useMemo(
    () => recommendationHealth?.summary || {},
    [recommendationHealth]
  );
  const teamGapSummary = useMemo(
    () => teamDetail?.gap_summary || { total_gap_roles: 0, total_missing_slots: 0, critical_gap_roles: 0 },
    [teamDetail]
  );
  const candidatePoolSummary = useMemo(
    () => recommendationHealth?.candidate_pool || [],
    [recommendationHealth]
  );
  const homepageMetrics = useMemo(() => {
    if (!testingComparison.length) return [];
    const batting = testingComparison.find((row) => row.domain === "batting");
    const bowling = testingComparison.find((row) => row.domain === "bowling");
    const allrounder = testingComparison.find((row) => row.domain === "allrounder");
    return [
      {
        label: "Teams Covered",
        value: teams.length,
        note: "IPL franchises available in the current API",
      },
      {
        label: "Batting Time-Based Accuracy",
        value: pct(batting?.accuracy_time || 0),
        note: "Realistic forward validation result",
      },
      {
        label: "Bowling Time-Based Accuracy",
        value: pct(bowling?.accuracy_time || 0),
        note: "Current production benchmark",
      },
      {
        label: "Allrounder Time-Based Accuracy",
        value: pct(allrounder?.accuracy_time || 0),
        note: "Highest current model performance",
      },
    ];
  }, [teams.length, testingComparison]);

  const filteredUnderperformers = useMemo(() => {
    const rows = teamDetail?.underperformers || [];
    if (underperformanceView === "current") {
      return [...rows]
        .filter((row) => Number(row.current_season_score ?? row.underperformance_score ?? 0) >= 12)
        .sort(
          (a, b) => Number(b.current_season_score ?? b.underperformance_score ?? 0) - Number(a.current_season_score ?? a.underperformance_score ?? 0)
        );
    }
    if (underperformanceView === "trend") {
      return [...rows]
        .filter((row) => Number(row.multi_season_trend_score ?? row.underperformance_score ?? 0) >= 12)
        .sort(
          (a, b) => Number(b.multi_season_trend_score ?? b.underperformance_score ?? 0) - Number(a.multi_season_trend_score ?? a.underperformance_score ?? 0)
        );
    }
    return [...rows]
      .filter((row) => Number(row.underperformance_score ?? 0) >= 12)
      .sort(
        (a, b) => Number(b.underperformance_score ?? 0) - Number(a.underperformance_score ?? 0)
      );
  }, [teamDetail, underperformanceView]);

  const filteredReplacements = useMemo(() => {
    const replacements = teamDetail?.replacements || [];
    const allowedPlayers = new Set(filteredUnderperformers.map((row) => row.player));
    return replacements.filter((row) => allowedPlayers.has(row.underperforming_player));
  }, [teamDetail, filteredUnderperformers]);

  const monitoringPlayers = useMemo(() => {
    const rows = teamDetail?.monitoring_players || [];
    if (underperformanceView === "current") {
      return [...rows]
        .sort(
          (a, b) => Number(b.current_season_score ?? b.underperformance_score ?? 0) - Number(a.current_season_score ?? a.underperformance_score ?? 0)
        );
    }
    if (underperformanceView === "trend") {
      return [...rows]
        .sort(
          (a, b) => Number(b.multi_season_trend_score ?? b.underperformance_score ?? 0) - Number(a.multi_season_trend_score ?? a.underperformance_score ?? 0)
        );
    }
    return [...rows]
      .sort(
        (a, b) => Number(b.underperformance_score ?? 0) - Number(a.underperformance_score ?? 0)
      );
  }, [teamDetail, underperformanceView]);

  const flaggedPlayersCount = (teamDetail?.underperformers?.length || 0) + (teamDetail?.monitoring_players?.length || 0);

  async function handleGenerateRecommendations(type = recommendationType) {
    if (!selectedTeam) return;
    try {
      setRecommendationLoading(true);
      const start = Date.now();
      const recommendations = await api.getTeamRecommendations(selectedTeam, type, 12);
      const elapsed = Date.now() - start;
      if (elapsed < 1200) {
        await new Promise((resolve) => setTimeout(resolve, 1200 - elapsed));
      }
      setTeamRecommendations(recommendations);
      if (recommendations[0]) {
        handleSelectEntity({ kind: "domestic_recommendation", data: recommendations[0] });
      } else {
        setSelectedEntity(null);
        setSelectedEntityProfile(null);
      }
      setRecommendationRequested(true);
      setRecommendationType(type);
      setError("");
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setRecommendationLoading(false);
    }
  }

  function handleSelectTeam(team) {
    setSelectedTeam(team);
    setTeamRecommendations([]);
    setSelectedEntity(null);
    setSelectedEntityProfile(null);
    setShowSelectedEntityModal(false);
    setRecommendationRequested(false);
    setRecommendationType("batting");
    setUnderperformanceView("combined");
    setTeamDetailLoading(true);
    api
      .getTeamDetail(team)
      .then((detail) => {
        setTeamDetail(detail);
        setError("");
      })
      .catch((err) => {
        setError(String(err.message || err));
        setTeamDetail(null);
      })
      .finally(() => setTeamDetailLoading(false));
  }

  async function handleSelectEntity(entity) {
    setSelectedEntity(entity);
    setShowSelectedEntityModal(false);
    const name = entity?.kind === "ipl_squad" ? entity?.data?.player_name : entity?.data?.domestic_player;
    if (!name) {
      setSelectedEntityProfile(null);
      return;
    }

    try {
      setSelectedEntityLoading(true);
      const profile = await api.getPlayerProfile(name);
      setSelectedEntityProfile(profile);
      setError("");
    } catch (err) {
      setError(String(err.message || err));
      setSelectedEntityProfile(null);
    } finally {
      setSelectedEntityLoading(false);
    }
  }

  async function handleLoadPlayerProfile() {
    if (!playerQuery.trim()) return;
    try {
      setPlayerLoading(true);
      if (playerSearchMode === "domestic") {
        const [inference, profile] = await Promise.all([
          api.inferExistingPlayer(playerDomain, playerQuery.trim()),
          api.getPlayerProfile(playerQuery.trim()),
        ]);
        setPlayerInference(inference);
        setPlayerProfile(profile);
      } else {
        const profile = await api.getPlayerProfile(playerQuery.trim());
        setPlayerInference(null);
        setPlayerProfile(profile);
      }
      setShowPlayerProfileModal(false);
      setError("");
    } catch (err) {
      setError(String(err.message || err));
      setPlayerInference(null);
      setPlayerProfile(null);
    } finally {
      setPlayerLoading(false);
    }
  }

  async function handleComparePlayers() {
    if (!comparePlayerA.trim() || !comparePlayerB.trim()) return;
    try {
      setComparisonLoading(true);
      const response = await api.comparePlayers(comparePlayerA.trim(), comparePlayerB.trim());
      setComparisonData(response);
      setError("");
    } catch (err) {
      setError(String(err.message || err));
      setComparisonData(null);
    } finally {
      setComparisonLoading(false);
    }
  }

  function renderSelectedEntityPanel() {
    if (selectedEntityLoading) {
      return (
        <div className="loading-circle-wrap">
          <div className="loading-circle" />
        </div>
      );
    }

    if (!selectedEntity) {
      return (
        <div className="empty-card">
          <strong>No player selected</strong>
          <p>Click a recommended player or a current squad player to inspect the profile here.</p>
        </div>
      );
    }

    if (selectedEntity.kind === "ipl_squad") {
      const player = selectedEntityProfile?.current_squad_profile?.[0] || selectedEntity.data;
      const statBlocks = buildStatBlocksForEntity(selectedEntity, selectedEntityProfile);
      return (
        <div className="profile-detail-stack">
          <div className="profile-highlight">
            <div>
              <div className="micro-label">Current IPL Player</div>
              <h3>{player.player_name}</h3>
              <p>{player.team} • {player.primary_role || "Role unavailable"}</p>
            </div>
            <div className="profile-score">{player.matches_played || 0}</div>
          </div>
          <div className="metric-rows">
            <div><span>Matches Played</span><strong>{player.matches_played || 0}</strong></div>
            <div><span>Runs</span><strong>{player.runs || 0}</strong></div>
            <div><span>Batting Strike Rate</span><strong>{player.batting_strike_rate ? compact(player.batting_strike_rate) : "N/A"}</strong></div>
            <div><span>Wickets</span><strong>{player.wickets || 0}</strong></div>
            <div><span>Bowling Economy</span><strong>{player.bowling_economy ? compact(player.bowling_economy) : "N/A"}</strong></div>
            <div><span>Status</span><strong>{player.status || "Current squad"}</strong></div>
          </div>
          {statBlocks.length ? (
            <button className="text-action" onClick={() => setShowSelectedEntityModal(true)}>
              View all player data
            </button>
          ) : null}
        </div>
      );
    }

    const row = selectedEntity.data;
    const statBlocks = buildStatBlocksForEntity(selectedEntity, selectedEntityProfile);
    return (
      <div className="profile-detail-stack">
        <div className="profile-highlight">
          <div>
            <div className="micro-label">Domestic Player</div>
            <h3>{row.domestic_player}</h3>
            <p>{row.domestic_team} • {row.target_role}</p>
          </div>
          <div className="profile-score">{compact(row.final_recommendation_score)}</div>
        </div>
        <div className="metric-rows">
          <div><span>Quality Score</span><strong>{compact(row.quality_score)}</strong></div>
          <div><span>Similarity Score</span><strong>{compact(row.similarity_score)}</strong></div>
          <div><span>Recent Form</span><strong>{compact(row.recent_form_score)}</strong></div>
          <div><span>ML Suitability</span><strong>{compact(row.ml_suitability_score)}</strong></div>
          <div><span>Closest IPL Benchmark</span><strong>{row.closest_ipl_benchmark || "N/A"}</strong></div>
          <div><span>Archetype</span><strong>{row.archetype_role || "N/A"}</strong></div>
        </div>
        {statBlocks.length ? (
          <button className="text-action" onClick={() => setShowSelectedEntityModal(true)}>
            View all player data
          </button>
        ) : null}
      </div>
    );
  }

  return (
    <div className="ui-shell">
      <header className="top-navigation">
        <div className="brand-block">
          <div className="brand-mark">ITI</div>
          <div>
            <div className="brand-name">IPL Talent Intelligence</div>
            <div className="brand-description">Domestic player recommendation system for coaches, analysts, and selectors</div>
          </div>
        </div>
        <nav className="nav-strip">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`nav-button ${activeView === item.id ? "active" : ""}`}
              onClick={() => setActiveView(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="workspace-shell">
        {error ? <div className="alert-banner">{error}</div> : null}
        {loading ? <div className="loading-panel">Loading data from backend…</div> : null}

        {!loading && activeView === "home" ? (
          <section className="view-stack">
            <div className="hero-panel">
              <div>
                <div className="micro-label">System Overview</div>
                <h1>Build balanced IPL squads from domestic talent, with clear player evidence.</h1>
              </div>
              <div className="hero-side-grid">
                <div className="hero-side-card">
                  <span>Recommendation Engine</span>
                  <strong>Ready</strong>
                </div>
                <div className="hero-side-card">
                  <span>Testing Mode</span>
                  <strong>Random + Time-Based</strong>
                </div>
              </div>
            </div>

            <div className="summary-grid">
              {homepageMetrics.map((item) => (
                <div key={item.label} className="summary-card">
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                  <p>{item.note}</p>
                </div>
              ))}
            </div>

            <div className="guide-grid">
              <div className="plain-card">
                <h3>What “Player Suitability” Means</h3>
                <p>
                  It is the model’s estimate of how well a player fits IPL-style role requirements based on engineered
                  cricket performance features. It is not a guarantee of future success, but a data-supported scouting score.
                </p>
              </div>
              <div className="plain-card">
                <div className="panel-head">
                  <h3>Candidate Pool Integrity</h3>
                  <span className="status-chip">Recommendation base</span>
                </div>
                <div className="health-check-list">
                  {candidatePoolSummary.slice(0, 6).map((row) => (
                    <div key={row.candidate_pool} className="health-check-row neutral">
                      <div>
                        <strong>{prettyTag(row.candidate_pool)}</strong>
                        <span>Validated pool segment</span>
                      </div>
                      <span className="health-badge neutral">{row.players}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        ) : null}

        {!loading && activeView === "teamlab" ? (
          <section className="view-stack">
            <div className="section-intro">
              <div>
                <div className="micro-label">Team Lab</div>
                <h2>Generate franchise recommendations only when you ask for them.</h2>
                <p>Select a team, then click the action button to load gaps, replacements, and domestic player suggestions.</p>
              </div>
            </div>

            <div className="team-lab-grid">
              <div className="plain-card">
                <div className="panel-head">
                  <h3>1. Select Franchise</h3>
                  <span className="status-chip">Controlled flow</span>
                </div>
                <div className="team-selector-list">
                  {teams.map((team) => (
                    <button
                      key={team}
                      className={`team-selector ${selectedTeam === team ? "active" : ""}`}
                      onClick={() => handleSelectTeam(team)}
                    >
                      {team}
                    </button>
                  ))}
                </div>
                <button
                  className="primary-cta top-gap"
                  onClick={() => handleGenerateRecommendations(recommendationType)}
                  disabled={!selectedTeam || recommendationLoading}
                >
                  {recommendationLoading ? (
                    <span className="button-loading">
                      <span className="button-spinner" />
                      Generating...
                    </span>
                  ) : (
                    "Generate Team Recommendations"
                  )}
                </button>
              </div>

              <div className="plain-card">
                <div className="panel-head">
                  <h3>2. Workspace Status</h3>
                  <span className="status-chip">{selectedTeam || "No team selected"}</span>
                </div>
                {teamDetailLoading ? (
                  <div className="loading-circle-wrap">
                    <div className="loading-circle" />
                  </div>
                ) : !teamDetail ? (
                  <div className="empty-card">
                    <strong>No team data loaded yet</strong>
                    <p>Choose a franchise to load the current squad automatically.</p>
                  </div>
                ) : (
                  <div className="workspace-status-stack">
                    <div className="mini-stat-grid">
                      <div className="mini-stat-card">
                        <span>Current Squad</span>
                        <strong>{teamDetail.current_squad?.length || 0}</strong>
                      </div>
                      <div className="mini-stat-card">
                        <span>Role Gaps</span>
                        <strong>{teamGapSummary.total_missing_slots || 0}</strong>
                      </div>
                      <div className="mini-stat-card">
                        <span>Flagged Players</span>
                        <strong>{flaggedPlayersCount}</strong>
                      </div>
                      <div className="mini-stat-card">
                        <span>High Risk</span>
                        <strong>{teamDetail.underperformers?.length || 0}</strong>
                      </div>
                      <div className="mini-stat-card">
                        <span>Monitoring</span>
                        <strong>{teamDetail.monitoring_players?.length || 0}</strong>
                      </div>
                      <div className="mini-stat-card">
                        <span>Replacements</span>
                        <strong>{teamDetail.replacements?.length || 0}</strong>
                      </div>
                    </div>
                    {/* <div className="health-check-list compact-health">
                      {recommendationChecks.map((check) => (
                        <div key={check.check} className={`health-check-row ${check.status === "pass" ? "pass" : "fail"}`}>
                          <div>
                            <strong>{prettyTag(check.check)}</strong>
                            <span>System QA</span>
                          </div>
                          <span className={`health-badge ${check.status === "pass" ? "pass" : "fail"}`}>{check.count}</span>
                        </div>
                      ))}
                    </div> */}
                  </div>
                )}
              </div>
            </div>

            <div className="team-lab-grid">
              <div className="plain-card">
                <div className="panel-head">
                  <h3>Recommended Players</h3>
                  <span className="status-chip">
                    {recommendationRequested ? `${teamRecommendations.length} ${recommendationType} loaded` : "Waiting"}
                  </span>
                </div>
                <div className="type-switcher">
                  {RECOMMENDATION_TYPE_OPTIONS.map((option) => (
                    <button
                      key={option.id}
                      className={`type-switch ${recommendationType === option.id ? "active" : ""}`}
                      onClick={() => {
                        setRecommendationType(option.id);
                        if (recommendationRequested) {
                          handleGenerateRecommendations(option.id);
                        }
                      }}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
                {!recommendationRequested ? (
                  <div className="empty-card">
                    <strong>No recommendations yet</strong>
                    <p>
                      Choose a recommendation type and generate the shortlist. You can switch separately between batters,
                      bowlers, and allrounders.
                    </p>
                  </div>
                ) : !teamRecommendations.length ? (
                  <div className="empty-card">
                    <strong>No {recommendationType} recommendations for {selectedTeam}</strong>
                    <p>
                      This team currently has no shortlisted domestic players in the selected recommendation group.
                      Try another tab or review the role-gap and replacement sections for the team context.
                    </p>
                  </div>
                ) : (
                  <div className="player-list-table recommendation-scroll">
                    {teamRecommendations.map((row) => (
                      <div
                        key={`${row.domestic_player}-${row.target_role}`}
                        onClick={() => handleSelectEntity({ kind: "domestic_recommendation", data: row })}
                        style={{ cursor: "pointer" }}
                      >
                        <ExplainabilityCard
                          row={row}
                          player={row.domestic_player}
                          team={selectedTeam}
                          score={Number(row.final_recommendation_score) || 0}
                          role={row.target_role}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="plain-card">
                <div className="panel-head">
                  <h3>Selected Player Profile</h3>
                  <span className="status-chip">Domestic or IPL player</span>
                </div>
                {renderSelectedEntityPanel()}
              </div>
            </div>

            <div className="team-lab-grid">
              <div className="plain-card">
                <div className="panel-head">
                  <h3>Current Squad</h3>
                  <span className="status-chip">
                    {teamDetail?.current_squad ? `${teamDetail.current_squad.length} players` : "Squad view"}
                  </span>
                </div>
                {!teamDetail?.current_squad?.length ? (
                  <div className="empty-card">
                    <strong>No squad data shown yet</strong>
                    <p>Select a franchise to review the existing squad for that team.</p>
                  </div>
                ) : (
                  <div className="player-list-table compact-table">
                    {teamDetail.current_squad.map((player) => (
                      <button
                        key={`${player.team}-${player.player_name}`}
                        className={`player-list-row ${selectedEntity?.kind === "ipl_squad" && selectedEntity?.data?.player_name === player.player_name ? "active" : ""}`}
                        onClick={() => handleSelectEntity({ kind: "ipl_squad", data: player })}
                      >
                        <div className="row-main">
                          <strong>{player.player_name}</strong>
                          <span>{player.primary_role || "Role unavailable"}</span>
                        </div>
                        <div className="score-cluster">
                          <span>{player.matches_played || 0} matches</span>
                          <small>{player.status || "Current squad"}</small>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="plain-card">
                <div className="panel-head">
                  <h3>Replacement Watch</h3>
                  <span className="status-chip">
                    {teamDetail?.underperformers ? `${filteredUnderperformers.length} flagged` : "Monitoring"}
                  </span>
                </div>
                {!recommendationRequested ? (
                  <div className="empty-card">
                    <strong>No replacement watch yet</strong>
                    <p>Generate the team workspace to reveal current underperformers and their suggested replacements.</p>
                  </div>
                ) : (
                  <div className="replacement-stack">
                    <div>
                      <div className="analysis-toolbar">
                        <div>
                          <div className="subsection-title">Analysis Lens</div>
                          <p className="muted-note compact-note">Switch between current-season form, multi-season decline, or the combined recommendation view.</p>
                        </div>
                        <div className="type-switcher underperformance-switcher">
                          {UNDERPERFORMANCE_VIEWS.map((option) => (
                            <button
                              key={option.id}
                              className={`type-switch ${underperformanceView === option.id ? "active" : ""}`}
                              onClick={() => setUnderperformanceView(option.id)}
                            >
                              {option.label}
                            </button>
                          ))}
                        </div>
                      </div>
                      <div className="subsection-title">Flagged Current Players</div>
                      {filteredUnderperformers.length ? (
                        <div className="replacement-watch-grid">
                          {filteredUnderperformers.map((row, index) => (
                            <article key={`${row.player}-${index}`} className="watch-card flagged-card">
                              <div className="watch-card-head">
                                <div className="row-main">
                                  <strong>{row.player}</strong>
                                  <span>{prettyTag(row.target_role)}</span>
                                </div>
                                <div className={`watch-score watch-score-${row.severity || "low"}`}>
                                  {compact(
                                    underperformanceView === "current"
                                      ? row.current_season_score ?? row.underperformance_score
                                      : underperformanceView === "trend"
                                        ? row.multi_season_trend_score ?? row.underperformance_score
                                        : row.underperformance_score
                                  )}
                                </div>
                              </div>
                              <div className="watch-tag-row">
                                <span className={`watch-tag severity-${row.severity || "low"}`}>{prettyTag(row.severity)}</span>
                                <span className="watch-tag neutral">{prettyTag(row.action_level)}</span>
                                {row.trend_label ? <span className="watch-tag neutral">{prettyTag(row.trend_label)}</span> : null}
                              </div>
                              <div className="replacement-metrics trend-metrics">
                                <div>
                                  <span>Current Season</span>
                                  <strong>{compact(row.current_season_score ?? row.underperformance_score)}</strong>
                                </div>
                                <div>
                                  <span>Trend Score</span>
                                  <strong>{compact(row.multi_season_trend_score ?? row.underperformance_score)}</strong>
                                </div>
                                <div>
                                  <span>Reliability</span>
                                  <strong>{compact(row.reliability_score)}</strong>
                                </div>
                              </div>
                              <p className="watch-reason">{row.reason}</p>
                            </article>
                          ))}
                        </div>
                      ) : (
                        <p className="muted-note">No watchlist or replace-now cases are flagged for this team in the current report.</p>
                      )}
                    </div>

                    <div>
                      <div className="subsection-title">Monitoring Players</div>
                      {monitoringPlayers.length ? (
                        <div className="replacement-watch-grid">
                          {monitoringPlayers.map((row, index) => (
                            <article key={`${row.player}-monitor-${index}`} className="watch-card monitoring-card">
                              <div className="watch-card-head">
                                <div className="row-main">
                                  <strong>{row.player}</strong>
                                  <span>{prettyTag(row.target_role)}</span>
                                </div>
                                <div className="watch-score watch-score-low">
                                  {compact(
                                    underperformanceView === "current"
                                      ? row.current_season_score ?? row.underperformance_score
                                      : underperformanceView === "trend"
                                        ? row.multi_season_trend_score ?? row.underperformance_score
                                        : row.underperformance_score
                                  )}
                                </div>
                              </div>
                              <div className="watch-tag-row">
                                <span className="watch-tag neutral">{prettyTag(row.action_level)}</span>
                                {row.trend_label ? <span className="watch-tag neutral">{prettyTag(row.trend_label)}</span> : null}
                              </div>
                              <div className="replacement-metrics trend-metrics">
                                <div>
                                  <span>Current Season</span>
                                  <strong>{compact(row.current_season_score ?? row.underperformance_score)}</strong>
                                </div>
                                <div>
                                  <span>Trend Score</span>
                                  <strong>{compact(row.multi_season_trend_score ?? row.underperformance_score)}</strong>
                                </div>
                                <div>
                                  <span>Reliability</span>
                                  <strong>{compact(row.reliability_score)}</strong>
                                </div>
                              </div>
                              <p className="watch-reason">{row.reason}</p>
                            </article>
                          ))}
                        </div>
                      ) : (
                        <p className="muted-note">No soft monitoring cases for this team in the current report.</p>
                      )}
                    </div>

                    <div>
                      <div className="subsection-title">Suggested Replacements</div>
                      {filteredReplacements.length ? (
                        <div className="replacement-watch-grid">
                          {filteredReplacements.map((row, index) => (
                            <article key={`${row.underperforming_player}-${row.recommended_player}-${index}`} className="watch-card replacement-card">
                              <div className="watch-card-head">
                                <div className="watch-rank-badge">#{row.replacement_rank}</div>
                                <div className="watch-score watch-score-strong">{compact(row.replacement_score)}</div>
                              </div>
                              <div className="replacement-flow">
                                <div className="replacement-person">
                                  <span className="replacement-label">Replace</span>
                                  <strong>{row.underperforming_player}</strong>
                                </div>
                                <div className="replacement-arrow">→</div>
                                <div className="replacement-person recommended">
                                  <span className="replacement-label">Target</span>
                                  <strong>{row.recommended_player}</strong>
                                </div>
                              </div>
                              <div className="watch-tag-row">
                                <span className="watch-tag neutral">{prettyTag(row.target_role)}</span>
                                <span className="watch-tag neutral">{row.recommended_team}</span>
                                <span className={`watch-tag ${row.action_level === "competition_needed" ? "match-monitoring" : "neutral"}`}>
                                  {row.action_level === "competition_needed" ? "Monitoring Replacement" : "High-Risk Replacement"}
                                </span>
                                <span className={`watch-tag ${row.role_match_type === "exact_role" ? "match-exact" : "match-fallback"}`}>
                                  {row.role_match_type === "exact_role" ? "Exact Role Match" : "Same Domain Fallback"}
                                </span>
                              </div>
                              <div className="replacement-metrics">
                                <div>
                                  <span>Recent Form</span>
                                  <strong>{compact(row.recent_form_score)}</strong>
                                </div>
                                <div>
                                  <span>Role Fit</span>
                                  <strong>{compact(row.role_fit_score)}</strong>
                                </div>
                              </div>
                            </article>
                          ))}
                        </div>
                      ) : (
                        <p className="muted-note">No replacement options found for this team in the current report.</p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        ) : null}

        {!loading && activeView === "players" ? (
          <section className="view-stack">
            <div className="section-intro">
              <div>
                <div className="micro-label">Player Profiles</div>
                <h2>Search one player and inspect the full profile clearly.</h2>
                <p>This view is for player-level analysis, not bulk dashboard noise.</p>
              </div>
            </div>

            <div className="plain-card">
              <div className="profile-form-grid">
                <label>
                  Search Type
                  <select value={playerSearchMode} onChange={(e) => setPlayerSearchMode(e.target.value)}>
                    {PLAYER_SEARCH_MODES.map((mode) => (
                      <option key={mode.id} value={mode.id}>{mode.label}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Role Domain
                  <select
                    value={playerDomain}
                    onChange={(e) => setPlayerDomain(e.target.value)}
                    disabled={playerSearchMode === "ipl"}
                  >
                    {domains.map((domain) => (
                      <option key={domain} value={domain}>{domain}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Player Name
                  <input
                    value={playerQuery}
                    onChange={(e) => setPlayerQuery(e.target.value)}
                    placeholder={
                      playerSearchMode === "domestic"
                        ? "Enter domestic player name"
                        : "Enter IPL player name"
                    }
                  />
                </label>
                <button className="primary-cta align-end" onClick={handleLoadPlayerProfile} disabled={playerLoading}>
                  {playerLoading ? (
                    <span className="button-loading">
                      <span className="button-spinner" />
                      Loading...
                    </span>
                  ) : (
                    "Open Player Profile"
                  )}
                </button>
              </div>
            </div>

            <div className="team-lab-grid">
              <div className="plain-card">
                <div className="panel-head">
                  <h3>Suitability Summary</h3>
                  <span className="status-chip">Live player scoring</span>
                </div>
                {!(playerSearchMode === "domestic" ? playerInference : playerProfile) ? (
                  <div className="empty-card">
                    <strong>No player loaded</strong>
                    <p>Search for a domestic player or an IPL player to inspect the profile.</p>
                  </div>
                ) : (
                  <div className="profile-detail-stack">
                    <div className="profile-highlight">
                      <div>
                        <div className="micro-label">{playerSearchMode === "domestic" ? "Domestic Player" : "IPL Player"}</div>
                        <h3>{playerSearchMode === "domestic" ? playerInference?.player : playerQuery}</h3>
                        <p>
                          {playerSearchMode === "domestic"
                            ? `${playerInference?.domain} domain • ${playerInference?.archetype_role || "No archetype"}`
                            : "Current IPL squad search"}
                        </p>
                      </div>
                      <div className="profile-score">
                        {playerSearchMode === "domestic"
                          ? compact(playerInference?.suitability_score)
                          : (playerProfile?.current_squad_profile?.[0]?.matches_played || 0)}
                      </div>
                    </div>
                    <div className="metric-rows">
                      {playerSearchMode === "domestic" ? (
                        <>
                          <div><span>Predicted Probability</span><strong>{pct(playerInference?.predicted_probability)}</strong></div>
                          <div><span>Cluster Label</span><strong>{playerInference?.cluster_label || "N/A"}</strong></div>
                          <div><span>Archetype Role</span><strong>{playerInference?.archetype_role || "N/A"}</strong></div>
                        </>
                      ) : (
                        <>
                          <div><span>Matches Played</span><strong>{playerProfile?.current_squad_profile?.[0]?.matches_played || 0}</strong></div>
                          <div><span>Primary Role</span><strong>{playerProfile?.current_squad_profile?.[0]?.primary_role || "N/A"}</strong></div>
                          <div><span>Team</span><strong>{playerProfile?.current_squad_profile?.[0]?.team || "N/A"}</strong></div>
                        </>
                      )}
                    </div>
                    {playerProfile ? (
                      <ProfileCharts
                        profile={playerProfile}
                        fallbackRow={playerSearchMode === "domestic" ? {
                          quality_score: playerInference?.suitability_score,
                          recent_form_score: playerInference?.predicted_probability ? Number(playerInference.predicted_probability) * 100 : 0,
                          ml_suitability_score: playerInference?.suitability_score,
                          strike_rate: playerProfile?.current_squad_profile?.[0]?.batting_strike_rate,
                          economy: playerProfile?.current_squad_profile?.[0]?.bowling_economy,
                        } : playerProfile?.current_squad_profile?.[0]}
                        title="Player Visual Breakdown"
                      />
                    ) : null}
                  </div>
                )}
              </div>

              <div className="plain-card">
                <div className="panel-head">
                  <h3>Recommendation History</h3>
                  <span className="status-chip">Profile evidence</span>
                </div>
                {!playerProfile ? (
                  <div className="empty-card">
                    <strong>No player evidence yet</strong>
                    <p>After loading a player, this panel shows where that player appears in recommendation outputs.</p>
                  </div>
                ) : (
                  <div className="player-evidence-stack">
                    <div className="mini-stat-grid slim">
                      <div className="mini-stat-card">
                        <span>Recommendation Rows</span>
                        <strong>{playerProfile.recommendations?.length || 0}</strong>
                      </div>
                      <div className="mini-stat-card">
                        <span>Similarity Matches</span>
                        <strong>{playerProfile.similarity_matches?.length || 0}</strong>
                      </div>
                      <div className="mini-stat-card">
                        <span>Current IPL Profile Rows</span>
                        <strong>{playerProfile.current_squad_profile?.length || 0}</strong>
                      </div>
                    </div>
                    {buildPlayerProfileBlocks(playerProfile).slice(0, 3).map((block) =>
                      renderStatBlock(block.title, block.record)
                    )}
                    <div className="inline-action-row">
                      {buildPlayerProfileBlocks(playerProfile).length > 3 ? (
                        <button className="text-action" onClick={() => setShowPlayerProfileModal(true)}>
                          View full profile data
                        </button>
                      ) : null}
                    </div>
                    <div className="compact-list">
                      {(playerProfile.similarity_matches || []).slice(0, 5).map((row, index) => (
                        <div key={`${row.ipl_player}-${index}`} className="compact-row">
                          <div className="row-main">
                            <strong>{row.ipl_player}</strong>
                            <span>{row.role} • {row.ipl_team}</span>
                          </div>
                          <strong>{compact(row.similarity_score)}</strong>
                        </div>
                      ))}
                      {(playerProfile.recommendations || []).slice(0, 6).map((row, index) => (
                        <div key={`${row.team}-${index}`} className="compact-row">
                          <div className="row-main">
                            <strong>{row.team}</strong>
                            <span>{row.target_role}</span>
                          </div>
                          <strong>{compact(row.final_recommendation_score)}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        ) : null}

        {!loading && activeView === "compare" ? (
          <section className="view-stack">
            <div className="section-intro">
              <div>
                <div className="micro-label">Player Compare</div>
                <h2>Compare two players side by side with charts, profile stats, and role evidence.</h2>
                <p>Use this page when selectors want precise player-vs-player analysis instead of reading separate profiles.</p>
              </div>
            </div>

            <div className="plain-card">
              <div className="comparison-form-grid">
                <label>
                  Player A
                  <input value={comparePlayerA} onChange={(e) => setComparePlayerA(e.target.value)} placeholder="Enter first player" />
                </label>
                <label>
                  Player B
                  <input value={comparePlayerB} onChange={(e) => setComparePlayerB(e.target.value)} placeholder="Enter second player" />
                </label>
                <button className="primary-cta" onClick={handleComparePlayers} disabled={comparisonLoading}>
                  {comparisonLoading ? (
                    <span className="button-loading">
                      <span className="button-spinner" />
                      Comparing...
                    </span>
                  ) : (
                    "Compare Players"
                  )}
                </button>
              </div>
            </div>

            {!comparisonData ? (
              <div className="plain-card">
                <div className="empty-card">
                  <strong>No comparison loaded yet</strong>
                  <p>Search two players and load the side-by-side comparison report.</p>
                </div>
              </div>
            ) : (
              <>
                <div className="comparison-player-grid">
                  {[comparisonData.summary.player_a, comparisonData.summary.player_b].map((player, index) => (
                    <div key={player.player} className={`plain-card comparison-player-card ${index === 0 ? "left" : "right"}`}>
                      <div className="micro-label">{index === 0 ? "Player A" : "Player B"}</div>
                      <h3>{player.player}</h3>
                      <p>{player.team || "Team unavailable"} • {prettyTag(player.role || player.comparison_domain)}</p>
                      <div className="mini-stat-grid slim">
                        <div className="mini-stat-card"><span>Matches</span><strong>{compact(player.matches)}</strong></div>
                        <div className="mini-stat-card"><span>Runs</span><strong>{compact(player.runs)}</strong></div>
                        <div className="mini-stat-card"><span>Strike Rate</span><strong>{compact(player.strike_rate)}</strong></div>
                        <div className="mini-stat-card"><span>Wickets</span><strong>{compact(player.wickets)}</strong></div>
                        <div className="mini-stat-card"><span>Economy</span><strong>{compact(player.economy)}</strong></div>
                        <div className="mini-stat-card"><span>ML Fit</span><strong>{compact(player.ml_suitability_score)}</strong></div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="comparison-visual-grid">
                  <div className="plain-card">
                    <RadarComparison left={comparisonData.summary.player_a} right={comparisonData.summary.player_b} />
                  </div>
                  <ComparisonBars left={comparisonData.summary.player_a} right={comparisonData.summary.player_b} />
                </div>

                <ComparisonStatTable
                  title="Model and Role Comparison"
                  leftLabel={comparisonData.summary.player_a.player}
                  rightLabel={comparisonData.summary.player_b.player}
                  rows={[
                    { label: "Profile Source", leftText: prettyTag(comparisonData.summary.player_a.profile_source), rightText: prettyTag(comparisonData.summary.player_b.profile_source) },
                    { label: "Role", leftText: prettyTag(comparisonData.summary.player_a.role || comparisonData.summary.player_a.comparison_domain), rightText: prettyTag(comparisonData.summary.player_b.role || comparisonData.summary.player_b.comparison_domain) },
                    { label: "Quality Score", left: comparisonData.summary.player_a.quality_score, right: comparisonData.summary.player_b.quality_score },
                    { label: "Recent Form", left: comparisonData.summary.player_a.recent_form_score, right: comparisonData.summary.player_b.recent_form_score },
                    { label: "Similarity Score", left: comparisonData.summary.player_a.similarity_score, right: comparisonData.summary.player_b.similarity_score },
                    { label: "ML Suitability", left: comparisonData.summary.player_a.ml_suitability_score, right: comparisonData.summary.player_b.ml_suitability_score },
                    { label: "Closest IPL Benchmark", leftText: comparisonData.summary.player_a.closest_ipl_benchmark || "N/A", rightText: comparisonData.summary.player_b.closest_ipl_benchmark || "N/A" },
                  ]}
                />

                <ComparisonStatTable
                  title="Performance Comparison"
                  leftLabel={comparisonData.summary.player_a.player}
                  rightLabel={comparisonData.summary.player_b.player}
                  rows={[
                    { label: "Matches", left: comparisonData.summary.player_a.matches, right: comparisonData.summary.player_b.matches },
                    { label: "Runs", left: comparisonData.summary.player_a.runs, right: comparisonData.summary.player_b.runs },
                    { label: "Strike Rate", left: comparisonData.summary.player_a.strike_rate, right: comparisonData.summary.player_b.strike_rate },
                    { label: "Avg Runs", left: comparisonData.summary.player_a.avg_runs, right: comparisonData.summary.player_b.avg_runs },
                    { label: "Wickets", left: comparisonData.summary.player_a.wickets, right: comparisonData.summary.player_b.wickets },
                    { label: "Avg Wkts/Match", left: comparisonData.summary.player_a.avg_wickets_per_match, right: comparisonData.summary.player_b.avg_wickets_per_match },
                    { label: "Economy", left: comparisonData.summary.player_a.economy, right: comparisonData.summary.player_b.economy },
                  ]}
                />

                <VisualBreakdownCompare
                  leftProfile={comparisonData.player_a}
                  rightProfile={comparisonData.player_b}
                  leftSummary={comparisonData.summary.player_a}
                  rightSummary={comparisonData.summary.player_b}
                />

              </>
            )}
          </section>
        ) : null}

        {!loading && activeView === "testing" ? (
          <section className="view-stack">
            <div className="section-intro">
              <div>
                <div className="micro-label">Model Testing</div>
                <h2>Review the actual validation results without clutter.</h2>
                <p>Time-based validation is the more realistic accuracy view. Random split is shown as a baseline.</p>
              </div>
            </div>

            <div className="testing-grid">
              <div className="plain-card">
                <div className="panel-head">
                  <h3>Random Split vs Time-Based</h3>
                  <span className="status-chip">Core comparison</span>
                </div>
                <div className="testing-table">
                  <div className="testing-header">
                    <span>Domain</span>
                    <span>Random Accuracy</span>
                    <span>Time-Based Accuracy</span>
                    <span>Drop</span>
                  </div>
                  {testingComparison.map((row) => (
                    <div key={row.domain} className="testing-row">
                      <span>{row.domain}</span>
                      <span>{pct(row.accuracy_random)}</span>
                      <span>{pct(row.accuracy_time)}</span>
                      <span>{Number(row.accuracy_drop_pct_points).toFixed(2)} pts</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="plain-card">
                <div className="panel-head">
                  <h3>Time-Based Validation Detail</h3>
                  <span className="status-chip">Main thesis result</span>
                </div>
                <div className="metric-rows">
                  {timeBasedMetrics.map((row) => (
                    <div key={row.domain}>
                      <span>{row.domain} ({row.test_seasons})</span>
                      <strong>{pct(row.accuracy)} • F1 {pct(row.f1_score)}</strong>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        ) : null}

      </main>

      {showSelectedEntityModal && selectedEntity ? (
        <div className="modal-backdrop" onClick={() => setShowSelectedEntityModal(false)}>
          <div className="modal-sheet" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div>
                <div className="micro-label">Full Player Data</div>
                <h3>
                  {selectedEntity.kind === "ipl_squad"
                    ? selectedEntity.data.player_name
                    : selectedEntity.data.domestic_player}
                </h3>
              </div>
              <button className="modal-close" onClick={() => setShowSelectedEntityModal(false)}>
                Close
              </button>
            </div>
            <div className="modal-body">
              {buildStatBlocksForEntity(selectedEntity, selectedEntityProfile).map((block) =>
                renderStatBlock(block.title, block.record)
              )}
              <ProfileCharts
                profile={selectedEntityProfile}
                fallbackRow={
                  selectedEntity.kind === "ipl_squad"
                    ? (selectedEntityProfile?.current_squad_profile?.[0] || selectedEntity.data)
                    : selectedEntity.data
                }
                title="Selected Player Charts"
              />
            </div>
          </div>
        </div>
      ) : null}

      {showPlayerProfileModal && playerProfile ? (
        <div className="modal-backdrop" onClick={() => setShowPlayerProfileModal(false)}>
          <div className="modal-sheet" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div>
                <div className="micro-label">Full Profile Data</div>
                <h3>{playerSearchMode === "domestic" ? playerInference?.player || playerQuery : playerQuery}</h3>
              </div>
              <button className="modal-close" onClick={() => setShowPlayerProfileModal(false)}>
                Close
              </button>
            </div>
            <div className="modal-body">
              {buildPlayerProfileBlocks(playerProfile).map((block) =>
                renderStatBlock(block.title, block.record)
              )}
              <ProfileCharts
                profile={playerProfile}
                fallbackRow={playerSearchMode === "domestic" ? {
                  quality_score: playerInference?.suitability_score,
                  recent_form_score: playerInference?.predicted_probability ? Number(playerInference.predicted_probability) * 100 : 0,
                  ml_suitability_score: playerInference?.suitability_score,
                  strike_rate: playerProfile?.current_squad_profile?.[0]?.batting_strike_rate,
                  economy: playerProfile?.current_squad_profile?.[0]?.bowling_economy,
                } : playerProfile?.current_squad_profile?.[0]}
                title="Player Visual Breakdown"
              />
            </div>
          </div>
        </div>
      ) : null}

    </div>
  );
}

export default App;
