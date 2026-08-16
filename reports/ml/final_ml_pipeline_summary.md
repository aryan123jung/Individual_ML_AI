# Final ML Pipeline Summary

## Production Models
The final production suitability model for all three domains is Random Forest.
This choice is based on consistent superiority over Logistic Regression in the internal model comparison stage.

## Why Time-Based Validation Is The Main Result
Random train/test split is useful as a baseline, but time-based validation is more academically correct for this thesis.
The system is intended to learn from historical IPL patterns and generalize forward to later seasons and domestic candidates.
Therefore, the main thesis result should be the time-based metrics, not the random-split metrics.

## Final Domain Results
- Batting: production model `random_forest`, time-based accuracy `0.7228`, F1-score `0.7713`, ROC-AUC `0.8128`, tested on seasons `2024,2025,2026`.
- Bowling: production model `random_forest`, time-based accuracy `0.8571`, F1-score `0.8070`, ROC-AUC `0.9508`, tested on seasons `2024,2025,2026`.
- Allrounder: production model `random_forest`, time-based accuracy `0.9427`, F1-score `0.9565`, ROC-AUC `0.9905`, tested on seasons `2024,2025,2026`.

## Domain Interpretation
- Batting is the hardest domain because batting performance is more volatile and role-sensitive, which explains its lower forward accuracy than bowling and all-rounder prediction.
- Bowling is more stable because economy, wickets, and phase control are comparatively consistent and structurally measurable.
- All-rounder performance is the strongest domain because the feature space captures both batting and bowling contribution together, creating clearer separation between successful and unsuccessful profiles.

## Allrounder High-Score Justification
The all-rounder model shows very high performance, so it requires explicit justification in the thesis.
An ablation study was used for this purpose.
- Full all-rounder feature set: accuracy `0.9427`, F1 `0.9565`.
- Without `allrounder_index`: accuracy `0.8889`, F1 `0.9136`.
- Raw features only: accuracy `0.8530`, F1 `0.8889`.
This shows that the performance is not purely artificial. The model remains strong even after removing composite strength features, although accuracy declines when the composite all-rounder index is removed.
Therefore, the correct thesis position is that the all-rounder score is high because the role is genuinely more separable in the engineered feature space, while composite features further strengthen that separability.

## Final Thesis Position
The system is genuinely AI/ML-based because it uses supervised learning to estimate IPL suitability and unsupervised clustering to assign player archetypes.
However, the recommendation engine is not only a classifier. It is a hybrid decision-support framework that combines:
- engineered cricket features,
- supervised suitability prediction,
- clustering-based archetype support,
- similarity benchmarking against IPL profiles, and
- rule-based squad gap and replacement logic.

This is the correct final description for the thesis: a hybrid AI/ML-driven player recommendation system for IPL squad formation.
