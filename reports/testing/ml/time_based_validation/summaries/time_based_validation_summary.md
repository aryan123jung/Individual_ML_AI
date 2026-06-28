# Time-Based Validation Summary

This validation uses older IPL seasons for training and the latest seasons for testing.
It is a stricter and more realistic test than a random split because it evaluates whether the model generalizes forward in time.

## Results
- Batting: tested on seasons `2024,2025,2026` with accuracy `0.7228`, F1-score `0.7713`, and ROC-AUC `0.8128`.
- Bowling: tested on seasons `2024,2025,2026` with accuracy `0.8571`, F1-score `0.8070`, and ROC-AUC `0.9508`.
- Allrounder: tested on seasons `2024,2025,2026` with accuracy `0.9427`, F1-score `0.9565`, and ROC-AUC `0.9905`.

## Confusion Matrices
- Batting: true negatives `47`, false positives `26`, false negatives `25`, true positives `86`.
- Bowling: true negatives `86`, false positives `5`, false negatives `17`, true positives `46`.
- Allrounder: true negatives `87`, false positives `15`, false negatives `1`, true positives `176`.

If the time-based scores are lower than the random-split scores, that is expected and academically healthier. It means the time-based evaluation is testing real generalization instead of same-distribution memorization.