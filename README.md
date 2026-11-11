
## Implementation Setup

This section elaborates on our implementation setup and key hyperparameters.

For uncertainty quantification of categorical parameters, we use split conformal prediction with inverse probability as the nonconformity score.

- **Conformal prediction:** The miscoverage level is set to $\alpha = 0.20$, selected by experimenting with $\{0.05, 0.10, 0.15, 0.20, 0.25\}$ on the calibration split.
- **Distance threshold:** We sample $N = 10$ different interpretations from the LLM at temperature $T = 0.9$.
- **Clustering:** For both distance threshold and anchor location, we apply DBSCAN with $\epsilon = 2$ km.
- **Clarification:** The maximum number of dialogue turns is set to 5.
- **Dataset split:** 80% test and 20% calibration.
- **Hardware:** All experiments are conducted on an NVIDIA A100-SXM4 GPU server with 40 GB of memory.

## Data Characteristics in Each Study Area

The EVClarify-QA dataset covers six study areas. The table below summarizes the number of road segments, intersections, percentage of roads intersecting high-risk flood zones, and mapped emergency facilities in each city.

| City | Road Segments | Intersections | Flood Zone (%) | Emergency Facilities |
|---|---:|---:|---:|---:|
| Chicago | 76,742 | 29,220 | 0.2 | 736 |
| Miami | 23,498 | 9,074 | 24.8 | 355 |
| New Orleans | 39,394 | 15,164 | 30.2 | 243 |
| New York City | 139,184 | 55,263 | 0.6 | 3,804 |
| Houston | 364,432 | 149,788 | 12.8 | 930 |
| San Antonio | 154,784 | 61,020 | 3.0 | 418 |
