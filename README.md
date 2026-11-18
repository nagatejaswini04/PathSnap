# PathSnap: Distributed Map-Matching & Route Reconstruction

**A Distributed Machine Learning System for GPS Trajectory Reconstruction**

---

## 1. Project Overview

GPS tracking systems frequently produce inaccurate location measurements due to urban buildings, signal interference, multipath effects, GPS dropouts and low sampling frequencies.

As a result, a vehicle travelling along a road may appear to move onto a parallel street, jump across buildings or deviate from its actual route.

**PathSnap** aims to solve this problem by developing a distributed Machine Learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations to a geographical road network.

The system combines GPS trajectory processing, road-network analysis, Machine Learning classification, probabilistic sequence modelling and distributed deployment to identify the most likely sequence of road segments travelled by a vehicle.

The project follows an end-to-end development approach, from raw dataset exploration and feature engineering to model training, route reconstruction, evaluation and deployment.

## 2. Problem Statement

Given a sequence of noisy GPS observations and an associated road network, develop a system capable of identifying the most likely road segments traversed by a vehicle and reconstructing a geographically and topologically consistent route.

The system must:

- Process raw GPS trajectory data.
- Identify candidate road segments for GPS observations.
- Train and compare multiple Machine Learning models.
- Estimate candidate road-matching probabilities.
- Reconstruct realistic road sequences using Hidden Markov Models (HMM) and Viterbi decoding.
- Evaluate reconstructed routes against supplied reference routes.
- Provide geographical route visualisation.
- Operate through a distributed two-laptop deployment architecture.

## 3. Project Objectives

1. Understand and explore the provided GPS trajectory and road-network dataset.
2. Perform data cleaning, preprocessing and feature engineering.
3. Generate candidate road segments for GPS observations.
4. Train and evaluate at least five distinct Machine Learning models.
5. Compare models using appropriate classification and route-matching metrics.
6. Select a suitable model based on experimental evidence.
7. Perform Bayesian hyperparameter optimisation on the selected model.
8. Implement HMM-based sequential route reconstruction using Viterbi decoding.
9. Compare classifier-only predictions against classifier-plus-HMM predictions.
10. Develop a FastAPI backend and an evaluator-facing user interface.
11. Deploy the complete system across two separate laptops.
12. Generate geographical visualisations and prepare a reproducible technical report.

## 4. Dataset Description

The dataset contains GPS trajectory information and associated road-network information. It is organised by trip.

Each trip contains four types of raw files.

| File Extension | Description |
|---|---|
| `.track` | GPS observations containing longitude, latitude and timestamp |
| `.nodes` | Geographical road-network nodes containing longitude and latitude |
| `.arcs` | Road-network connections represented by from-node and to-node IDs |
| `.route` | Chronological sequence of reference road-segment arc IDs |

### Dataset Structure

A typical trip is organised as follows:

```text
trip_id/
    trip_id.track
    trip_id.nodes
    trip_id.arcs
    trip_id.route
```

### Dataset Partitions

The complete hackathon dataset is divided into three partitions:

| Partition | Number of Trips | Purpose |
|---|---:|---|
| Training | 80 | Model training, development, cross-validation and optimisation |
| Student Testing | 10 | Development-stage evaluation and model comparison |
| Evaluator Testing | 10 | Final evaluation |

### Data Integrity

- Trips must remain within their assigned partitions.
- Evaluator-testing data must not be used for model training or model selection.
- Evaluator ground-truth labels must remain separate from the prediction pipeline.
- Predictions must not be hardcoded.
- Learned preprocessing transformations must be fitted using permitted training data only.

**Important:** Supplied reference routes are reference information and should not automatically be treated as independently verified, error-free physical ground truth.

## 5. Proposed Methodology

The project follows a structured pipeline.

### Phase 1: Dataset Exploration and Understanding

- Inspect the raw dataset structure.
- Identify file formats, column meanings and data types.
- Analyse GPS observations and timestamp intervals.
- Investigate missing or invalid observations.
- Examine geographical distributions.
- Analyse road-network nodes, arcs and connectivity.
- Identify isolated nodes and disconnected components.
- Visualise selected trajectories and road networks.

### Phase 2: Data Preprocessing

- Load and organise trajectory data.
- Handle invalid or missing observations where appropriate.
- Validate timestamp ordering and irregular time gaps.
- Inspect coordinate plausibility.
- Apply appropriate geographical distance calculations.
- Prepare road-network representations.
- Maintain trip-level partition separation.

### Phase 3: Candidate Generation and Feature Engineering

Generate possible road-segment candidates for each GPS observation.

Investigate and extract suitable features based on the GPS observations, candidate road segments and road-network structure.

Potential feature categories include:

- GPS-to-road geographical distance.
- Vehicle heading and road direction difference.
- Temporal differences between observations.
- Movement and displacement information.
- Road-network connectivity.
- Spatial relationships between consecutive observations.

The final feature set will be determined through dataset exploration and experimental validation.

### Phase 4: Machine Learning Model Experimentation

Train and compare at least five distinct Machine Learning models.

Possible algorithm families include:

- Linear and probabilistic models.
- Instance-based learning.
- Tree-based models.
- Ensemble methods.
- Gradient boosting.
- Kernel-based methods.
- Neural networks.

Model selection will be supported by recorded experimental results rather than assumptions.

### Phase 5: Model Comparison and Selection

Compare trained models using:

- Cross-validation Macro F1-score.
- Observation-level Top-1 Accuracy.
- Precision.
- Recall.
- Macro F1-score.
- Log Loss, where applicable.
- Route continuity.
- Training time.
- Inference time.

Select the final model using measured performance, generalisation, computational efficiency and compatibility with the sequential decoder.

### Phase 6: Bayesian Hyperparameter Optimisation

Perform Bayesian hyperparameter optimisation on the selected model.

The process includes:

1. Define the hyperparameter search space.
2. Specify the optimisation objective.
3. Apply group-aware cross-validation on the 80 training trips.
4. Record optimisation trials and their results.
5. Identify the best parameter configuration.
6. Refit the selected model.
7. Evaluate the tuned model on the student-testing partition.
8. Compare tuned and untuned performance.

### Phase 7: HMM and Viterbi Route Reconstruction

Independent Machine Learning predictions may produce unrealistic road sequences because consecutive GPS observations can be assigned to disconnected road segments.

To address this, implement a Hidden Markov Model (HMM) using sequential information and road-network connectivity.

The route reconstruction process includes:

- Define hidden states representing candidate road segments.
- Model emission probabilities.
- Model transition probabilities.
- Incorporate road-network connectivity.
- Implement Viterbi decoding.
- Handle missing candidates and invalid transitions.
- Handle GPS gaps.
- Generate the most likely sequence of road segments.

Evaluate and compare:

1. Classifier-only predictions.
2. Classifier-plus-HMM predictions.

### Phase 8: Evaluation and Visualisation

Evaluate the complete route-reconstruction pipeline.

Generate:

- Model comparison tables.
- Cross-validation performance graphs.
- Student-testing performance graphs.
- Confusion matrices where appropriate.
- Log Loss comparisons.
- Route continuity analysis.
- Geographical trajectory visualisations.
- Folium maps showing GPS observations and reconstructed routes.

All reported results and visualisations must be generated from actual experiments.

### Phase 9: Distributed Deployment

Develop a two-laptop distributed architecture.

**Laptop 1 – Backend and Model Server**

Responsibilities:

- Host the trained Machine Learning model.
- Load preprocessing components and model configurations.
- Execute candidate prediction and HMM/Viterbi decoding.
- Maintain road-network information.
- Process API requests.
- Perform server-side evaluation.
- Return prediction results.

**Laptop 2 – Frontend and Evaluator Interface**

Responsibilities:

- Display the ten reserved evaluation-trip options.
- Allow the evaluator to select a trip.
- Send the selected trip ID to the backend.
- Receive the API response.
- Display the predicted route and evaluation results.

### Distributed Communication Flow

```text
             LAPTOP 2
        Frontend / Evaluator UI
                  |
                  |
             HTTP Request
                  |
                  v
             LAPTOP 1
           FastAPI Backend
                  |
                  v
          Data Preprocessing
                  |
                  v
           Candidate Generation
                  |
                  v
          Trained ML Classifier
                  |
                  v
             HMM + Viterbi
                  |
                  v
          Predicted Road Sequence
                  |
                  v
          Server-side Evaluation
                  |
                  v
             HTTP Response
                  |
                  v
             LAPTOP 2
          Results and Route Map
```

**Evaluation Integrity:** Ground-truth labels must remain server-side and must never be exposed to the prediction model or frontend interface. The complete prediction pipeline must run through the API.

## 6. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Optimisation | Bayesian Hyperparameter Optimisation |
| Sequential Modelling | Hidden Markov Model |
| Route Decoding | Viterbi Algorithm |
| Backend | FastAPI |
| API Communication | HTTP |
| Geographical Visualisation | Folium |
| Development Environment | Visual Studio Code |
| Version Control | Git and GitHub |

Additional libraries and tools will be finalised based on implementation requirements.

## 7. Repository Structure

```text
PathSnap/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── dataset_exploration.ipynb
│   ├── trajectory_analysis.ipynb
│   ├── road_network_analysis.ipynb
│   └── feature_engineering.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── candidate_generation/
│   ├── feature_engineering/
│   ├── training/
│   ├── evaluation/
│   └── route_reconstruction/
│
├── models/
│   ├── trained/
│   └── configurations/
│
├── experiments/
│   ├── model_comparison/
│   ├── hyperparameter_optimization/
│   └── results/
│
├── backend/
│   ├── main.py
│   ├── routes/
│   └── services/
│
├── frontend/
│   ├── app.py
│   └── components/
│
├── visualizations/
│   ├── trajectories/
│   ├── road_network/
│   └── model_comparison/
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── technical_report/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

The directory structure may evolve as implementation progresses.

## 8. Evaluation Metrics

The following metrics will be considered wherever applicable.

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Correct positive predictions among positive predictions |
| Recall | Correctly identified positive examples |
| Macro F1-score | Balanced performance across classes |
| Log Loss | Quality of predicted probabilities |
| Top-1 Accuracy | Correctness of the highest-scoring road candidate |
| Candidate Miss Rate | Cases where the reference road is absent from the candidate set |
| Route Continuity | Consistency of consecutive predicted road segments |
| Sequence-level Performance | Quality of the reconstructed road sequence |
| Training Time | Model training cost |
| Inference Time | Prediction computational cost |

The primary model-selection metric and its justification will be documented based on experimental findings.

## 9. Experimental Reproducibility

The project will maintain records of:

- Model configurations.
- Random seeds.
- Training and validation results.
- Cross-validation results.
- Student-testing results.
- Hyperparameter optimisation trials.
- Training and inference times.
- Model artifacts.
- Evaluation outputs.

Every reported graph, table and metric must be reproducible from the submitted implementation.

## 10. Team Collaboration

The project is developed by a four-member team.

The initial work distribution is:

| Member | Responsibility |
|---|---|
| Member 1 | Dataset exploration, preprocessing and feature engineering |
| Member 2 | Machine Learning models, comparison and Bayesian optimisation |
| Member 3 | Candidate road matching, HMM and Viterbi route reconstruction |
| Member 4 | FastAPI backend, frontend and distributed integration |

All team members are expected to understand the overall architecture and contribute to integration, testing and technical documentation.

### Version Control Workflow

Git and GitHub are used for collaborative development.

The team follows a branch-based workflow:

```text
main
 |
 |--- feature/data-preprocessing
 |
 |--- feature/ml-models
 |
 |--- feature/hmm-decoder
 |
 |--- feature/backend-ui
```

Changes are committed to individual branches, reviewed through Pull Requests and merged into the main branch after verification.

## 11. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the selected dependencies.
- Git.
- Visual Studio Code.
- GitHub account.

### Clone the Repository

```bash
git clone https://github.com/nagatejaswini04/PathSnap.git

cd PathSnap
```

### Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv

.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

After the project dependencies have been added to `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

### Execution

The exact commands for dataset exploration, model training, route reconstruction, API execution and frontend execution will be added after the corresponding components are implemented and tested.

## 12. Expected Final Deliverables

The hackathon requires exactly two final submission deliverables.

### Deliverable 1: Public GitHub Repository

The repository will contain:

- Dataset processing and preprocessing source code.
- Feature engineering implementation.
- Training scripts and notebooks for all experimented models.
- Model comparison and evaluation implementation.
- Bayesian optimisation implementation and search records.
- HMM and Viterbi implementation.
- FastAPI backend.
- Evaluator-facing UI.
- Model artifacts and configurations.
- Folium visualisation code.
- Experiment logs and supporting results.
- Installation and execution instructions.

### Deliverable 2: Consolidated Technical Report (PDF)

The technical report will document:

- Dataset understanding and exploratory data analysis.
- Data preprocessing and feature engineering.
- Machine Learning model experimentation and comparison.
- Individual model configurations, strengths and limitations.
- Experimental results and visualisations.
- Bayesian hyperparameter optimisation.
- HMM and Viterbi route reconstruction.
- Classifier-only versus classifier-plus-HMM comparison.
- API and UI implementation.
- Distributed deployment.
- Evaluation, limitations and conclusions.

All results included in the report must be verifiable using the submitted GitHub repository.

## 13. Academic Integrity and Evaluation Requirements

The team will follow the hackathon's assessment and academic integrity policies.

- All models must be independently implemented, trained and evaluated by the team.
- Sharing trained weights, model configurations, implementation code or experimental results between teams is prohibited.
- No experimental results, metrics, graphs or implementation details will be fabricated.
- All reported technical claims must be supported by actual implementation and experiments.
- Every team member must be able to explain the project's methodology and implementation.
- Evaluator-testing data and labels must remain protected during development.
- Hardcoded predictions and API bypassing are not permitted.

## 14. Project Status

**Current Stage:** Initial Repository Setup

| Component | Status |
|---|---|
| GitHub repository | Created |
| Team collaboration | Configured |
| Project directory structure | Created |
| Dataset exploration | Pending |
| Data preprocessing | Pending |
| Feature engineering | Pending |
| Machine Learning experimentation | Pending |
| Bayesian optimisation | Pending |
| HMM and Viterbi | Pending |
| Backend development | Pending |
| Frontend development | Pending |
| Distributed deployment | Pending |
| Technical report | Pending |

Project status will be updated as implementation progresses.

---

**PathSnap – Reconstructing Realistic Vehicle Routes from Noisy GPS Trajectories.**