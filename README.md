# LLM-Powered Feature Engineering for Recommender Systems

## Project Overview

This thesis project investigates the efficacy of Large Language Models (LLMs) in generating high-quality, thematic features for recommender systems. Specifically, it explores whether LLM-generated keywords can enhance the predictive accuracy of a Factorization Machine (FM) model compared to traditional human-curated keywords. The project involved building a robust, scalable, and automated cloud-based pipeline for data processing and LLM-powered feature generation, followed by rigorous model training and evaluation.


## Key Findings

The experimental results showed **no statistically significant difference** in predictive accuracy (RMSE) between the model trained with human-curated keywords and the model trained with LLM-generated keywords, as the confidence intervals for the two models' RMSE scores overlapped. This finding suggests that, in this specific context, LLM-generated features can perform on par with high-quality, human-curated features for a Factorization Machine model.

## Methodology Highlights

The project's methodology is structured into three interconnected modules, ensuring a rigorous and reproducible comparison between human-curated and LLM-generated features:

1.  **Data Ingestion & Unification:**
    *   Integrated MovieLens 25M (user ratings) and TMDB 5000 (movie metadata, including human-curated keywords and plot overviews) datasets.
    *   Unified data into a `master_dataframe.parquet` for efficient, columnar storage.

2.  **LLM Feature Generation Pipeline:**
    *   Developed a robust, scalable, and automated serverless pipeline on Google Cloud Platform (GCP) using Cloud Functions.
    *   Employed a precise, multi-shot prompt engineered with the R.I.S.E. framework to guide the **Gemini 2.5 Flash** model in generating thematic keywords from movie plot overviews.
    *   Implemented memory-efficient sampling (`pyarrow`) and parallel API calls (`ThreadPoolExecutor`) to handle large datasets and API rate limits.
    *   Processed movies "one prompt, one movie" to prioritize output quality and consistency for research rigor.

3.  **Model Training & Evaluation:**
    *   Utilized **Factorization Machine (FM)** models, implemented in PyTorch, known for their effectiveness with sparse data and ability to capture feature interactions.
    *   Trained two separate and independent FM models: a **Control Group** (human keywords) and an **Experimental Group** (LLM keywords).
    *   Performed systematic **hyperparameter optimization using Optuna** for both models to ensure peak performance.
    *   Ensured **reproducibility** through consistent global random seeding.
    *   Evaluated models using **Root Mean Squared Error (RMSE)** as the primary metric.
    *   Assessed statistical significance using **95% Confidence Intervals (CIs)** calculated via 10,000 bootstraps, interpreting results with a two-tailed comparison.

## Technology Stack

*   **Programming Language:** Python 3.13
*   **Data Manipulation:** pandas, NumPy, scikit-learn
*   **Machine Learning:** PyTorch (for Factorization Machine implementation)
*   **LLM Interaction:** Google Gemini 2.5 Flash (google-generativeai library)
*   **Cloud Platform:** Google Cloud Platform (Cloud Functions, Cloud Storage, Cloud Scheduler)
*   **Development Environment:** Google Colaboratory (Colab), Google AI Studio
*   **Hyperparameter Optimization:** Optuna
*   **Version Control:** Git & GitHub

## Repository Structure

*   `cloud_function/`: Contains the source code for the Google Cloud Functions (`movie_feature_generation.py`, `requirements.txt`).
*   `data/`: Placeholder for raw and processed data files (e.g., `master_dataframe.parquet`, `final_llm_features_dataset.parquet`).
*   `notebooks/`: Jupyter notebooks for data unification (`01-data-unification.ipynb`), merging processed batches (`02-merge-batches.ipynb`), and the main model training/evaluation workflow (`training_fm.ipynb`).
*   `scripts/`: Python scripts, including `run_fm_model.py` for the core experiment.

## How to Run the Experiment (using Google Colab)

To easily reproduce the experiment and run the Factorization Machine model training and evaluation:

1.  **Open in Colab:** Click on the `training_fm.ipynb` notebook located in the `notebooks/` directory and open it directly in Google Colaboratory.
2.  **Select GPU Runtime:** In Colab, go to `Runtime > Change runtime type` and select `T4 GPU` as the hardware accelerator. This will significantly speed up model training.
3.  **Run All Cells:** Execute all cells in the notebook (`Runtime > Run all`). The notebook will automatically clone the repository, install dependencies, and run the full experiment, including hyperparameter tuning and statistical evaluation.

