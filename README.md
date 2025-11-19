## Structure of this Repository

This repository contains the following main components.
- `data/`: This directory is intended to store datasets. It is structured into `raw/` for unprocessed data and `processed/` for cleaned and transformed data ready for analysis.
- `notebooks/`: This directory contains Jupyter notebooks used for data exploration, preprocessing, and model development.
- `src/`: This directory holds the source code for some helper modules developed during the project. It mainly contains three things:
  - `evaluate.py`: Function to calculate the AMS metric.
  - `naive_bayes/`: Implementations of the Naive Bayes model factories, which allow easy creation of different variants of the Naive Bayes model.
  - `naive_bayes/experiments/`: Lightweight experiment framework to run experiments with different configurations of the Naive Bayes models and log the results.
- `models/`: This directory is used to cache trained models and related artifacts.
- `plots/`: This directory is used to store generated plots and visualizations.
- `results/`: This directory is used to store results from experiments, such as performance metrics and logs (More on this later).
- **Project Files**: These are files such as `README.md`, `pyproject.toml`, and `uv.lock`, which define dependencies and provide project documentation.
- **Data Version Control**: These are files related to DVC (Data Version Control) such as `dvc.yaml` and `.dvc/`. DVC is a tool that helps manage and version control large datasets and machine learning models, ensuring reproducibility and collaboration in data science projects.

## Installation

To install the necessary dependencies for this project, please follow the steps below:

1. **Clone the Repository**  
   Open your terminal and run the following command to clone the repository:
   ```bash
   git clone https://github.com/AimbotParce/MDS-AML-HiggsBosonATLAS.git
   ```

2. **Navigate to the Project Directory**
3. **Install Dependencies**
    **Recommended way:** using `uv`, which can be installed following instructions at https://docs.astral.sh/uv/getting-started/installation/
    ```bash
    uv sync
    ```
    **Alternative way:** using `pip` to install dependencies from `pyproject.toml` into a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install .
    ```
4. **Verify Installation**
    You can verify that the installation was successful by running:
    ```bash
    dvc --version
    ```
    If the command returns the version of DVC, the installation was successful. If you don't have DVC properly
    installed, you may run into issues in the next steps.

## Data Setup
To set up the data for this project, it's as easy as running the following command in the project directory:
```bash
dvc repro
```
This command will download the raw dataset into `data/raw/` and process it into `data/processed/` as required for the
project.

> [!NOTE]
> If you don't want to run the pipeline using dvc, you can do so manually using the following command (check that you have the virtual environment activated):
> ```bash
> python -c "import urllib.request; urllib.request.urlretrieve('https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz','data/raw/higgs-challenge.csv.gz')"
> ```
> And then running the entire preprocessing notebook located at `notebooks/00-preprocess.ipynb`.

## Jupyter Notebooks

At this point, you should be able to run Jupyter notebooks in the `notebooks/` directory. You can execute them in order, as they have been named to reflect the intended sequence of execution. If you have already stepped through the data setup section, you can start directly from the first notebook (`00-preprocess.ipynb`):

> [!IMPORTANT]
> Before proceeding, take a minute to explore how the Naive Bayes experiments framework operates. We have provided a set of all the experiments we ran, and their results in the `results/` directory, and the framework is designed to check if an experiment has already been run before executing it again. This means you can safely re-run the notebooks without worrying about overwriting previous results, nor waiting for hours to move forward.
> 
> However, this also means that if you wish to re-run any experiments, you will need to manually delete them from their corresponding result file in the `results/` directory (`results/experiments-results.jsonl` for hyperparameter tuning experiments, and `results/final-experiments-results.jsonl` for the final model trainings).
> 
> This design choice was made to facilitate easy experimentation without the risk of losing prior results.

1. `00-preprocess.ipynb`: Data preprocessing and cleaning.
2. `01-logistic-regression.ipynb`: Implementation and evaluation of Logistic Regression model.
3. `02-bayesian-logistic-regression.ipynb`: Implementation and evaluation of Bayesian Logistic Regression model.
4. `03-non-gaussian-naive-bayes.ipynb`: Exploration of the different implementations of the Naive Bayes model, without further hyperparameter tuning.
5. `04-naive-bayes-model-selection.ipynb`: Definition and execution of all the experiments for hyperparameter tuning of all the Naive Bayes models.
6. `05-model-selection-results-analysis.ipynb`: Ranking and mean AMS score analysis of all the experiments run in the previous notebook, plus selection of the best hyperparameters for each model.
7. `06-naive-bayes-final-trainings.ipynb`: Final training of the best models with the selected hyperparameters over the entire learning set, with evaluation on the test set.

An extra notebook, `P01-marginal-distributions.ipynb` can be found in the `notebooks/` directory, which was used to create one of the plots from the project report.

## Report

The project report can be found in the root directory as a PDF file named `HiggsBosonATLAS-Report.pdf`. In there, all the methodologies, experiments, results, and conclusions drawn from this project are detailed comprehensively.

