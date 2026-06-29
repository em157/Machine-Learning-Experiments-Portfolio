# MAB-Enhanced Classifier for Image Anomaly Detection

This project explores anomaly detection on the MVTec dataset using image embeddings, k-nearest neighbors scoring, PCA-based dimensionality reduction, and a multi-armed bandit strategy for adaptive threshold selection.

## Included artifacts
- `nb_script.py`: main notebook-exported workflow for embedding extraction and kNN evaluation
- `MAB_kNN_Anomaly_Detection.ipynb`: presentation-oriented notebook with executive summary
- `notebook_1.ipynb`: working notebook with the full embedding and anomaly-scoring pipeline
- `Evan_Morrow_MAB_kNN_Anomaly_Detection.pdf`: project write-up

## Approach
- Extract image embeddings from MVTec categories using torchvision or timm backbones
- Build a reference set from normal training images
- Score test samples with kNN similarity over normalized embeddings
- Optionally reduce dimensionality with PCA fit on normal examples only
- Use a multi-armed bandit policy to adapt anomaly thresholds based on feedback

## Dataset
This repository copy does not include the MVTec dataset. Point the notebooks or script at a local MVTec download, for example:

`E:\Computational_Specialty_Projects\MAB_Enhanced_Classifier\mvtec_anomely_dataset`

## Dependencies
Install the Python packages listed in `requirements.txt` before running the workflow.

## Notes
The current script contains local Windows paths and is oriented toward notebook-driven experimentation rather than a packaged CLI.
