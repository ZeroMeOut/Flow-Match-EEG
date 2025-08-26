# Exploration of Flow Matching for Cleaning EEG Artifacts
## Introduction
Electroencephalography (EEG) signal processing is an ongoing challenge in neuroscience research. Deep learning approaches have shown promise for many EEG processing tasks, but the application of continuous normalizing flows, specifically flow-match models, to EEG signals remains unexplored. This study presents the first investigation of flow-match models in the EEG domain, using artifact removal as a test case to establish baseline performance.

Two model variants were implemented: Unit2 (flow-match framework) and Unit4 (direct prediction), both trained on the EEGdenoiseNet dataset, which contains 514 clean EEG segments, 3400 ocular artifacts, and 5598 muscular artifacts across 64-channel recordings. Both models use transformer encoders and residual blocks for signal processing, making their architectures very similar to each other.

The evaluation using root mean squared error, signal correlation, and frequency band analysis shows that the flow-match model achieved a higher RMSE (5.30 ± 1.80) compared to direct prediction methods (4.34 ± 1.19) and existing baselines (4.53 ± 1.04). Signal correlations were similarly lower for flow-match (0.55 ± 0.10) versus direct methods (0.72 ± 0.07). The frequency domain analysis reveals that the flow-match model tends to overpreserve lower-frequency components, potentially retaining artifacts.

These findings show that naive applications of flow-match frameworks to EEG processing do not automatically give better results over the previous methods. The work lays the foundation for more advanced applications of continuous normalizing flows in EEG signal processing.

## Getting started
### Pre-requirements
- Python 3.10 or higher
- Git
### Clone repository
```bash
git clone https://github.com/ZeroMeOut/Flow-Match-EEG.git
cd Flow-Match-EGG
```
### Create a virtual environment in the repository
```python
pyhton -m venv .venv
```
### Install dependencies
```python
pip install -r requirements.txt
```
### Generating data
Enter into EEGDenoisenetTraining/eegdenoisenet and run the ```generate_data.py``` file
```python
python run generate_data.py
```
### Training and Testing
All IPython notebooks that have training_ are for training, and ```testing.ipynb``` for testing the trained models. You can also open TensorBoard via the training notebooks for the training and validation losses.
