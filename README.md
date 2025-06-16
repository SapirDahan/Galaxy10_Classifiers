# Galaxy Classifier 🔭✨

Welcome to **Galaxy Classifier**, an interactive Jupyter-based project for classifying galaxies into 10 morphological categories using classical and deep-learning algorithms on both ResNet101 feature embeddings and raw images.

---

## 🌟 Project Overview

- **Goal**  
  Explore how well k-Nearest Neighbors, Support Vector Machines, Random Forests, and Convolutional Neural Networks can distinguish between galaxy types (Smooth, Spiral, Edge-on, Merging, Disturbed and their subclasses).

- **Key Questions**
    1. What classification accuracy can be achieved on the 10-class Galaxy10 dataset?
    2. How do different algorithms compare in terms of accuracy, precision, recall, F1-score and confusion patterns?
    3. Do misclassifications respect the intuitive hierarchy (e.g., confusing Spiral-Tight with Spiral-Loose more often than Spiral with Smooth)?
    4. Can an end-to-end CNN trained on raw images outperform models trained on pre-extracted features? fileciteturn1file0

- **Pipeline**
    1. **Feature Extraction**: Already done via ResNet101 → stored in `galaxy10_resnet101_embeddings_augmented_balanced_regularized.npz`
    2. **Model Training & Validation** (in notebooks):
        - **k-NN**: Grid-search over `k`, compute metrics & visualizations
        - **SVM**: RBF-kernel SVM hyperparameter tuning (`C`, `γ`), metrics & visualizations
        - **Random Forest**: Tune number of trees & max depth, metrics & visualizations
        - **CNN**: Build and train a convolutional neural network directly on the raw Galaxy10 images; evaluate performance and inspect feature maps and confusion patterns fileciteturn1file0
        - 3D PCA visualizations of test embeddings (for the classical models)
        - Hierarchical clustering dendrograms to inspect class proximity

---

## 🔍 Dataset

This project uses the **Galaxy10** dataset (69×69 px RGB images in 10 classes). We rely on:

- **Embedding file**:
  ```
  galaxy10_resnet101_embeddings_augmented_balanced_regularized.npz
  ```
- **Raw image folder** (for CNN notebook):
  ```
  Data_Processing/images/galaxy10/
  ```

The NPZ file contains: `train_features`, `val_features`, `test_features`, `train_labels`, `val_labels`, `test_labels`, `class_names`.

---

## 📁 Repository Structure

```
Galaxy_Classifier/
├── .venv/…                             # Python virtual environment
├── Data_Processing/                    # scripts for raw image loading & preprocessing
├── galaxy10_resnet101_embeddings_…npz  # pre-computed feature embeddings
├── k-NN.ipynb                          # k-Nearest Neighbors pipeline & analysis
├── SVM.ipynb                           # Support Vector Machine pipeline & analysis
├── Random_Forest.ipynb                 # Random Forest pipeline & analysis
├── CNN.ipynb                           # Convolutional Neural Network pipeline & analysis
├── requirements.txt                    # project dependencies
└── Final_Project_Proposal___Sapir_…pdf # project proposal & research questions
```

---

## 🛠️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/YourUsername/Galaxy_Classifier.git
   cd Galaxy_Classifier
   ```

2. **Create & activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # Linux / macOS
   .venv\Scripts\activate         # Windows PowerShell
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # For CNN notebook, also install:
   pip install torch torchvision
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```
   Open any of the `.ipynb` notebooks in your browser.

---

## 💻 Usage

1. **k-NN.ipynb**
    - Grid-search over `k` values
    - Compute accuracy, precision, recall, F1, confusion matrix
    - 3D PCA scatter of test embeddings
    - Hierarchical clustering & dendrogram

2. **SVM.ipynb**
    - RBF-kernel SVM hyperparameter tuning (`C`, `γ`)
    - Same suite of metrics & visualizations

3. **Random_Forest.ipynb**
    - Tune number of trees & max depth
    - Evaluate metrics, confusion patterns, PCA plot

4. **CNN.ipynb**
    - Define a CNN architecture (e.g., small stack of conv + pooling layers)
    - Train end-to-end on raw Galaxy10 images
    - Evaluate metrics: accuracy, precision, recall, F1, confusion matrix
    - Visualize feature maps or saliency (e.g., Grad-CAM) to interpret model

> **Tip:** Feel free to adjust hyperparameter grids, swap in different backbone architectures, or incorporate data augmentation strategies.

---

## 📊 Results & Insights

> Check the “Results” section in each notebook for tables, plots, and discussion on:
> - Overall accuracy vs. per-class performance
> - Confusion-matrix heatmaps revealing which subclasses get mixed up
> - Dendrograms showing learned class hierarchies
> - CNN feature visualizations and comparison to classical embeddings

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Please adhere to PEP8 style, document your code, and update this README if you add new notebooks or scripts.

---

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` for details.
   