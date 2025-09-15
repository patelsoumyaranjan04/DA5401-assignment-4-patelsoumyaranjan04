# DA5401 — Assignment 4

- **Name**: Soumya Ranjan Patel  
- **Roll Number**: DA25M029
- **Dataset**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud



## Repository Structure

```
DA5401-assignment-4-patelsoumyaranjan04/
├── DA5401_A4.ipynb 
└── README.md 
```


- **`DA5401_A4_Clustering_Resampling.ipynb`**  
  Contains the full implementation of the assignment:
  - Part A: Class distribution analysis and baseline model  
  - Part B: Resampling methods GNN and CBU  
  - Part C: Comparison of results and Final Conclusion  

- **`creditcard.csv`**  
  The dataset used for training and testing. This file must be present in the same directory as the notebook.  


---

## How to Run

1. Clone/download the repository.  
2. Ensure you have Python 3.8+ with Jupyter Notebook installed.  
3. Install required packages (if not already installed):  
   ```bash
   pip install numpy pandas matplotlib scikit-learn imbalanced-learn
4. Download the dataset from the provided link and extract it.
5. Place the file creditcard.csv in the same folder as the notebook.

## 📊 Result Summary

- **Baseline (imbalanced data):**
  - High precision (0.85) but moderate recall (0.61).
  - Model misses a significant number of fraud cases.

- **GMM-based oversampling (v1 & v2):**
  - Recall improves drastically (0.85) → fewer frauds missed.
  - Precision collapses (0.09) → many false alarms.
  - F1 score drops sharply, indicating poor overall balance.

- **Clustering-Based Undersampling (CBU):**
  - Mitigates imbalance, but precision remains poor.
  - Minority distribution is too sparse for reliable oversampling.
 
  <img width="845" height="597" alt="image" src="https://github.com/user-attachments/assets/beb331ed-51df-4a0e-bece-67f6648e32c3" />


### 🚀 Conclusion
GMM and CBU improved recall but severely hurt precision due to unrealistic synthetic samples diverging from true fraud patterns.  
Hence, **GMM-based oversampling is not effective for this dataset**.  
Alternative strategies like **cost-sensitive learning, SMOTE/ADASYN, or hybrid ensemble methods** may be better suited for handling the extreme imbalance.
