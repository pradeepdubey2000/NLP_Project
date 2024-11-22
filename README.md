# 🛡️ Cybersecurity Vulnerability Predictor - NLP Project 🛡️  

Welcome to the **Cybersecurity Vulnerability Predictor** repository! This project leverages advanced machine learning and NLP techniques to predict and analyze cybersecurity vulnerabilities effectively. It features interactive visualizations, similarity searches, and a detailed exploration of CVE (Common Vulnerabilities and Exposures) data.  

---

## 🌟 Key Features  

1. **Vulnerability Prediction**  
   - Predict potential vulnerabilities using advanced ML models.  
   - Gain insights into base scores, impact scores, and exploitability scores.  

2. **CVE Analysis**  
   - Interactive visualizations for exploring trends and relationships in CVE data.  
   - Filter and analyze vulnerabilities across categories.  

3. **CVE Similarity Search**  
   - Search for vulnerabilities similar to a given description using NLP and cosine similarity.  
   - Quickly identify related CVEs for better threat assessment.  

4. **Detailed Experimentation**  
   - Comprehensive notebooks detailing data extraction, cleaning, embeddings, model building, and predictions.  

---

## 📂 Repository Structure  

### 🗂️ **Main Files and Directories**  

```
NLP_Project/
├── MODELS2/                             # Contains trained ML models for predictions
│   ├── Access_Complexity_best_catboost_model.pkl         # Model for Access Complexity prediction
│   ├── Access_Complexity_label_encoder.pkl               # Label encoder for Access Complexity
│   ├── Access_Vector_label_encoder.pkl                   # Label encoder for Access Vector
│   ├── Availability_Impact_best_catboost_model.pkl       # Model for Availability Impact prediction
│   ├── Availability_Impact_label_encoder.pkl             # Label encoder for Availability Impact
│   ├── Confidentiality_Impact_best_xgboost_model.pkl     # Model for Confidentiality Impact prediction
│   ├── Confidentiality_Impact_label_encoder.pkl          # Label encoder for Confidentiality Impact
│   ├── Integrity_Impact_best_xgboost_model.pkl           # Model for Integrity Impact prediction
│   ├── Integrity_Impact_label_encoder.pkl                # Label encoder for Integrity Impact
│   ├── accessVector_best_catboost_model.pkl              # Model for Access Vector prediction
│   ├── base_score_xgb_regressor.pkl                      # XGBoost model for base score regression
│   ├── exploitability_Score_xgb_regressor.pkl            # XGBoost model for exploitability score
│   ├── impact_Score_xgb_regressor.pkl                    # XGBoost model for impact score regression
├── notebook/                          # Jupyter notebooks for detailed experimentation
│   ├── 0.Data_Extraction_nlp.ipynb                  # Data extraction and sourcing
│   ├── 1.Data_Cleaning.ipynb                        # Data cleaning and preprocessing
│   ├── 2.description-embedding.ipynb                # Generating embeddings for descriptions
│   ├── 3.Merge_data.ipynb                           # Merging datasets for analysis
│   ├── 4.ml-model-building.ipynb                    # Machine learning model development
│   ├── 5.final-prediction.ipynb                     # Final model integration and predictions
├── pages/                             # Contains additional Streamlit app pages
│   ├── CVE_Analysis.py                              # Page for CVE data analysis
│   ├── CVE_similarity_search.py                    # Page for similarity search
│   ├── Detailed_Metrics.py                         # Page for detailed metrics and evaluations
├── final_project_report.docx         # Comprehensive project report
├── Demo_Video_NLP.mp4                # Demo video showcasing the app
├── home.py                           # Main home page of the Streamlit app
└── README.md                         # You are here!

```

---

## 🚀 Installation  

### Prerequisites  
Ensure you have the following installed:  
- **Python 3.8+**  
- Libraries: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`, `xgboost`, `catboost`, `plotly`  

### Steps  

1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/your-repo/NLP_Project.git  
   cd NLP_Project  
   ```  

2. **Install dependencies**:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Run the application**:  
   ```bash  
   streamlit run home.py  
   ```  

---

## 📘 How to Use  

1. **Home Page**  
   - Navigate to the home page for an overview and feature selection.  
   - Select a module from the sidebar: CVE Analysis or Similarity Search.  

2. **CVE Analysis**  
   - Analyze trends, distributions, and correlations in CVE data.  
   - Choose from multiple visualization types, including bar plots, scatter plots, and box plots.  

3. **CVE Similarity Search**  
   - Enter a vulnerability description to find similar CVEs.  
   - View results in JSON format with detailed scores and metrics.  

4. **Model Predictions**  
   - Predict access complexity, impact scores, and other attributes using trained ML models.  

---

## 📊 Notebooks and Experimentation  

- **`0.Data_Extraction_nlp.ipynb`**: Extract CVE data from various sources.  
- **`1.Data_Cleaning.ipynb`**: Clean and preprocess data for analysis and modeling.  
- **`2.description-embedding.ipynb`**: Generate embeddings for vulnerability descriptions using NLP.  
- **`4.ml-model-building.ipynb`**: Train ML models for predicting key attributes.  
- **`5.final-prediction.ipynb`**: Integrate models for end-to-end predictions.  

---

## 🎥 Demo  

🎬 **Watch the Demo**: Check out the [Demo Video](./Demo_Video_NLP.mp4) to see the application in action!  

---

## 📄 Report  

📖 **Detailed Report**: For a comprehensive overview of the project, refer to the [Final Project Report](./final_project_report.docx).  

---

## 💡 Models  

The `MODELS2` directory contains pre-trained models used in this project:  

- **Access Complexity Models**: Best-performing CatBoost models and label encoders.  
- **Impact Prediction Models**: XGBoost models for predicting confidentiality, integrity, and availability impacts.  
- **Score Prediction Models**: XGBoost regressors for base scores, exploitability scores, and impact scores.  

---

## 🤝 Contributors  

👨‍💻 **Team Members**:  
- **Pradeep Dubey**  
- **Aman Kumar**  
- **Apoorva Kakkar**  
- **Anshi Gupta**  

👨‍🏫 **Supervision**: *Dr. Gaurav Sir*  

---

## 📜 License  

This project is licensed under the [MIT License](./LICENSE).  

---

## ❤️ Acknowledgments  

- Thanks to the open-source community for providing tools and datasets.  
- Special thanks to our mentor for guidance throughout the project.  

---  

✨ **Feel free to explore, use, and contribute to the project!** 🎉  
