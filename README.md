### **Obesity Risk Prediction Using Machine Learning**  

## **Project Overview**  
Obesity is a growing public health concern, often linked to serious conditions like cancer, diabetes, and cardiovascular disease. However, in Nigeria, limited access to structured health data makes it difficult to build effective predictive models. This project explores how machine learning can be used to predict obesity risk levels using a dataset that includes demographic, health, and lifestyle factors. The goal was to test different machine learning models and determine the best approach for accurate predictions.  

## **Dataset Used**  
The dataset consists of key health indicators such as **age, gender, BMI, blood pressure, cholesterol, blood sugar levels, physical activity, and diet quality.** The target variable is obesity risk, categorized into three levels:  
- **Low Risk (0)**  
- **Medium Risk (1)**  
- **High Risk (2)**  

The data is synthetic and here is a document that explains the generation of the synthetic dataset: [Link](https://docs.google.com/document/d/1ILz1-6Ef8_rBtbUFg1d3rqaP5tDzqSmBKqNjcanc52c/edit?usp=sharing)
I preprocessed the data by normalizing numerical features, encoding categorical variables, and splitting it into training (70%), validation (15%), and test (15%) sets.  

---

## **Model Implementation and Findings**  
Six different models were trained and evaluated, including five neural network configurations and one classical machine learning model (XGBoost). Each model was assessed based on accuracy, precision, recall, F1-score, and AUC.  

### **Model Performance Comparison**  

| Model | Optimizer | Regularization | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision | AUC |
|--------|------------|---------------|--------|---------------|--------|--------------|----------|----------|--------|-----------|------|
| **Instance 1** (Baseline) | Adam (Default) | None | 50 | No | 2 | 0.001 | 89.33% | 0.9329 | 0.9333 | 0.9341 | 0.9939 |
| **Instance 2** | Adam | L2 | 50 | No | 3 | 0.001 | 90.67% | 0.9404 | 0.9400 | 0.9412 | 0.9861 |
| **Instance 3** | RMSprop | L1_L2 | 50 | No | 3 | 0.0005 | 84.67% | 0.7958 | 0.8400 | 0.8105 | 0.9965 |
| **Instance 4** | Adam | L1 | 50 | Yes | 4 | 0.0007 | 88.00% | 0.8701 | 0.8800 | 0.8762 | 0.9861 |
| **Instance 5** *(Best Neural Network)* | RMSprop | L2 + BatchNorm | 50 | No | 4 | 0.0005 | **92.67%** | **0.9273** | **0.9267** | **0.9301** | **0.9965** |
| **XGBoost Model** | - | L2 | 100 Boosting Rounds | Yes | - | 0.01 | 90.67% | 0.9105 | 0.9100 | 0.9120 | 0.9105 |

---

## **Which Model Performed Best?**  
The best-performing model was **Instance 5**, which used a neural network with **RMSprop optimizer, L2 regularization, and Batch Normalization**. It achieved **92.67% accuracy and an AUC score of 0.9965**, making it the most reliable model for obesity risk prediction.  

The **XGBoost model also performed well**, with **90.67% accuracy and an AUC score of 0.9105**, demonstrating that classical machine learning models can still be competitive. However, the deep learning model had a slight edge in terms of generalization and precision.  

---

## **XGBoost vs Neural Networks – Key Differences**  
While both approaches performed well, there are some trade-offs to consider:  
- **Neural Networks (Instance 5)** showed better generalization, with higher accuracy and AUC.  
- **XGBoost* was computationally more efficient and required less fine-tuning compared to deep learning.  
- Regularization techniques played a key role in improving both models, helping to reduce overfitting.  
- **XGBoost* was more interpretable, making it a strong alternative for real-world applications where model explainability is crucial.  

---

## **Hyperparameters Used in XGBoost**  
| Parameter | Value |
|-----------|-------|
| Objective | `multi:softmax` |
| Number of Classes | `3` |
| Learning Rate | `0.01` |
| Boosting Rounds | `100` |
| Max Depth | `6` |
| Subsample | `0.8` |
| Random State | `42` |
| Early Stopping Rounds | `10` |

These hyperparameters were optimized to improve model performance and prevent overfitting.

---

## **Graph of the Best Model (Instance 5)**  
*I*

---

## **Video Presentation**  
A walkthrough of the project, findings, and insights is available in the video presentation below:  
[**Video Link**]  

---

## **Final Thoughts**  
This project demonstrated how **both deep learning and classical machine learning** can be applied to predict obesity risk levels with high accuracy. The best model was a **neural network with batch normalization**, but **XGBoost* proved to be a strong alternative, especially for cases where explainability and efficiency matter.   

This study provides a foundation for using **AI-driven health solutions** to assess and mitigate obesity risks more effectively.  

---

## **Acknowledgments**  
Special thanks to the few **health research communities in Nigeria** emphasizing the importance of data collection especially on health risks like obesity.  

---
