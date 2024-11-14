# Predicting timely cancer diagnosis using demographic and societal drivers of healthcare inequities

## Background
<img width="1130" alt="Screenshot 2024-11-12 at 2 38 38 PM" src="https://github.com/user-attachments/assets/9ac943a4-dfa7-49d8-9627-e6c7cb04f013">

### Metastatic TNBC requires most urgent and timely treatment

"Metastatic TNBC is considered the most aggressive TNBC and requires most urgent and timely treatment. Unnecessary delays in diagnosis and subsequent treatment can have devastating effects in these difficult cancers. **Differences in the wait time to get treatment is a good proxy for disparities in healthcare access**."

The goal of building predictive models is to **detect relationships between demographics of the patient with likelihood of getting timely treatment**

### The Dataset 
* A rich, *real-world dataset* which contains information about demographics, diagnosis and treatment options, and insurance provided about patients who were diagnosed with breast cancer from 2015-2018
* It was enriched with third-party geo-demographic data to provide views into the socio-economic aspects that may contribue to health equity
* Further enriched with zip code-level toxicology data NASA/Columbia University
* *Each row corresponds to a single patient and their Diagnosis Period*
<img width="620" alt="Screenshot 2024-11-12 at 2 57 24 PM" src="https://github.com/user-attachments/assets/71c622c3-dab3-45d5-9be3-fc09d99f7959">

### Target

| Target | Type | Information|
| :-- | :-- | :-- |
| DiagPeriodL90D | Binary target: Values are 0 and 1 | Diagnosis Period Less Than 90 Days; this is an *indication of whether the cancer was diagnosed within 90 days* |

## Purpose

The purpose of this notebook is to implement the tools learned thus far to develop a supervised learning model that has

- *Good performance metrics*
  - Performs well on both test and train datasets (low error rate on training set and low generalization gap)
- *Good fit*
  - Perfoms well at the intersection between overfitting (high variance) and underfitting (high bias)
- *Generalizability*
  - Performs well on new, unseen data with low bias and low variance

## Goal

The goal is to **implement a reproducible, transparent model preparation process**
<img width="635" alt="Screenshot 2024-11-12 at 2 51 51 PM" src="https://github.com/user-attachments/assets/00a65d1a-632c-4972-8323-d966c4534c52">

## Exploratory Data Analysis 

EDA is the process of exploring data to discover insights, identify patterns, establish relationships and trends, and test assumptions. It is an iterative process comprised of Data Cleaning, Data Exploration, and Feature Engineering. **The goal of successful EDA is to produce high quality data and identify the most useful features/variables to be used in subsequent modeling**. Therefore proper EDA is critical and should be performed with great consideration, patience, attention to detail.

### I. Data Cleaning
1. **Deal with missing values**
   - Data cleaning/munging/wrangling is the *first step of EDA*
   - Detect and eliminate problems in the dataset that would prevent further analysis
   - Variables that were missing 5% or more of total records were removed from the dataset to *avoid introducing bias and gross inaccuracies*
2. **Handle Categorical Variables**
   - Tautologous variables were dropped from the dataframe
   - *Label encoding* was used in lieu of get_dummies()
3. **Deal with outliers**
  - Ouliers were *tranformed into something harmless*
<img width="1367" alt="Screenshot 2024-11-12 at 2 58 55 PM" src="https://github.com/user-attachments/assets/321c5c2f-c931-4974-ae21-e68566fadad8">

4. **Scale the features**
   - *Standard scaler was applied to continuous variables*

5. **Normalize the features** (as much as possible)
  - *Shapiro test* statistics
  - *Yeo-Johnson* Transformation

## Data Cleaning produced a high-quality dataset suitable for Featue Engineering and Modeling
Importantly, this clean dataframe
* Lacks missing values and records
* Has one-hot encoded/labeled categorical variables
* Has log-transformed continuous variables (where applicable)
* Has scaled and normalized continuous variables

The clean dataframe is a high-dimensional real-world dataset comprised of **12,624 records** and **73 variables**.

### II. Data Exploration 
#### What does my target look like?
The target is **class-imbalanced and not linearly separable**
![image](https://github.com/user-attachments/assets/5f360494-2212-48cf-8a3d-e10848ff922c)![image](https://github.com/user-attachments/assets/df7bdc39-1c2f-419d-87f1-fb40fa201ae7)

<img width="489" alt="Screenshot 2024-11-12 at 3 03 10 PM" src="https://github.com/user-attachments/assets/a4ca09d1-a5cb-4459-bd29-086080b9c2bb">

### What's my base? Establishing a "base model" is industry best practice

My Mentor taught me that establishing a base model and baseline performance metrics is an industry best practice. This enables direct quantification of changes made throughout the process of selecting and identifying a model. Establishing a base model and baseline performance metrics  involves running a model on *all features in a the clean dataset*. 

**Baseline SVM Classifier model**
- Train accuracy score: 0.628
- Test accuracy score: 0.612

**Compare model accuracy with null accuracy**
- Null accuracy score: 0.625

### III. Feature Engineering 

<img width="593" alt="Screenshot 2024-11-12 at 3 04 26 PM" src="https://github.com/user-attachments/assets/08311955-8a1e-432a-967a-374bb691d3f5">

<img width="600" alt="Screenshot 2024-11-12 at 3 13 13 PM" src="https://github.com/user-attachments/assets/1222d6c3-3a3d-4b5a-8cda-4ce3528bf232">

## Nonparametric Classification models on post-EDA clean dataset with new features 
### KNN
The KNN accuracy scores with statistically significant feature set showed overfittingand poor predictive power. While the predictive performance is relatively poor, **the generated/engineered features were found to have a significant impact on the out-of-sample predictions**

<img width="323" alt="Screenshot 2024-11-12 at 3 19 54 PM" src="https://github.com/user-attachments/assets/b9d40cec-f75f-48f9-a85b-b8333fcadc1c">

<img width="407" alt="image" src="https://github.com/user-attachments/assets/0fcc129f-5ad3-4dda-8850-7eefa0c977b6">

### Random Forest
Random Forest models also had low accuracy scores and poor predictive power
<img width="589" alt="Screenshot 2024-11-12 at 3 21 38 PM" src="https://github.com/user-attachments/assets/a0739109-ae76-4f00-8fd6-ebe7045c920b">

### Gradient Boost Classification models showed the most potential 

<img width="592" alt="Screenshot 2024-11-12 at 3 23 02 PM" src="https://github.com/user-attachments/assets/8b9b85c4-fc4f-4733-a300-7160be84353c">

### Optimization efforts
**The Recall score was prioritized during optimization efforts**. The target indicates whether ornote a patient received a metastatic cancer diagnosis within 90 days of screening. *A false negative equates to a failure to diagnose*, which has serious consequences for metastatic TNBC.

#### GridSearchCV identified optimal parameters and improved model performance 

<img width="634" alt="Screenshot 2024-11-12 at 3 25 29 PM" src="https://github.com/user-attachments/assets/2eb7f36b-f792-43f3-9abe-6b2baf9b0a80">

## Gradient Boost Classification on post-EDA dataset is the best model for this task 
<img width="603" alt="Screenshot 2024-11-12 at 3 26 35 PM" src="https://github.com/user-attachments/assets/8c6fa76c-c3fd-4a6f-a745-8ccf908a0c5b">

- *Performed better than Dummy Classifier*
- *Better performance metrics* than base model
- With Gradient boost models, one can *readily obtain feature importance data*

## Improving the GB model (a model can always be improved!)
1. The GB model built here has higher Accuracy and AUC-ROC scores than those in the "[WiDs Data Exploration | ML | Starter](https://www.kaggle.com/code/ddosad/wids-data-exploration-ml-starter)" 
- 3rd highest-voted notebook
- Served as a good "sanity check" during model development
2. Real-world clinical data can be messy, complex and tricky to navigate at times
- Might need more records to increase predictive power
  - The post-EDA "clean" dataset had 12,624 records and a class-imbalanced target (0: 37.5% | 1: 62.5%)
  - Some models might need >15K, 50K, or even 500K to have strong predictive power
- Medical diagnosis models can be particularly challenging due to limited samples and nature of data
3. Go back to Feature Engineering phase with feature importance data obtained during GB model optimization
4. Research new and different supervised learning Classfication models
  - XGBost, LightGBM, CatBoost, Extra Trees, Multilayer Perceptron, etc.

## The EDA process executed herein is clear, reproducible and thorough
Importantly, the Feature Engineering performed was *demonstrably useful*. 
