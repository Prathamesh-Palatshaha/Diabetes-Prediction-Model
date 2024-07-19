# Diabetes Prediction Model

## Overview

This project involves predicting the onset of diabetes using a Support Vector Machine (SVM) model. The dataset used is the Pima Indian Diabetes Dataset from Kaggle. The goal is to build a predictive model that can determine whether an individual is likely to develop diabetes based on various health metrics.

## Dataset

The dataset can be downloaded from [Kaggle's Pima Indian Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database). It contains information about various health indicators such as glucose level, blood pressure, BMI, age, and more.

### Data Fields

- `Pregnancies`: Number of times pregnant.
- `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
- `BloodPressure`: Diastolic blood pressure (mm Hg).
- `SkinThickness`: Triceps skin fold thickness (mm).
- `Insulin`: 2-Hour serum insulin (mu U/ml).
- `BMI`: Body mass index (weight in kg/(height in m)^2).
- `DiabetesPedigreeFunction`: Diabetes pedigree function.
- `Age`: Age in years.
- `Outcome`: Class variable (0: non-diabetic, 1: diabetic).

## Project Structure

The repository contains the following files:

- `README.md`: Project overview and setup instructions.
- `Diabetes_Prediction_SVM.ipynb`: Jupyter Notebook with the data analysis, preprocessing, model training, and evaluation.
- `diabetes.csv`: The dataset used for training and testing (not included, must be downloaded from Kaggle).

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Preprocessing

The following preprocessing steps were performed:

1. Handling missing values: Filling or dropping missing values in the dataset.
2. Feature scaling: Standardizing the features to have mean = 0 and variance = 1.
3. Splitting the data: Dividing the dataset into training and testing sets.

## Model Training

The default SVM model from scikit-learn was used for training. The steps involved are:

1. Importing the necessary libraries and the dataset.
2. Preprocessing the data.
3. Splitting the data into training and testing sets.
4. Training the SVM model.
5. Evaluating the model's performance using metrics accuracy.

## Evaluation

The model's performance was evaluated using the following metrics:

- Accuracy

## Results

The SVM model provided a strong baseline for predicting diabetes onset. Further improvements can be made by trying different algorithms, feature selection methods, and hyperparameter tuning.

## Conclusion

This project demonstrates how to build an SVM model to predict the onset of diabetes. The steps include data preprocessing, feature scaling, model training, and evaluation. The notebook serves as a starting point for further exploration and improvement.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Prathamesh-Palatshaha/Diabetes-Prediction-Model.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Diabetes-Prediction-Model
   ```

3. Download the dataset from Kaggle and place `diabetes.csv` in the project directory.

4. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

5. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Diabetes_Prediction_SVM.ipynb
   ```

6. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

## Acknowledgments

- [Kaggle](https://www.kaggle.com) for providing the Pima Indian Diabetes dataset.
- [Scikit-learn](https://scikit-learn.org) for the machine learning tools.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

Happy coding!
