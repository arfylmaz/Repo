import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Load the CSV file
df = pd.read_csv('chard_features.csv')

# Select columns 2 to 7 (1 to 6 in zero-based index)
df_selected = df.iloc[:, 1:7]

# Plotting scatter graphs, polynomial regression lines, and displaying correlation coefficients
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
axs = axs.flatten()

for i, column in enumerate(df_selected.columns):
    # Scatter plot
    sns.scatterplot(x=df_selected.index, y=df_selected[column], ax=axs[i])

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(df_selected.index.values.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, df_selected[column])
    y_poly_pred = model.predict(X_poly)

    # Plotting the polynomial regression line
    sns.lineplot(x=df_selected.index, y=y_poly_pred, ax=axs[i], color='red')

    # Calculate correlation coefficient
    #corr_coef = np.corrcoef(df_selected.index, df_selected[column])[0, 1]
    axs[i].set_title(f'{column}')
    #axs[i].set_title(f'Scatter and Polynomial Regression for {column}\nCorrelation: {corr_coef:.2f}')
    axs[i].set_xlabel('Index')
    axs[i].set_ylabel(column)

plt.tight_layout()
plt.show()
