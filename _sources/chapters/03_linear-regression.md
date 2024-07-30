---
jupytext:
  formats: qmd:myst
  text_representation:
    extension: .qmd
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [remove-cell]
import os

os.chdir("..")
```

Linear Regression
===================

We can use deep learning to estimate a linear regression model.


Begin by importing the necessary packages.

```{code-cell}
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

import torchinfo
```

Import the data
------------------

This example uses data on the progression of diabetes in 442 people with the disease. The inputs are medical variables that were measured in a physical exam: age; sex; blood pressure; body mass index (BMI); blood glucose; LDL, HDL, and total cholesterol. The measurement of disease progression is unitless and based on an unknown calculation. The data is built into the `sklearn` package's `datasets` module.

The predictor variables have already been centered and scaled, so they have mean zero and unit standard deviation. The measure of diabetes has not been centered or scaled.

```{code-cell}
# import the diabetes dataset
diabetes = datasets.load_diabetes()

print(diabetes.DESCR)

# convert the data to a pandas dataframe
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# plot the combinations of variables
sns.pairplot(df)
```


It looks like the target variable is growing with BMI and blood glucose. Let's create a linear model to examine that.


```{code-cell}
# create a linear model with BMI and blood glucose.
# estimate the model
my_model = smf.ols('target ~ bmi + s6', data=df).fit()

# check the model fit
my_model.summary()

# generate model fits 
fit = my_model.get_prediction(df).summary_frame()
fit['target'] = df['target']

# plot model fits
fit_plot = sns.regplot(data=fit, x="mean", y="target")
fit_plot.set(xlabel='Predicted', ylabel='Actual')
plt.show()
```



Linear regression via deep learning
-------------------------------------

Now we will estimate the same model via deep learning. The process is broken into three steps:
1. Define a model
2. Instantiate the model
3. Fit the model

### Define the model

Recall that to create a Torch model, we define a Python class that inherits the base class `torch.nn.Module`.

```{code-cell}
# define the torch model
class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
```


### Instantiate the model

Now that the model class is defined, we need to create an instance of it. This is called instantiating the model.

```{code-cell}
# instantiate the model
my_dl = LinearRegressionModel(input_dim=2)
```


### Train the model

We now have a model object stored in the computer's memory but its parameters have not been trained to match the data. Here is how to do that.

```{code-cell}
# define the loss function and the optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(my_dl.parameters(), lr=1)

# convert the data to tensors
x_data = torch.tensor(df[['bmi', 's6']].values, dtype=torch.float32)
y_data = torch.tensor(df['target'].values, dtype=torch.float32).unsqueeze(-1)

# train the model by repeatedly making small steps toward the solution
for epoch in range(10000):
    # Forward pass
    y_pred = my_dl(x_data)
    
    # Compute and print loss
    loss = criterion(y_pred, y_data)
    if epoch % 1000 == 0:
        print(f'Epoch: {epoch} | Loss: {loss.item()}')
    
    # Zero gradients, perform a backward pass, and update the parameters.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Check the model
------------------------------------------------------

Now we can look at the actual parameters values, which is only practical in very small deep learning models.

```{code-cell}
# print out the model parameters:
for param in my_dl.parameters():
    print(param)
```

The main thing to notice here is that the  parameters are identical to the linear regression coefficient estimates, which provide some support for the idea that we have specified a model with the same structure as linear regression.

```{code-cell}
fit_lin = pd.DataFrame({'fitted':y_pred.squeeze(1).detach().numpy()})
fit_lin['target'] = df['target']

# plot model fits
fit_plot = sns.regplot(data=fit_lin, x="fitted", y="target")
```

