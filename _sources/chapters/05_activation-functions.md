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

import torch 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

# import the diabetes dataset
diabetes = datasets.load_diabetes()

# convert the data to a pandas dataframe
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

```


Activation Functions
====================

So far, we have looked at examples where the output is a weighted sum of the inputs. Generally, you'd use classical regression software in that case rather than torch, since the classical software provies greater speed and interpretability for linear regression. The benefit of deep learning is the ability to estimate nonlinear functions, which are generally beyond the scope of linear regression.

Nonlinearity is introduced to deep learning by using **activation functions**, which are simple, nonlinear functions that introduce complexity into the model. In this chapter, we will look at three of the most common activation functions: rectified linear units (ReLU), Sigmoid, and tanh. Let's use the ReLU activation function as a specific example to understand how they work in a deep learning model.

## **ReLU Activation Function**

Activation functions get their name from the Restricted Linear Unit (ReLU) activation function. It defines a threshold for its input and sets the output to zero if the input is below the threshold. Otherwise, the input is the same as the output. So, you could say that the output is active above the threshold and inactive below the threshold. Here is what the ReLU function looks like as it maps its input to its output:

![ReLU function](img/relu.png)

The power of the ReLU activation function is that it introduces conditional statements like "if... then..." into the model. This allows the model to have different behavior depending on its inputs, rather than always giving the same weight to the same column of inputs. In order to make use of the complex conditionality, ReLU activation functions are typically put between linear layers. Here is an example that makes the linear regression model for diabetes progression into a nonlinear dep learning model:


```{code-cell}
# define the torch model
class NonlinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(NonlinearRegressionModel, self).__init__()
        self.linear_1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear_2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU(hidden_dim)


    def forward(self, x):
        result = self.linear_1(x)
        result = self.relu(result)
        result = self.linear_2(result)
        return result
```

Note that the model now has multipl linear layers with that operate on different dimensions and with a ReLU layer between them. We can now create an instance of the nonlinear model and train it:

```{code-cell}
# instantiate the model
my_nonlin = NonlinearRegressionModel(input_dim=2, hidden_dim=5)

# define the loss function and the optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(my_nonlin.parameters(), lr=1)

# convert the data to tensors
x_data = torch.tensor(df[['bmi', 's6']].values, dtype=torch.float32)
y_data = torch.tensor(df['target'].values, dtype=torch.float32).unsqueeze(-1)

# train the model by repeatedly making small steps toward the solution
for epoch in range(40000):
    # Forward pass
    y_pred = my_nonlin(x_data)
    
    # Compute and print loss
    loss = criterion(y_pred, y_data)
    if epoch % 1000 == 0:
        print(f'Epoch: {epoch} | Loss: {loss.item()}')
    
    # Zero gradients, perform a backward pass, and update the parameters.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The training loss seems stable after about 30000 iterations.


```{code-cell}
# print out the model parameters:
for param in my_nonlin.parameters():
    print(param)
```


```{code-cell}
fit_nonlin = pd.DataFrame({'fitted':y_pred.squeeze(1).detach().numpy()})
fit_nonlin['target'] = df['target']

# plot model fits
fit_plot = sns.regplot(data=fit_nonlin, x="fitted", y="target")
```