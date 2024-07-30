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

PyTorch basics
===============

[PyTorch](https://pytorch.org) (usually just called torch) is the package we'll use to create deep learning models. It's currently the dominant framework for deep learning, which is a title that PyTorch took over from TensorFlow sometime around 2020. There are similarities and differences between deep learning frameworks, but we aren't going to get into those comparisons.

Like all deep learning frameworks, torch relies on data being structured as **tensors**, which you can think of as matrices with possibly more than two dimensions. The software is designed to do calculations on tensors very quickly, especially when running on graphics processing units or **GPUs**. That's why NVIDIA, the company that makes most GPUs, has become so important in the past few years. But we can run deep learning models on our laptops' central processing units or **CPUs** since we are working with small examples here. This workshop won't cover GPU computation but using PyTorch it is very simple to convert a model to use a GPU if one is available.

Tensors
---------

Remember that a tensor is just like a matrix with possibly more than two dimnsions. In another context, we'd probably refer to a one-dimensional tensor as a vector and a two-dimensional tensor as a matrix. Lets look at some examples to make that more concrete. We'll begin with a very simple example: a one-dimensional tensor that contains the numbers zero through nine. First, we must make sure to load the `torch` package in Python.

```{code-cell}
import torch

# create a new tensor
ten = torch.tensor(range(10), dtype=torch.int)

print(ten)
```

You can derive some useful information from the printout of the tensor. First, the output is wrapped by `tensor(...)`, which lets you know that the object is a tensor (in a less simple example this might not have been obvious). The values are wrapped by a single set of square braces, `[...]`, which indicates that there is one dimension to the tensor. The actual values are printed out, so you know what numbers are held in the tensor. And finally, we are told what data type is used for each entry. Every value in the tensor must have the same data type, which in this case (on my computer) is `torch.int32`. This is because we specified the `dtype` when creating the tensor.

Now let's look at the shape of this tensor. There are two ways to do so: checking the `shape` property or running the `size()` method. Let's try both:

```{code-cell}
# check the shape property
print(ten.shape)

# run the size method
print(ten.size())
```

### Reshaping 

In both cases, we see that the size of the tensor is 10. But we can also represent the same data in other sizes like 1x10, 10x1, or 5x2. You can reshape a tensor by the `reshape`, `squeeze`, and `unsqueeze` methods. With `reshape`, you would specify a vector giving the nw sizes for each dimension.


```{code-cell}
# make the tensor wide
wide = ten.reshape([1, 10])
print(wide)
```

Notice that the result now is wrapped in two square brackets (`[[...]]`), indicating that there are two dimensions.

```{code-cell}
# make the tensor tall
tall = ten.reshape([10, 1])
print(tall)
```

This time the new shape was specified as ten rows and one column, which makes the tensor tall rather than wide. It is also printed in a way that makes it a bit more clear that there are two dimensions. We could specify any new shape that does not change the number of elements from 10. Here is an example with five rows and two columns:

```{code-cell}
# represent the data in two columns
mat = ten.reshape([5, 2])
print(mat)
```

The most common ways you'll reshape tensors in practice is by the `squeeze` and `unsqueeze` methods. The `squeeze` method eliminates a dimension of the tensor if its size is one but has no effect otherwise. The unsqueeze method is an inverse function: it creates a new dimension of size one. Examples may make this more clear. First, the `squeeze` method eliminates the specified dimension but only has effect if that dimension has size one.

```{code-cell}
tall.shape

tall.squeeze(0)

tall.squeeze(1)
```

And the `unsqueeze` method creates a new dimension of size one at a specified index:

```{code-cell}
ten.unsqueeze(0)

ten.unsqueeze(1)

ten.unsqueeze(-1)
```

That final example specifies the index to unsqueeze in a new way: by a negative index. This means to count back from the last dimension rather than forward from the beginning. So -1 means to unsqueeze the tensor and create a new dimension after the final current dimension.

```{code-cell}
ten.shape

ten.unsqueeze(-1).shape
```

Defining a model
-----------------

To use torch for deep learning, you will need to define a model for your data. The model converts inputs to outputs, which are both in the form of tensors. Defining a model means describing how the inputs are turned to outputs. For instance, in a linear regression model the output is a weighted sum of the inputs. This turns out to be pretty easy to represent in torch, as we'll see later.

A torch model is defined by writing a Python class that inherits `torch.nn.Module`. It must have two methods (but may have more):
1. `__init__()` is run when the model object is created and defines the layers that compose the model.
2. `forward()` takes the input tensor(s) and converts them into an output using the layers that are defined in `__init()__`.

Here is the definition of a linear regression model in torch:

```{code-cell}
class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
```

If you've never defined a class in Python, there are some things here that may be new to you. The class definition begins with the keyword `class`, then specifies the class name (`LinearRegressionModel`), and the super class from which this new model class inherits (`torch.nn.Module`). This line ends with a colon and is followed by indented lines that all belong to the class definition.  The two parts of that class definition are the two methods `__init__()` and `forward()`. The first argument of each method is `self`, which is a Python requirement for a method definition. it allows an object of the class to refer to itself.

### `__init__()` method
The `__init__()` method is only run once: when the model object is first created. Its purpose is to set up the model object, especially to define layers to be used in the model. There are only three lines of code in the method definition, so we can look at them all.

#### `def __init__(self, input_dim, output_dim=1):`
Here the `def` keyword tells python that we are starting to define a method with the name `__init__`. If you've ever written a function in python you may recognize the pattern here: first the `def` keyword, then the name, then a list of arguments contained within parenteses, and finally a colon to indicate that the following lines will describe what the method does.

Here there are three arguments, `self`, `input_dim`, and `output_dim`. The first, `self`, is standard for methods in a python class. It is a special name that lets objects of this class refer to themselves. That's important because when we create a model of the class `LinearRegressionModel`, we want it to be unaffected by other models even if they are of the same class.

The second argument, `input_dim`, is used to tell the model how many columns of predictors will be used in a particular case. The final argument, `output_dim`, tells the model how many outputs to generate. We provide a default value of one so this argument would only need to be specified when creating a regression model with more than one output.

#### `super(LinearRegressionModel, self).__init__()` 
This line of code is necessary to initialize the parent class (aka super class) from which the model inherits (remember that a torch model inherits from `torch.nn.Module`). The only thing to customize for your model is the name of the model class (`LinarRegressionModel` in this case), which must match the name of the model class.

#### `self.linear = torch.nn.Linear(input_dim, output_dim)`
Here we define a linear layer for the model. A [linear layer in torch](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) creates its outputs as weighted sums of its inputs. The size of the layer has to be set at the time of model creation, and that is handled here by the `input_dim` and `output_dim` parameters. The method definition provides a default value of one for `output_dim`, which means that unless otherwise instructed, the model will calculate a single weighted sum each time the linear layer is run. The number of terms that go into the weighted sum is given by the `input_dim` argument.

### `forward()` method
The `forward()` method takes the model inputs and uses them to calculate the outputs. This method is required to have the name `foward` in a torch model. The `forward` method wil be run once per epoch for each row of training data. Once again there are only three lines of code so we can look at them individualy.

#### `def forward(self, x):`
This is another method definition, so it follows the same pattern as before: first the `def` keyword, then the method name, then a list of arguments is contained within parentheses. Again, as is standard for methods in python, the first argument is `self`. The second argument, `x`, will be used to provide a row of predictors to the model.

#### `y_pred = self.linear(x)`
Here we calculate the model output as a weighted sum of the inputs. This line uses the linear layer that was defined in the `__init__()` method and applies it to `x`, the row of data that is being used to generate an output.

#### `return y_pred`
Now that we've calcluated the output for this row of data as `y_pred`, we pass it back so the user can do something with it.

