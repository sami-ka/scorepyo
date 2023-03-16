# Scorepyo

<p align="left">
  <a href="https://github.com/drskd/scorepyo/releases/"><img alt="Version" src="https://img.shields.io/github/v/release/drskd/scorepyo?color=orange&label=Release" /></a>
  <a href="https://pypi.org/project/scorepyo/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/scorepyo" /></a>
  <a href="https://github.com/drskd/scorepyo/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/pypi/l/scorepyo" /></a>
  <a href="https://pypi.org/project/scorepyo/"><img alt="pypy latest version" src="  https://img.shields.io/pypi/v/scorepyo" /></a>
  <a href="https://pypi.org/project/scorepyo/"><img alt="downloads" src="  https://img.shields.io/pypi/dm/scorepyo" /></a>


</p>

<!-- ![Scorepyo](./source/_static/scorepyo_logo.jpg) -->
<!-- ![Scorepyo](./source/_static/full_scorepyo_logo.png) -->
<!-- <img src="./docs/images/square_logo_1000pxl.svg" width="200" height="200"> -->
<img src="./docs/images/logo_zoom.PNG" width="230">

<!-- <div style="text-align: left">Image by <a href="https://www.freepik.com/free-vector/scorpion-branding-logo-template_21251044.htm#page=2&query=scorpio&from_query=scorpio%20logo&position=3&from_view=search&track=sph">Freepik</a></div>
<br /> -->
<br />

**Scorepyo** is a python package for binarizing features, and creating risk-score type models for binary classification, based on data. The created models can be used like other ML models, with fit and predict methods.
<br /> <br />

### *Example on Scikit-learn breast cancer dataset*
<br />

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import average_precision_score
from scorepyo.models import EBMRiskScore

# Getting data
data = load_breast_cancer()
data_X, data_y = data.data, data.target

X = pd.DataFrame(data=data_X, columns=data.feature_names)
y = pd.Series(data_y)

scorepyo_model = EBMRiskScore()

scorepyo_model.fit(X, y)

scorepyo_model.summary()
```
#### Feature-point card 


| Feature         | Description            |   Point(s) | |      |
|:----------------|:-----------------------|--------:|---: |--------:|
| worst concave points   | worst concave points >= 0.14 | -2         || ...   |
| worst radius | worst radius >= 16.8 | -2          | |+ ... |
| mean texture     | mean texture >= 20.72     | -1          | |+ ... |
| worst area     | worst area < 553.3    | 1          | |+ ... |
|                 |                        |         |  |        |
|                 | |<div style="text-align: right"> **SCORE =**</div> | |**...**|

####   Score card     
| SCORE   | -5.0   | -4.0   | -3.0   | -2.0   | -1.0   | 0.0    | 1.0     |
|:--------|:-------|:-------|:-------|:-------|:-------|:-------|:--------|
| RISK    | 0.00%  | 1.03%  | 1.13%  | 45.95% | 84.09% | 98.10% | 100.00% |

<br />
<br />

# Installation
Python 3.8, 3.9

```shell
pip install scorepyo
```
<br />

# Documentation

Want to know more? 
<div style="text-align: left"> <a href="https://drskd.github.io/scorepyo/welcome.html" target="_blank">Check the documentation!</a> <br />
<br /> <br />

# Risk-Score model

Risk-score model are mostly used in medicine, justice, psychology or even credit application. They are widely appreciated, as the computation of the risk is fully explained with two simple tables:
* a point-card, with points to sum depending on features value;
* a score-card that associate a score to a risk. <br />Final score is computed by summing the points defined by the point-card.

The points should be small integers, in order to be easily manipulated and remembered by people using them. 

You can find hereafter another example of such risk-score model for assessing a stroke risk:

<!-- ![Risk-score example for stroke risk](./source/_static/example_risk_score.PNG) -->
<img src="./docs/depth_model/example_risk_score.PNG" width="400">

<div style="text-align: left"> <a href="https://en.wikipedia.org/wiki/CHA2DS2%E2%80%93VASc_score#CHADS2" target="_blank">Source</a> <br>
<!-- Berk Ustun and Cynthia Rudin<br>
Journal of Machine Learning Research, 2019.</div>  
<br /> -->
<br />
The extreme interpretability of such model is especially useful since it helps to understand and trust the model decision, but also to easily investigate fairness issues and make sure to satisfy legal requirements, and eventually remember it.

The simple computation also allows to write it down on a piece of paper for usage.

<br />

# Components of Scorepyo
The **Scorepyo** package provides two components that can be used independently:
* **Automatic feature binarizer**
* **Risk-score model**

##  Automatic feature binarizer
Datasets usually comes with features of various type. Continuous feature must be binarized in order to be used for risk-score model. 
Scorepyo leverages the awesome <a href="https://github.com/interpretml/interpret" target="_blank">interpretML</a> package and their EBM model to automatically extract binary features.

## Risk score model
The risk-score model can be modeled as an optimization problem with 3 sets of decision variables:
* Subset of binary features to use
* Points associated to each selected binary feature
* Probabilities associated to each possible score

The objective function is the optimization of a binary classification metric (e.g. the logloss, ROC AUC, average precision) of the computed risk on training samples.


This formulation is already used in other packages such as <a href="https://github.com/ustunb/risk-slim" target="_blank">risk-slim</a> or <a href="https://github.com/jiachangliu/FasterRisk" target="_blank">FasterRisk</a>.

<br />

The novelty in **Scorepyo** is that it decomposes the model search into simple and easily customizable components:
* Ranking of binary features
* Enumeration maximization metric
* Probability calibration

It also drops the link with the sigmoid function when defining the probability of each score, in order to widen the search space of risk-score model.

<br /><br />


# Acknowledgements
> ### <a href="https://en.wikipedia.org/wiki/Standing_on_the_shoulders_of_giants" target="_blank">*Standing on the shoulders of giants*</a>
> #### <div style="text-align: right">*Bernard de Chartres* </div>
<br />

This package is built on top of great packages:
* <a href="https://github.com/interpretml/interpret" target="_blank">interpretML</a> for the binarizer
* <a href="https://github.com/dask/dask" target="_blank">Dask</a> to easily scale the costly enumeration step

# More context

To better understand the justification of automatically creating risk score model from data, or to not only round coefficients from logistic regression, I refer to the great introduction of this Neurips 2022 paper associated with <a href="https://github.com/jiachangliu/FasterRisk" target="_blank">FasterRisk</a>:

 > https://arxiv.org/pdf/2210.05846.pdf.
 
 
#

<a href="https://github.com/ustunb/risk-slim" target="_blank">risk-slim</a> has an elegant approach mixing Machine Learning and Integer Linear Programming (ILP), that provides the ability to integrate preferences and constraints on the subset of features, and their associated point. It is unfortunately based on CPLEX, a commercial ILP solver that limits its use, and also have trouble converging in large dimensions.

#

<a href="https://github.com/jiachangliu/FasterRisk" target="_blank">FasterRisk</a> is a recent package that makes the computation much faster by dropping the ILP approach and providing an other approach to explore this large space of solutions and generate a list of interesting risk-score models that will be diverse. This approach does not integrate constraints as risk-slim does, but does a great job at quickly computing risk-score models. It only provides a binarizer based on quantile.









