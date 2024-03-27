[![PyPI](https://img.shields.io/pypi/v/tsproto)](https://pypi.org/project/tsproto/)  ![License](https://img.shields.io/github/license/sbobek/tsproto)
 ![PyPI - Downloads](https://img.shields.io/pypi/dm/tsproto) [![Documentation Status](https://readthedocs.org/projects/tsproto/badge/?version=latest)](https://tsproto.readthedocs.io/en/latest/?badge=latest)
    
![](https://raw.githubusercontent.com/sbobek/tsproto/main/pix/workflow.svg)
# TSProto
Post-host prototype-based explanations with rules for time-series classifiers.

Key features:
  * Extracts interpretable prototype for any black-box model and creates a decision tree, where each node is constructed from the visual prototype
  * Integrated with SHAP explainer, as a backbone for extraction of interpretable components (However, SHAP can be replaced with any other feature-importance method)

## Install
TSProto can be installed from either [PyPI](https://pypi.org/project/tsproto/) or directly from source code from this repository.

To install form PyPI:

```
pip install tsproto
````

To install from source code:

```
git clone https://github.com/sbobek/tsproto
cd tsproto
pip install .
```

## Usage
For full examples on two illustrative cases go to:
  * Example of extracting sine wave prototype and explaining class with existence ora absence of a prototype: [Jupyter Notebook](https://github.com/sbobek/tsproto/blob/main/examples/illustrative-example-frequency.ipynb)
  * Example of extracting sine wave as a prototype end explaining class by difference in frequency of a prototype [Jupyter Notebook](https://github.com/sbobek/tsproto/blob/main/examples/illustrative-example.ipynb)

The basic usage of the TSProto assuming you have your model trained is straightforward:

``` python
from tsproto.models import *
from tsproto.utils import *

#assuming that trainX, trainy and model are given

pe = PrototypeEncoder(clf, n_clusters=2, min_size=50, method='dtw',
                      descriptors=['existance'],
                      jump=1, pen=1,multiplier=2,n_jobs=-1,
                      verbose=1)

trainX, shapclass = getshap(model=model, X=trainX, y=trainy,shap_version='deep',
                        bg_size = 1000,  absshap = True)               
                        
#The input needs to be a 3D vector: number of samples, lenght of time-series, number of dimensions (features)                        
trainXproto = train.reshape((trainX.shape[0], trainX.shape[1],1))
shapclassXproto = shapclass.reshape((shapclass.shape[0], shapclass.shape[1],1))
       
ohe_train, features, target_ohe,weights = pe.fit_transform(trainXproto,shapclassXproto)

im  = InterpretableModel()
acc,prec,rec,f1,interpretable_model = im.fit_or_predict(ohe_train, features, 
                        target_ohe,
                        intclf=None, # if intclf is given, the funciton behaves as predict, 
                        verbose=0, max_depth=2, min_samples_leaf=0.05,
                        weights=None)
                 
```

After the Interpretable model has been created it now can be visualised.

``` python
                       
# Visualize model
from  tsproto.plots import *

ds_final = ohe_train.copy()
dot = export_decision_tree_with_embedded_histograms(decision_tree=interpretable_model, 
                                              dataset=ds_final, 
                                              target_name='target', 
                                              feature_names=features, 
                                              filename='synthetic', 
                                              proto_encoder=pe, figsize=(6,3))

from IPython.display import SVG, Image
Image('synthetic.png')

```

![Prototype visualization](https://raw.githubusercontent.com/sbobek/tsproto/main/pix/illustrative-example.png "Title")


## Cite this work
More details on how the TSProto works and evaluation benchmarks can eb found in the following paper:

```Comming soon```
