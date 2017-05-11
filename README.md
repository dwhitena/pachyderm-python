## Pachyderm Pipeline for Python-based Classification

This tutorial will walk you through the implementation of training and inference with a scikit-learn model in a Pachyderm pipeline.  Specifically, we will train a scikit-learn model to infer the species of iris flowers based on attributes of iris flowers.

## The pipeline

To deploy and manage the model discussed above, we will implement it’s training, model persistence, and inference in a Pachyderm pipeline.  This will allow us to:

- Keep a rigorous historical record of exactly what models were used on what data to produce which results.
- Automatically update online ML models when training data or parameterization changes.
- Easily revert to other versions of an ML model when a new model is not performing or when “bad data” is introduced into a training data set.

The general structure of our pipeline will look like this:

![Alt text](pipeline.png)

The cylinders represent data “repositories” in which Pachyderm will version training, model, etc. data (think “git for data”).  These data repositories are then input/output of the linked data processing stages (represented by the boxes in the figure).  

## Getting up and running with Pachyderm

You can experiment with this pipeline locally using a quick [local installation of Pachyderm](http://docs.pachyderm.io/en/latest/getting_started/local_installation.html).  Alternatively, you can quickly spin up a real Pachyderm cluster in any one of the popular cloud providers.  Check out the [Pachyderm docs](http://docs.pachyderm.io/en/latest/deployment/deploy_intro.html) for more details on deployment.

Once deployed, you will be able to use the Pachyderm’s `pachctl` CLI tool to create data repositories, create pipelines, and analyze our results.

## Training/fitting the model

First, let’s look our training stage.  The [pytrain.py SVM](pytrain-svm/pytrain.py) and [pytrain.py LDA](pytrain-lda/pytrain.py) Python scripts (and corresponding Docker images) allow us to train an SVM model and an LDA model, respectively, using scikit-learn.  The data that we will use to train the model is freely available (e.g., [here](https://archive.ics.uci.edu/ml/datasets/Iris)) in CSV format.  

The `pytrain.py` scripts take this CSV dataset as input and output representations of the trained/fit models in a Pickle format, `model.pkl`, along with a human readable summary of the model, `model.txt`.

## Inference with the caret model.

The [pyinfer.py](pyinfer/pyinfer.py) script (and corresponding Docker image) allows us to infer iris species based on the persisted Pickle representation of our model (see above).  `pyinfer.py` takes that Pickled model representation as input along with one or more CSV files, each listing particular attributes of flowers (for which we are trying to infer species):

```
5.7,2.8,4.1,1.3
6.3,3.3,6.0,2.5
5.8,2.7,5.1,1.9
7.1,3.0,5.9,2.1
5.1,3.5,1.4,0.2
4.9,3.0,1.4,0.2
```

`pyinfer.py` then outputs an inference based on these attributes:

```
setosa 
virginica
versicolor
virginica
virginica
setosa
```

## Putting it all together, running the pipeline

First let's create Pachyderm "data repositories" in which we will version our training dataset and our attributes (from which we will make predictions):

```
➔ pachctl create-repo training
➔ pachctl create-repo attributes
➔ pachctl list-repo
NAME                CREATED             SIZE                
attributes          1 seconds ago       0 B                 
training            5 seconds ago       0 B                 
➔
```

Next we put our training data set in the `training` repo:

```
➔ cd data/
➔ pachctl put-file training master -c -f iris.csv
➔ pachctl list-repo
NAME                CREATED             SIZE                
training            55 seconds ago      4.444 KiB           
attributes          51 seconds ago      0 B                 
➔ pachctl list-file training master
NAME                TYPE                SIZE                
iris.csv            file                4.444 KiB           
➔
```

We can then create our training and inference pipelines based on JSON specifications ([train.json](train.json) and [infer.json](infer.json)) specifying the Docker images to run for each processing stage, the input to each processing stage, and commands to run in the Docker images.  You could use either the SVM (`dwhitena/pytrain:svm`) or LDA (`dwhitena/pytrain:lda`) models in [train.json](train.json). This will automatically trigger the training of our model and output of the Pickle model representation, because Pachyderm sees that there is training data in `training` that has yet to be processed:

```
➔ cd ..
➔ pachctl create-pipeline -f train.json 
➔ pachctl list-job
ID                                   OUTPUT COMMIT                          STARTED        DURATION   RESTART PROGRESS STATE            
a7c30bdf-0bd1-4e8c-b2b2-d485d7d6ec6e model/d08bda7cf7df45ba8d8989f1461def7c 39 seconds ago 18 seconds 0       1 / 1    success 
➔ pachctl list-repo
NAME                CREATED             SIZE                
model               47 seconds ago      9.626 KiB           
training            3 minutes ago       4.444 KiB           
attributes          3 minutes ago       0 B                 
➔ pachctl list-file model master
NAME                TYPE                SIZE                
model.pkl           file                3.448 KiB           
model.txt           file                226 B
➔ pachctl get-file model master model.txt
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
```

Finally, we can create out inference pipeline stage and commit some attribute files into `attributes` to trigger predictions:

```
➔ pachctl create-pipeline -f infer.json 
➔ cd data/test/
➔ ls
1.csv  2.csv
➔ pachctl put-file attributes master -c -r -f .
➔ pachctl list-job
ID                                   OUTPUT COMMIT                              STARTED        DURATION   RESTART PROGRESS STATE            
eaac63e3-176d-49e0-a1f5-4ce9be173fea inference/e8fae4ab58ba46e5a8014f0b95fc68ba 17 seconds ago 7 seconds  0       2 / 2    success 
a7c30bdf-0bd1-4e8c-b2b2-d485d7d6ec6e model/d08bda7cf7df45ba8d8989f1461def7c     2 minutes ago  18 seconds 0       1 / 1    success 
➔ pachctl list-repo
NAME                CREATED             SIZE                
inference           39 seconds ago      65 B                
attributes          5 minutes ago       112 B               
model               2 minutes ago       9.626 KiB           
training            5 minutes ago       4.444 KiB           
➔ pachctl list-file inference master
NAME                TYPE                SIZE                
1                   file                10 B                
2                   file                55 B                
➔ pachctl get-file inference master 2
Iris-versicolor
Iris-virginica
Iris-virginica
Iris-virginica
Iris-setosa
Iris-setosa
➔
```
