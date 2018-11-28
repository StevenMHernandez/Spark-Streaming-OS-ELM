# ELM and OS-ELM for Apache Spark MLlib (and Spark Streaming)

The goal of this project is to develop both an 
[Extreme Learning Machine (ELM)](https://ieeexplore.ieee.org/document/1380068) and 
[Online Sequential Extreme Learning Machine (OS-ELM)](https://ieeexplore.ieee.org/document/4012031) 
MLlib implementation for Apache Spark. Specific application of these algorithms is for use in 
Fingerprint-based Indoor Localization which uses ambient radio signal strengths such as the 
strength of any nearby Wi-Fi routers to predict the location or (x,y) coordinates of a device as 
can be see from [Zou et al.](https://www.ncbi.nlm.nih.gov/pubmed/25599427) While this use-case is 
the primary focus for this work, the ELM and OS-ELM implementations are generic enough for any use 
case where a Spark MLlib algorithm would be used.

## Installation and Running Simulations

This project requires [spark](https://spark.apache.org/downloads.html), and requires [sbt](https://www.scala-sbt.org/1.x/docs/Setup.html) for building the project.
Development was done in Intellij which offers a plugin [Scala](https://confluence.jetbrains.com/display/SCA/Scala+Plugin+for+IntelliJ+IDEA) which supports both scala and sbt, although this is not required.

With `sbt` installed, we can run sbt from the terminal.

```
$ sbt
```

To compile the classes:

```
$ compile
```

Finally, to run the project call:

```
$ run
```

From this, a dialog will appear which allows you to select which main file to run. Note, running 
these `Main` classes does require supporting data files which can be created through the MATLAB 
scripts found in the `./matlab` directory.

## Usage (Apache Spark)

The following files contain full example usages of the learning algorithms for both standard Spark 
cases as well as Spark Streaming cases: 

`./src/main/scala/Main` for standard Spark

`./src/main/scala/StreamingMain` for a Streaming based version which listens for changes in two 
directories, a training and testing directory.

`./src/main/scala/MainSimulationOne` runs a simulation by taking in 6 files: 

1. `src/main/resources/simulationOne/train_XY_1.csv`
2. `src/main/resources/simulationOne/train_XY_2.csv`
3. `src/main/resources/simulationOne/train_XY_3.csv`
4. `src/main/resources/simulationOne/test_XY_1.csv`
5. `src/main/resources/simulationOne/test_XY_2.csv`
6. `src/main/resources/simulationOne/test_XY_3.csv`

And running one instance of training, then one instance of testing repeated three times.

`./src/main/scala/MainSimulationOne` runs a simulation by taking in 1 file containing a large number of data samples: 

1. `src/main/resources/simulationFinal/train_XY_1.csv`

And running training and testing on different sizes of data (using the `.limit()` method) to track how long training and testing takes for different sizes of datasets.

### ELM and OSELM usage

Given we have Dataframes (`trainingDF`, `testingDF`) with columns `input` and `output`, we begin by 
creating the ELM estimator.

```
val elm = new ELM()
  .setHiddenNodes(5)
  .setActivationFunction(ELM.SIGMOID)
``` 

Or alternatively, OS-ELM

```
val oselm = new OSELM()
  .setHiddenNodes(5)
  .setActivationFunction(ELM.SIGMOID)
``` 

Activation Functions can be one of the following functions: `ELM.SIGMOID`, `ELM.SIGNUM`, `ELM.TANH`.

The training and predicting is the same for either learning algorithm. Train begins with:

```
elm.fit(trainingDF)
```

For the OS-ELM, this method can be called multiple times in succession. By doing this, we use the 
previous knowledge from previous calls to `fit()` along with the new training data. On the other
hand, the ELM model would learn from scratch if `fit()` is called multiple times, forgetting the 
previously learned internal variables.

After the learning phase, we go on to the prediction phase:

```
elm.transform(testingDF)
```

## Usage (Spark Streaming)

Streaming is a slightly different beast. Previous versions of Spark MLlib have algorithms designed 
specifically for streaming such as `StreamingKMeans` and `StreamingLinearRegressionWithSGD`. These 
classes, along with the streaming class developed for this project `StreamingOSELM` follow a 
slightly different method of usage. 

Streaming does not use Dataframes directly, instead relying on Dstreams. In this streaming context, 
we need two streams `trainingDataStream` and `testDataStream` where the streams contain a tuple of 
Doubles where the first item in the tuple is the input and the second is the expected output.

We begin similarly by creating the learning algorithm object

```
val model = new StreamingOSELM()
  .setHiddenNodes(5)
  .setActivationFunction(ELM.SIGMOID)
``` 

From here, the system can begin listening and reacting to updates to the stream as follows: 

```
model.trainOn(trainingDataStream)
model.predictOn(testDataStream).print()
```
