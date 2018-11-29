// Databricks notebook source
// MAGIC %md # Part II: Using Spark ML with Scala
// MAGIC 
// MAGIC In the last part of the TP, we will perform some learning tasks using the ML library but this time using Scala.

// COMMAND ----------

// MAGIC %md ## 1. Load dataset

// COMMAND ----------

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.util.MLUtils.{loadLibSVMFile, convertVectorColumnsToML}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.mllib.classification.LogisticRegressionModel



// Show available datasets from data-bricks
display(dbutils.fs.ls("/databricks-datasets/samples/data/mllib/"))

// COMMAND ----------

// MAGIC %md We will use the `sample_tree_data` dataset.
// MAGIC 
// MAGIC First we load and parse the data.

// COMMAND ----------

// Load and parse the data file.
val data = sc.textFile("dbfs:/databricks-datasets/samples/data/mllib/sample_tree_data.csv")
val parsedData = data.map { line =>
  val parts = line.split(',').map(_.toDouble)
  (parts(0), Vectors.dense(parts.tail))
}

// COMMAND ----------

// MAGIC %md Next, we need to define the `label` and `features`. 
// MAGIC 
// MAGIC **Note:** `label` and `features` are the default names used in the ML library, you can use other names but then you need to specified them when using a learning algorithm.

// COMMAND ----------

val data_df = sqlContext.createDataFrame(parsedData).toDF("label", "features")
// This is what the data 'looks like' (only first 3 samples shown)
data_df.head(3).foreach(println)

// COMMAND ----------

// MAGIC %md ## 2. Split data into `train` and `test` sets
// MAGIC 
// MAGIC Split the data into training and test sets (30% held out for testing). 
// MAGIC 
// MAGIC Notice that by defining `seed` we ensure that experiments are replicable.

// COMMAND ----------

val splits = data_df.randomSplit(Array(0.7, 0.3), seed=1)
val (trainingData, testData) = (splits(0), splits(1))

// COMMAND ----------

// MAGIC %md ## 3. Train a Decision Tree classifier
// MAGIC 
// MAGIC At first, we will train a 'shallow' tree.
// MAGIC 
// MAGIC Remember that we are using the `ML` implementation.

// COMMAND ----------

val dt = new DecisionTreeClassifier()
             .setMaxDepth(3)
val model = dt.fit(trainingData)

// COMMAND ----------

// MAGIC %md ## 4. Evaluation

// COMMAND ----------

val predictions = model.transform(testData);
val predictions_rdd = predictions.select("label", "prediction").as[(Double, Double)].rdd;
val metrics = new MulticlassMetrics(predictions_rdd);

println("Accuracy:");
println(metrics.accuracy);
println("Confusion Matrix:");
println(metrics.confusionMatrix);

// COMMAND ----------

display(model)

// COMMAND ----------

// MAGIC %md ## 5. Visualize the tree model

// COMMAND ----------

// MAGIC %md ## 6. Experimental Evaluation
// MAGIC 
// MAGIC Now, we will use different parameter values to build the Decision Tree model and check how different this affect the results of the evaluation.
// MAGIC 
// MAGIC ### 1. Change Max Depth.
// MAGIC 
// MAGIC Let's change the parameter `setMaxDepth(n)` where n=5 (default), n=10, n=20. Compute the accuracy for each of these depths. Are the results different?
// MAGIC 
// MAGIC ### 2. Change the Node Impurity measure.
// MAGIC 
// MAGIC Set the `maxDepth` to the default value (5) and compare between using "gini" (default) and "entropy" as impurity measure. What is the best impurity measure (for this dataset)?
// MAGIC 
// MAGIC ### 3. Train a Tree Ensemble (Random Forest) and visualize the first two members of the ensemble.
// MAGIC 
// MAGIC Use `import org.apache.spark.ml.classification.RandomForestClassifier`. Train a RandomForest model using the default parameters and compare it against the DecisionTree model (also using the default parameters). How is performance different?
// MAGIC 
// MAGIC You can use `display(model.trees(n))` to visualize the *n*th tree in the ensemble.
// MAGIC 
// MAGIC ### 4. Change the seed value.
// MAGIC 
// MAGIC The seed value is set by default to some number (`getSeed`). Change that number (`setSeed(n)`). What happens to the model/performance? Explain your answer.
// MAGIC 
// MAGIC ### 5. Change the number of trees in the ensemble.
// MAGIC 
// MAGIC What happens in terms of performance if we change the number of trees? Use `setNumTrees(value)` to set 5, 30 and 100 trees.  
// MAGIC 
// MAGIC ### 6. Use Cross-Validation to find the best Decision Tree model and the best Random Forest model
// MAGIC 
// MAGIC You can use `import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}`, see [example](https://spark.apache.org/docs/latest/ml-tuning.html). However, you are free to imlement the cross validation part on your own.
// MAGIC 
// MAGIC Use the following setup:
// MAGIC - Folds = 3
// MAGIC - Decision Tree hyper-parameter grid:
// MAGIC   - maxDepth: 3, 5, 10
// MAGIC   - impurity: "gini", "entropy"
// MAGIC - Random Forest hyper-parameter grid:
// MAGIC   - maxDepth: 3, 5, 10
// MAGIC   - impurity: "gini", "entropy"
// MAGIC   - numTrees: 10, 30, 100
// MAGIC   
// MAGIC What is the best Decision Tree and the best Random Forest Model? Which one would you choose? Justify your answer.
// MAGIC 
// MAGIC ### 7. Use Cross-Validation to find the best Decision Tree model and the best Random Forest model for *"sample_binary_classification_data"*
// MAGIC 
// MAGIC - File location: "dbfs:/databricks-datasets/samples/data/mllib/sample_binary_classification_data.txt"
// MAGIC - You have to change the code to load the dataset and to create the corresponding DataFrame. Hint:`import org.apache.spark.mllib.util.MLUtils.{loadLibSVMFile, convertVectorColumnsToML}`
// MAGIC - Use the same setup as in the previous step.
// MAGIC 
// MAGIC What is the best Decision Tree and the best Random Forest Model? Which one would you choose? Justify your answer.

// COMMAND ----------

// MAGIC %md ### 1. Change Max Depth.
// MAGIC 
// MAGIC Let's change the parameter `setMaxDepth(n)` where n=5 (default), n=10, n=20. Compute the accuracy for each of these depths. Are the results different?

// COMMAND ----------


def test_max_depth(n:Int) ={
  val dt = new DecisionTreeClassifier()
               .setMaxDepth(n)
  val model = dt.fit(trainingData)
  val predictions = model.transform(testData);
  val predictions_rdd = predictions.select("label", "prediction").as[(Double, Double)].rdd;
  val metrics = new MulticlassMetrics(predictions_rdd);

  println("Accuracy:");
  println(metrics.accuracy);
  println("Confusion Matrix:");
  println(metrics.confusionMatrix);
}

// COMMAND ----------

for(i <- List.range(5, 20, 5))test_max_depth(i)

// COMMAND ----------

// MAGIC %md The accuracy decrease a bit for the values n = 10 or 20 compared to the n = 5 value.

// COMMAND ----------

// MAGIC %md ### 2. Change the Node Impurity measure.
// MAGIC 
// MAGIC Set the `maxDepth` to the default value (5) and compare between using "gini" (default) and "entropy" as impurity measure. What is the best impurity measure (for this dataset)?

// COMMAND ----------

def test_max_depth(measure:String) ={
  val impurity = measure;
  val dt = new DecisionTreeClassifier()
               .setMaxDepth(5)
               .setImpurity(impurity)
  val model = dt.fit(trainingData)
  val predictions = model.transform(testData);
  val predictions_rdd = predictions.select("label", "prediction").as[(Double, Double)].rdd;
  val metrics = new MulticlassMetrics(predictions_rdd);

  println("Accuracy:");
  println(metrics.accuracy);
  println("Confusion Matrix:");
  println(metrics.confusionMatrix);
}

// COMMAND ----------

test_max_depth("entropy")
test_max_depth("gini")

// COMMAND ----------

// MAGIC %md The gini impurity function is better than the entropy one (0.97 vs 0.94 in the accuracy results). 

// COMMAND ----------

// MAGIC %md ### 3. Train a Tree Ensemble (Random Forest) and visualize the first two members of the ensemble.
// MAGIC 
// MAGIC Use `import org.apache.spark.ml.classification.RandomForestClassifier`. Train a RandomForest model using the default parameters and compare it against the DecisionTree model (also using the default parameters). How is performance different?
// MAGIC 
// MAGIC You can use `display(model.trees(n))` to visualize the *n*th tree in the ensemble.

// COMMAND ----------

val dt = new RandomForestClassifier()
             .setMaxDepth(5)
val model = dt.fit(trainingData)
val predictions = model.transform(testData);
val predictions_rdd = predictions.select("label", "prediction").as[(Double, Double)].rdd;
val metrics = new MulticlassMetrics(predictions_rdd);
  
println("Accuracy:");
println(metrics.accuracy);
println("Confusion Matrix:");
println(metrics.confusionMatrix);

// COMMAND ----------

display(model.trees(2))

// COMMAND ----------

// MAGIC %md ### 4. Change the seed value.
// MAGIC 
// MAGIC The seed value is set by default to some number (`getSeed`). Change that number (`setSeed(n)`). What happens to the model/performance? Explain your answer.

// COMMAND ----------

def change_seed_value(n:Int)={
  val dt = new RandomForestClassifier()
             .setMaxDepth(5)
            .setSeed(n)
val model = dt.fit(trainingData)
}

// COMMAND ----------

change_seed_value(1)

// COMMAND ----------

change_seed_value(2)

// COMMAND ----------

change_seed_value(3)

// COMMAND ----------

def change_seed_value_accuracy(n:Int)={
   val dt = new RandomForestClassifier()
               .setMaxDepth(5)
              .setSeed(n)
  val model = dt.fit(trainingData)
  val predictions = model.transform(testData);
  val predictions_rdd = predictions.select("label", "prediction").as[(Double, Double)].rdd;
  val metrics = new MulticlassMetrics(predictions_rdd);

  println("Accuracy:");
  println(metrics.accuracy);
}

// COMMAND ----------

for (i <- List.range(1,10))change_seed_value_accuracy(i)

// COMMAND ----------

// MAGIC %md From the results from above, we can see that the seed parameter have a great impact on the Accuracy and the model form, 
// MAGIC the accuracy range from 0.94 to 0.96 and with different seed the root in the decision tree change (from feature 23 to feature 22). 

// COMMAND ----------

// MAGIC %md ### 5. Change the number of trees in the ensemble.
// MAGIC 
// MAGIC What happens in terms of performance if we change the number of trees? Use `setNumTrees(value)` to set 5, 30 and 100 trees. 

// COMMAND ----------

def change_num_trees(n:Int)={
  val dt = new RandomForestClassifier()
             .setMaxDepth(5)
            .setNumTrees(n)
  val model = dt.fit(trainingData)
  val predictions = model.transform(testData);
  val predictions_rdd = predictions.select("label", "prediction").as[(Double, Double)].rdd;
  val metrics = new MulticlassMetrics(predictions_rdd);

  println("Accuracy:");
  println(metrics.accuracy);
}

// COMMAND ----------

change_num_trees(5)
change_num_trees(30)
change_num_trees(100)

// COMMAND ----------

// MAGIC %md The accuracy results doas not seem to be affected if we change the number of trees 

// COMMAND ----------

change_num_trees(5)

// COMMAND ----------

change_num_trees(30)

// COMMAND ----------

change_num_trees(100)

// COMMAND ----------

// MAGIC %md However we can see that by setting the trees to 5 we have a computation of 1.84 seconds, 30 is 2.52 seconds and 100 is 2.55 seconds. For this cas it does not affect the accuracy but it does affect the computing time.

// COMMAND ----------

// MAGIC %md ### 6. Use Cross-Validation to find the best Decision Tree model and the best Random Forest model
// MAGIC 
// MAGIC You can use `import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}`, see [example](https://spark.apache.org/docs/latest/ml-tuning.html). However, you are free to imlement the cross validation part on your own.
// MAGIC 
// MAGIC Use the following setup:
// MAGIC - Folds = 3
// MAGIC - Decision Tree hyper-parameter grid:
// MAGIC   - maxDepth: 3, 5, 10
// MAGIC   - impurity: "gini", "entropy"
// MAGIC - Random Forest hyper-parameter grid:
// MAGIC   - maxDepth: 3, 5, 10
// MAGIC   - impurity: "gini", "entropy"
// MAGIC   - numTrees: 10, 30, 100
// MAGIC   
// MAGIC What is the best Decision Tree and the best Random Forest Model? Which one would you choose? Justify your answer.

// COMMAND ----------


val rf = new RandomForestClassifier()
val pipeline = new Pipeline()
  .setStages(Array(rf))

val paramGrid = new ParamGridBuilder()
  .addGrid(rf.maxDepth, Array(3, 5, 10))
  .addGrid(rf.impurity, Array("gini","entropy"))
  .addGrid(rf.numTrees,Array(10,30,100))
  .build()
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new RegressionEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)  // Use 3+ in practice
  .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

// Run cross-validation, and choose the best set of parameters.
val cvModel = cv.fit(trainingData)
val predictions = cvModel.transform(testData);
val predictions_rdd = predictions.select("label", "prediction").as[(Double, Double)].rdd;
val metrics = new MulticlassMetrics(predictions_rdd);
println("Accuracy of the RandomForest:");
println(metrics.accuracy);
val bestPipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
val stages = bestPipelineModel.stages
val lrStage = stages(0)
println("RandomForest params = " + lrStage.extractParamMap()) 

// COMMAND ----------

val dt = new DecisionTreeClassifier()
val pipeline = new Pipeline()
  .setStages(Array(dt))
// We use a ParamGridBuilder to construct a grid of parameters to search over.
// With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
// this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
val paramGrid = new ParamGridBuilder()
  .addGrid(rf.maxDepth, Array(3, 5, 10))
  .addGrid(rf.impurity, Array("gini","entropy"))
  .build()
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new RegressionEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)  // Use 3+ in practice
  .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

// Run cross-validation, and choose the best set of parameters.
val cvModel = cv.fit(trainingData)
val predictions = cvModel.transform(testData);
val predictions_rdd = predictions.select("label", "prediction").as[(Double, Double)].rdd;
val metrics = new MulticlassMetrics(predictions_rdd);
println("Accuracy of the DecisionTree:");
println(metrics.accuracy);
val bestPipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
val stages = bestPipelineModel.stages
val lrStage = stages(0)
println("DecisionTree params = " + lrStage.extractParamMap())


// COMMAND ----------

// MAGIC %md 
// MAGIC The best accuracy found after the cross validation to the random forest classifier is 0.955 and 0.97 for the decision tree classifier.
// MAGIC 
// MAGIC The best parameters for the randomForest: gini, numTrees = 100, maxDepth = 10.
// MAGIC 
// MAGIC The best parameters for the DecisionTree: gini, maxDepth = 5.
// MAGIC 
// MAGIC I would choose the decision tree because it gives the best accuracy and have a lower depth (5 vs 10)

// COMMAND ----------

// MAGIC %md
// MAGIC ### 7. Use Cross-Validation to find the best Decision Tree model and the best Random Forest model for *"sample_binary_classification_data"*
// MAGIC 
// MAGIC - File location: "dbfs:/databricks-datasets/samples/data/mllib/sample_binary_classification_data.txt"
// MAGIC - You have to change the code to load the dataset and to create the corresponding DataFrame. Hint:`import org.apache.spark.mllib.util.MLUtils.{loadLibSVMFile, convertVectorColumnsToML}`
// MAGIC - Use the same setup as in the previous step.
// MAGIC 
// MAGIC What is the best Decision Tree and the best Random Forest Model? Which one would you choose? Justify your answer.

// COMMAND ----------

// Load and parse the data file.
val data2 = loadLibSVMFile(sc, "dbfs:/databricks-datasets/samples/data/mllib/sample_binary_classification_data.txt")
val cols = Set("label", "features")
val data_df2 = convertVectorColumnsToML(data2.toDF())//,cols );
val splits = data_df2.randomSplit(Array(0.7, 0.3), seed=1)
val (trainingData2, testData2) = (splits(0), splits(1))

// COMMAND ----------

val dt = new DecisionTreeClassifier()
val pipeline = new Pipeline()
  .setStages(Array(dt))
// We use a ParamGridBuilder to construct a grid of parameters to search over.
// With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
// this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
val paramGrid = new ParamGridBuilder()
  .addGrid(rf.maxDepth, Array(3, 5, 10))
  .addGrid(rf.impurity, Array("gini","entropy"))
  .build()
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)  // Use 3+ in practice
  .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

// Run cross-validation, and choose the best set of parameters.
val cvModel = cv.fit(trainingData2)
val predictions = cvModel.transform(testData2);
val predictions_rdd = predictions.select("label", "prediction").as[(Double, Double)].rdd;
val metrics = new MulticlassMetrics(predictions_rdd);
println("Accuracy of the DecisionTree:");
println(metrics.accuracy);
val bestPipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
val stages = bestPipelineModel.stages
val lrStage = stages(0)
println("DecisionTree params = " + lrStage.extractParamMap())

// COMMAND ----------

val rf = new RandomForestClassifier()
val pipeline = new Pipeline()
  .setStages(Array(rf))
// We use a ParamGridBuilder to construct a grid of parameters to search over.
// With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
// this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
val paramGrid = new ParamGridBuilder()
  .addGrid(rf.maxDepth, Array(3, 5, 10))
  .addGrid(rf.impurity, Array("gini","entropy"))
  .addGrid(rf.numTrees,Array(10,30,100))
  .build()
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)  // Use 3+ in practice
  .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

// Run cross-validation, and choose the best set of parameters.
val cvModel = cv.fit(trainingData2)
val predictions = cvModel.transform(testData2);
val predictions_rdd = predictions.select("label", "prediction").as[(Double, Double)].rdd;
val metrics = new MulticlassMetrics(predictions_rdd);
println("Accuracy of the RandomForest:");
println(metrics.accuracy);
val bestPipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
val stages = bestPipelineModel.stages
val lrStage = stages(0)
println("RandomForest params = " + lrStage.extractParamMap())

// COMMAND ----------

// MAGIC %md
// MAGIC The best Decision Tree and the best Random Forest Model are :
// MAGIC 
// MAGIC Decision Tree : gini, maxDepth = 5
// MAGIC 
// MAGIC RandomForest : gini, maxDepth = 3, numTrees = 10 
// MAGIC 
// MAGIC The best accuracy found after the cross validation to the random forest classifier is 1 and 0.939 for the decision tree classifier.
// MAGIC 
// MAGIC 
// MAGIC I would choose the random forest classifier because it gives the best performances and have a lower depth.
