package com.dev.bruno.ml.classification

import java.io.File

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object RandomForest {

  def main(args: Array[String]): Unit = {
    // Dependency to run in standalone mode on windows
    val hadoopFolder = new File("./hadoop").getAbsolutePath
    System.setProperty("hadoop.home.dir", hadoopFolder)

    val conf = new SparkConf()
      .setAppName("RandomForest")
      .setMaster("local[*]")

    val spark = SparkSession.builder.config(conf).getOrCreate

    val dataSet = spark.read.option("header", "true")
      .option("inferSchema", "true").csv("./data/social_network_ads.csv")

    val Array(trainingSet, testSet) = dataSet.randomSplit(Array(0.75, 0.25), 123)

    val assembler = new VectorAssembler()
      .setInputCols(Array("Age", "EstimatedSalary"))
      .setOutputCol("features")

    val labelIndexer = new StringIndexer()
      .setInputCol("Purchased")
      .setOutputCol("label")

    val classifier = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(4000)

    val pipeline = new Pipeline()
      .setStages(Array(assembler, labelIndexer, classifier))

    val model = pipeline.fit(trainingSet)

    val testSetTested = model.transform(testSet)

    testSetTested.show()

    // Compute raw scores on the test set
    val predictionAndLabels = testSetTested.rdd.map(row => (row.getDouble(9), row.getDouble(6)))

    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { label =>
      println(s"Precision($label) = " + metrics.precision(label))
    }

    spark.close()
  }
}
