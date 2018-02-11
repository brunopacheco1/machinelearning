package com.dev.bruno.ml.regression

import java.io.File

import co.theasi.plotly._
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object MultipleLinearRegression {

  def main(args: Array[String]): Unit = {
    // Dependency to run in standalone mode on windows
    val hadoopFolder = new File("./hadoop").getAbsolutePath
    System.setProperty("hadoop.home.dir", hadoopFolder)

    val conf = new SparkConf()
      .setAppName("MultipleLinearRegression")
      .setMaster("local[*]")

    val spark = SparkSession.builder.config(conf).getOrCreate

    val features = Array("R&D Spend")

    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    val dataSet = spark.read.option("header", "true")
      .option("inferSchema", "true").csv("./data/startups.csv")

    val featuredDataSet = assembler.transform(dataSet)

    val Array(trainingSet, testSet) = featuredDataSet.randomSplit(Array(0.8, 0.2), 123)

    val linearRegression = new LinearRegression()
      .setLabelCol("Profit")
      .setFeaturesCol("features")
      .setPredictionCol("predict")

    val model = linearRegression.fit(trainingSet)

    val xTraining = trainingSet.select("R&D Spend").collect.map(_.getDouble(0))
    val yTraining = trainingSet.select("Profit").collect.map(_.getDouble(0))
    val yTrainingPredicted = model.transform(trainingSet).select("predict").collect.map(_.getDouble(0))

    val xTest = testSet.select("R&D Spend").collect.map(_.getDouble(0))
    val yTest = testSet.select("Profit").collect.map(_.getDouble(0))
    val yTestPredicted = model.transform(testSet).select("predict").collect.map(_.getDouble(0))

    val commonAxisOptions = AxisOptions()

    val xAxisOptions = commonAxisOptions.title("R&D Spend")
    val yAxisOptions = commonAxisOptions.title("Profit")

    val p = Plot().withScatter(xTraining, yTraining, ScatterOptions().mode(ScatterMode.Marker).name("Training Set"))
      .withScatter(xTraining, yTrainingPredicted, ScatterOptions().mode(ScatterMode.Line).name("Linear Regression"))
      .withScatter(xTest, yTest, ScatterOptions().mode(ScatterMode.Marker).name("Test Set"))
      .withScatter(xTest, yTestPredicted, ScatterOptions().mode(ScatterMode.Marker).name("Predictions"))
      .xAxisOptions(xAxisOptions)
      .yAxisOptions(yAxisOptions)

    val figure = Figure()
      .plot(p) // add the plot to the figure
      .title("R&D Spend vs Profit")

    draw(figure, "multiple-linear-regression", writer.FileOptions(overwrite = true))

    spark.close()
  }
}
