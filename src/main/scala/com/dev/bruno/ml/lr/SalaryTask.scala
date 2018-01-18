package com.dev.bruno.ml.lr

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

import co.theasi.plotly._

object SalaryTask {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("LR.ClosePriceTask")
      .setMaster("local[*]")

    val spark = SparkSession.builder.config(conf).getOrCreate

    val features = Array("YearsExperience")

    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    val dataSet = spark.read.option("header", "true")
      .option("inferSchema", "true").csv("./data/salary.csv")

    val featuredDataSet = assembler.transform(dataSet)

    val Array(trainingSet, testSet) = featuredDataSet.randomSplit(Array(0.66, 0.34), 123)

    val linearRegression = new LinearRegression()
      .setLabelCol("Salary")
      .setFeaturesCol("features")
      .setPredictionCol("predict")

    val model = linearRegression.fit(trainingSet)

    val xTraining = trainingSet.select("YearsExperience").collect.map(_.getDouble(0))
    val yTraining = trainingSet.select("Salary").collect.map(_.getDouble(0))
    val yTrainingPredicted = model.transform(trainingSet).select("predict").collect.map(_.getDouble(0))

    val xTest = testSet.select("YearsExperience").collect.map(_.getDouble(0))
    val yTest = testSet.select("Salary").collect.map(_.getDouble(0))
    val yTestPredicted = model.transform(testSet).select("predict").collect.map(_.getDouble(0))

    val commonAxisOptions = AxisOptions()

    val xAxisOptions = commonAxisOptions.title("Years of Experience")
    val yAxisOptions = commonAxisOptions.title("Salary")

    val p = Plot().withScatter(xTraining, yTraining, ScatterOptions().mode(ScatterMode.Marker).name("Training Set"))
      .withScatter(xTraining, yTrainingPredicted, ScatterOptions().mode(ScatterMode.Line).name("Linear Regression"))
      .withScatter(xTest, yTest, ScatterOptions().mode(ScatterMode.Marker).name("Test Set"))
      .withScatter(xTest, yTestPredicted, ScatterOptions().mode(ScatterMode.Marker).name("Predictions"))
      .xAxisOptions(xAxisOptions)
      .yAxisOptions(yAxisOptions)

    val figure = Figure()
      .plot(p) // add the plot to the figure
      .title("Salary vs Experience")

    draw(figure, "linear-regression", writer.FileOptions(overwrite = true))

    spark.close()
  }
}
