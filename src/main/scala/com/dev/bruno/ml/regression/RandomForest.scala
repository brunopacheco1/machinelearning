package com.dev.bruno.ml.regression

import java.io.File

import co.theasi.plotly.{AxisOptions, Figure, Plot, ScatterMode, ScatterOptions, draw, writer}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{DecisionTreeRegressor, RandomForestRegressor}
import org.apache.spark.sql.SparkSession

object RandomForest {

  def main(args: Array[String]): Unit = {
    // Dependency to run in standalone mode on windows
    val hadoopFolder = new File("./hadoop").getAbsolutePath
    System.setProperty("hadoop.home.dir", hadoopFolder)

    val conf = new SparkConf()
      .setAppName("RandomForestRegression")
      .setMaster("local[*]")

    val spark = SparkSession.builder.config(conf).getOrCreate

    val trainingSet = spark.read.option("header", "true")
      .option("inferSchema", "true").csv("./data/position_salaries_training.csv").sort("Level")

    val testSet = spark.read.option("header", "true")
      .option("inferSchema", "true").csv("./data/position_salaries_test.csv").sort("Level")

    val trainingGridSet = spark.read.option("header", "true")
      .option("inferSchema", "true").csv("./data/position_salaries_grid.csv").sort("Level")

    val assembler = new VectorAssembler()
      .setInputCols(Array("Level"))
      .setOutputCol("features")

    val regressor = new RandomForestRegressor()
      .setLabelCol("Salary")
      .setNumTrees(4000)
      .setFeaturesCol("features")
      .setPredictionCol("predict")

    val pipeline = new Pipeline()
      .setStages(Array(assembler, regressor))

    val model = pipeline.fit(trainingSet)

    val xTraining = trainingSet.select("Level").collect.map(_.getDouble(0))
    val yTraining = trainingSet.select("Salary").collect.map(_.getDouble(0))
    val xGrid = trainingGridSet.select("Level").collect.map(_.getDouble(0))
    val yGridPredicted = model.transform(trainingGridSet).select("predict").collect.map(_.getDouble(0))

    val xTest = testSet.select("Level").collect.map(_.getDouble(0))
    val yTestPredicted = model.transform(testSet).select("predict").collect.map(_.getDouble(0))

    val commonAxisOptions = AxisOptions()

    val xAxisOptions = commonAxisOptions.title("Level")
    val yAxisOptions = commonAxisOptions.title("Salary")

    val p = Plot().withScatter(xTraining, yTraining, ScatterOptions().mode(ScatterMode.Marker).name("Training Set"))
      .withScatter(xGrid, yGridPredicted, ScatterOptions().mode(ScatterMode.Line).name("Decision Tree Regression"))
      .withScatter(xTest, yTestPredicted, ScatterOptions().mode(ScatterMode.Marker).name("Prediction"))
      .xAxisOptions(xAxisOptions)
      .yAxisOptions(yAxisOptions)

    val figure = Figure()
      .plot(p) // add the plot to the figure
      .title("Level vs Salary")

    draw(figure, "random-forest-regression", writer.FileOptions(overwrite = true))

    spark.close()
  }
}
