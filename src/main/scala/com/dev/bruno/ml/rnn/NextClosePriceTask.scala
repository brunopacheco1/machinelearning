package com.dev.bruno.ml.rnn

import java.io.File
import co.theasi.plotly.{AxisOptions, Figure, Plot, ScatterMode, ScatterOptions, draw, writer}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.deeplearning4j.spark.stats.StatsUtils
import org.deeplearning4j.util.ModelSerializer

import scala.collection.mutable.ListBuffer

object NextClosePriceTask {

  def main(args: Array[String]): Unit = {
    val features = Array("Open", "Close")

    val labels = Array("NextClose")

    val epochs = 1000

    val miniBatchSize = 512

    val timeFrameSize = 50

    // Dependency to run in standalone mode on windows
    val hadoopFolder = new File("./hadoop").getAbsolutePath
    System.setProperty("hadoop.home.dir", hadoopFolder)

    // Basic configuration
    val conf = new SparkConf()
      .setAppName("RNN.NextClosePriceTask")
      .setMaster("local[*]")

    // Initialization of Spark And Spark SQL Context
    val spark = SparkSession.builder.config(conf).getOrCreate

    val sc = spark.sparkContext

    val cols = features ++ labels

    spark.read.parquet("goog.parquet").createOrReplaceTempView("google_stocks")

    val dataSet = spark.sql("select " + cols.mkString(", ") + " from google_stocks order by Date")

    val min = spark.sql("select min(" + cols.mkString("), min(") + ") from google_stocks").head()

    val max = spark.sql("select max(" + cols.mkString("), max(") + ") from google_stocks").head()

    val splitRatio = 0.95D

    val (trainingSet, trainingTestSet, testSet) = DataSetBuilder.build(features, labels, miniBatchSize, timeFrameSize, min, max, dataSet.collectAsList(), splitRatio)

    val sparkNetwork = RNNBuilder.build(features.length, labels.length, timeFrameSize, miniBatchSize, sc)

    for (_ <- 1 to epochs) {
      sparkNetwork.fit(spark.sparkContext.parallelize(trainingSet))
    }

    val net = sparkNetwork.getNetwork
    val locationToSave = new File("./lstm_model_" + epochs + "_" + miniBatchSize + "_" + timeFrameSize + ".zip")
    //saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
    ModelSerializer.writeModel(net, locationToSave, true)

    val stats = sparkNetwork.getSparkTrainingStats
    StatsUtils.exportStatsAsHtml(stats, "SparkStats.html", sc)

    var trainingX = new ListBuffer[Double]()
    var trainingY = new ListBuffer[Double]()
    var trainingYPredicted = new ListBuffer[Double]()
    var testX = new ListBuffer[Double]()
    var testY = new ListBuffer[Double]()
    var testYPredicted = new ListBuffer[Double]()

    var counter = 1D

    trainingTestSet.foreach(set => {
      val labelIndex = features.length + labels.length - 1
      val actual = set.getLabels.getDouble(timeFrameSize - 1) * (max.getDouble(labelIndex) - min.getDouble(labelIndex)) + min.getDouble(labelIndex)
      val prediction = sparkNetwork.getNetwork.rnnTimeStep(set.getFeatures).getDouble(timeFrameSize - 1) * (max.getDouble(labelIndex) - min.getDouble(labelIndex)) + min.getDouble(labelIndex)

      trainingX += counter

      trainingY += actual

      trainingYPredicted += prediction

      counter += 1D
    })

    testSet.foreach(set => {
      val labelIndex = features.length + labels.length - 1
      val actual = set.getLabels.getDouble(timeFrameSize - 1) * (max.getDouble(labelIndex) - min.getDouble(labelIndex)) + min.getDouble(labelIndex)
      val prediction = sparkNetwork.getNetwork.rnnTimeStep(set.getFeatures).getDouble(timeFrameSize - 1) * (max.getDouble(labelIndex) - min.getDouble(labelIndex)) + min.getDouble(labelIndex)

      testX += counter

      testY += actual

      testYPredicted += prediction

      counter += 1D
    })

    val commonAxisOptions = AxisOptions()

    val xAxisOptions = commonAxisOptions.title("Time Series")
    val yAxisOptions = commonAxisOptions.title("Close Price")

    val p = Plot()
      .withScatter(trainingX, trainingY, ScatterOptions().mode(ScatterMode.Line).name("Training Set"))
      .withScatter(trainingX, trainingYPredicted, ScatterOptions().mode(ScatterMode.Line).name("Training Predictions"))
      .withScatter(testX, testY, ScatterOptions().mode(ScatterMode.Line).name("Test Set"))
      .withScatter(testX, testYPredicted, ScatterOptions().mode(ScatterMode.Line).name("Test Predictions"))
      .xAxisOptions(xAxisOptions)
      .yAxisOptions(yAxisOptions)

    val figure = Figure()
      .plot(p)
      .title("Stock Market Prediction")

    draw(figure, "stock-prediction-" + epochs + "-" + miniBatchSize + "-" + timeFrameSize, writer.FileOptions(overwrite = true))

    spark.close()
  }
}