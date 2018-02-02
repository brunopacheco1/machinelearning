package com.dev.bruno.ml.rnn

import java.io.File
import java.util

import co.theasi.plotly.{AxisOptions, Figure, Plot, ScatterMode, ScatterOptions, draw, writer}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{Row, SparkSession}
import org.deeplearning4j.spark.stats.StatsUtils
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ListBuffer

object NextClosePriceTask {

  def main(args: Array[String]): Unit = {
    val features = Array("Open", "Close", "Low", "High")

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

    val list = createTimeFrameList(timeFrameSize, dataSet.collectAsList())

    val splitAt = list.size - 7

    val (trainingList, testList) = list.splitAt(splitAt)

    val trainingSet = spark.sparkContext.parallelize(createTrainingSet(features, labels, miniBatchSize, timeFrameSize, min, max, trainingList))

    val trainingTestSet = createTestSet(features, labels, timeFrameSize, min, max, trainingList)

    val testSet = createTestSet(features, labels, timeFrameSize, min, max, testList)

    val sparkNetwork = RNNBuilder.build(features.length, labels.length, timeFrameSize, miniBatchSize, sc)

    for (_ <- 1 to epochs) {
      sparkNetwork.fit(trainingSet)
    }

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

    val net = sparkNetwork.getNetwork
    val locationToSave = new File("./lstm_model_" + epochs + "_" + miniBatchSize + "_" + timeFrameSize + ".zip")
    //saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
    ModelSerializer.writeModel(net, locationToSave, true)
    spark.close()
  }

  def createTrainingSet(features: Array[String], labels: Array[String], miniBatchSize: Int, timeFrameSize: Int, min: Row, max: Row, timeFrames: ListBuffer[ListBuffer[Row]]): ListBuffer[DataSet] = {
    var start = 0
    var currentBatchSize = Math.min(miniBatchSize, timeFrames.size)
    var result = new ListBuffer[DataSet]()

    do {
      val input = Nd4j.create(Array[Int](currentBatchSize, features.length, timeFrameSize), 'f')
      val label = Nd4j.create(Array[Int](currentBatchSize, labels.length, timeFrameSize), 'f')

      for (batchIndex <- 0 until currentBatchSize) {
        val rows = timeFrames(start)

        for (timeFrameIndex <- rows.indices) {
          val price = rows(timeFrameIndex)

          for(featureIndex <- features.indices) {
            val value = (price.getDouble(featureIndex) - min.getDouble(featureIndex)) / (max.getDouble(featureIndex) - min.getDouble(featureIndex))

            input.putScalar(Array[Int](batchIndex, featureIndex, timeFrameIndex), value)
          }

          for(labelIndex <- labels.indices) {
            val priceRowIndex = features.length + labelIndex

            val value = (price.getDouble(priceRowIndex) - min.getDouble(priceRowIndex)) / (max.getDouble(priceRowIndex) - min.getDouble(priceRowIndex))

            label.putScalar(Array[Int](batchIndex, labelIndex, timeFrameIndex), value)
          }
        }

        start += 1
      }

      result += new DataSet(input, label)

      currentBatchSize = Math.min(miniBatchSize, timeFrames.size - start)

    } while (start < timeFrames.size)

    result
  }

  def createTestSet(features: Array[String], labels: Array[String], timeFrameSize: Int, min: Row, max: Row, timeFrames: ListBuffer[ListBuffer[Row]]): ListBuffer[DataSet] = {
    var start = 0
    var result = new ListBuffer[DataSet]()

    do {
      val input = Nd4j.create(Array[Int](timeFrameSize, features.length), 'f')
      val label = Nd4j.create(Array[Int](timeFrameSize, labels.length), 'f')

      val rows = timeFrames(start)

      for (timeFrameIndex <- rows.indices) {
        val price = rows(timeFrameIndex)

        for(featureIndex <- features.indices) {
          val value = (price.getDouble(featureIndex) - min.getDouble(featureIndex)) / (max.getDouble(featureIndex) - min.getDouble(featureIndex))

          input.putScalar(Array[Int](timeFrameIndex, featureIndex), value)
        }

        for(labelIndex <- labels.indices) {
          val priceRowIndex = features.length + labelIndex

          val value = (price.getDouble(priceRowIndex) - min.getDouble(priceRowIndex)) / (max.getDouble(priceRowIndex) - min.getDouble(priceRowIndex))

          label.putScalar(Array[Int](timeFrameIndex, labelIndex), value)
        }
      }

      start += 1

      result += new DataSet(input, label)

    } while (start < timeFrames.size)

    result
  }

  def createTimeFrameList(timeFrameSize: Int, rows: util.List[Row]): ListBuffer[ListBuffer[Row]] = {
    var start = 0
    var end = Math.min(timeFrameSize, rows.size)
    var result = new ListBuffer[ListBuffer[Row]]()

    do {
      var timeFrame = new ListBuffer[Row]()

      for (j <- 0 until timeFrameSize) {
        timeFrame += rows.get(start + j)
      }

      start += 1
      end += 1

      result += timeFrame

    } while (end < rows.size)

    result
  }
}