package com.dev.bruno.ml.rnn

import java.io.File
import java.util

import org.apache.spark.SparkConf
import org.apache.spark.sql.{Row, SparkSession}
import org.deeplearning4j.spark.stats.StatsUtils
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ListBuffer

object NextClosePriceTask {

  def main(args: Array[String]): Unit = {
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

    val dataset = spark.read.format("com.databricks.spark.csv")
      .option("header", "true") // The CSV file has header and use them as column names
      .option("inferSchema", "true").load("./GOOG.csv")

    // Geting the NextOpenPrice for all dataset
    dataset.createOrReplaceTempView("temp_stocks")

    val nextOpenDatasetSql = "select date_add(Date, -1) as Date, Close as NextClose from temp_stocks "

    val nextOpenDataset = spark.sql(nextOpenDatasetSql)
    nextOpenDataset.createOrReplaceTempView("temp_next_close_price")

    val sql = "select s.Date, s.Open, s.Close, s.Low, s.High, s.Volume, c.NextClose from temp_stocks s, temp_next_close_price c where to_date(s.Date) = c.Date order by s.Date"
    val filter = "Date is not null and Open is not null and Close is not null and Low is not null and High  is not null and Volume is not null and NextClose is not null"

    val updatedDataset = spark.sql(sql).filter(filter).distinct().sort("Date")

    val min = updatedDataset.groupBy().min("Open", "Close", "Low", "High", "Volume").head()

    val max = updatedDataset.groupBy().max("Open", "Close", "Low", "High", "Volume").head()

    val miniBatchSize = 64

    val timeFrameSize = 22

    val splitRatio = 0.9

    val list = updatedDataset.collectAsList()

    val splitAt = list.size * splitRatio

    val trainingList = list.subList(0, splitAt.intValue())

    val testList = list.subList(splitAt.intValue(), list.size())

    val trainingSet = spark.sparkContext.parallelize(createTrainingSet(miniBatchSize, timeFrameSize, min, max, trainingList))

    val testSet = createTestSet(timeFrameSize, min, max, testList)

    val sparkNetwork = RNNBuilder.build(5, 1, 1, sc)

    val epochs = 100

    for (i <- 1 to epochs) {
      sparkNetwork.fit(trainingSet)
    }

    //sparkNetwork.fit(trainingData)

    val stats = sparkNetwork.getSparkTrainingStats //Get the collect stats information
    StatsUtils.exportStatsAsHtml(stats, "SparkStats.html", sc)

    testSet.foreach(set => {
      val actual = set.getLabels.getDouble(timeFrameSize - 1) * (max.getDouble(1) - min.getDouble(1)) + min.getDouble(1)
      val prediction = sparkNetwork.getNetwork.rnnTimeStep(set.getFeatures).getDouble(timeFrameSize - 1) * (max.getDouble(1) - min.getDouble(1)) + min.getDouble(1)

      println(actual + " -> " + prediction)
    })

    val net = sparkNetwork.getNetwork

    val locationToSave = new File("./lstm_model.zip")
    // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
    ModelSerializer.writeModel(net, locationToSave, true)
    spark.close()
  }

  def createTrainingSet(miniBatchSize: Int, timeFrameSize: Int, min: Row, max: Row, rows: util.List[Row]): ListBuffer[DataSet] = {
    var start = 0
    var end = Math.min(timeFrameSize, rows.size)
    var currentBatchSize = Math.min(miniBatchSize, rows.size)
    var result = new ListBuffer[DataSet]()

    do {
      val input = Nd4j.create(Array[Int](currentBatchSize, 5, timeFrameSize), 'f')
      val label = Nd4j.create(Array[Int](currentBatchSize, 1, timeFrameSize), 'f')

      for (i <- 0 to currentBatchSize - 1) {

        for (j <- 0 to timeFrameSize - 1) {
          val price = rows.get(start + j)
          val Open = price.getDouble(1)
          val Close = price.getDouble(2)
          val Low = price.getDouble(3)
          val High = price.getDouble(4)
          val Volume = price.getInt(5).toDouble
          val NextClose = price.getDouble(6)
          val date = price.getTimestamp(0)

          println(date + " - " + currentBatchSize + " - " + timeFrameSize + " - " + start + " - " + end + " - " + i + " - " + j + " - " + rows.size)

          input.putScalar(Array[Int](i, 0, j), (Open - min.getDouble(0)) / (max.getDouble(0) - min.getDouble(0)))
          input.putScalar(Array[Int](i, 1, j), (Close - min.getDouble(1)) / (max.getDouble(1) - min.getDouble(1)))
          input.putScalar(Array[Int](i, 2, j), (Low - min.getDouble(2)) / (max.getDouble(2) - min.getDouble(2)))
          input.putScalar(Array[Int](i, 3, j), (High - min.getDouble(3)) / (max.getDouble(3) - min.getDouble(3)))
          input.putScalar(Array[Int](i, 4, j), (Volume - min.getInt(4).toDouble) / (max.getInt(4).toDouble - min.getInt(4).toDouble))

          label.putScalar(Array[Int](i, 0, j), (NextClose - min.getDouble(1)) / (max.getDouble(1) - min.getDouble(1)))
        }

        start += 1
        end += 1
      }

      result += new DataSet(input, label)

      currentBatchSize = Math.min(miniBatchSize, rows.size - end)
    } while (end < rows.size)

    result
  }

  def createTestSet(timeFrameSize: Int, min: Row, max: Row, rows: util.List[Row]): ListBuffer[DataSet] = {
    var start = 0
    var end = Math.min(timeFrameSize, rows.size)
    var result = new ListBuffer[DataSet]()

    do {
      val input = Nd4j.create(Array[Int](timeFrameSize, 5), 'f')
      val label = Nd4j.create(Array[Int](timeFrameSize, 1), 'f')

      for (j <- 0 to timeFrameSize - 1) {
        val price = rows.get(start + j)
        val Open = price.getDouble(1)
        val Close = price.getDouble(2)
        val Low = price.getDouble(3)
        val High = price.getDouble(4)
        val Volume = price.getInt(5).toDouble
        val NextClose = price.getDouble(6)
        val date = price.getTimestamp(0)

        println(date + " - " + " - " + timeFrameSize + " - " + start + " - " + end + " - " + j + " - " + rows.size)

        input.putScalar(Array[Int](j, 0), (Open - min.getDouble(0)) / (max.getDouble(0) - min.getDouble(0)))
        input.putScalar(Array[Int](j, 1), (Close - min.getDouble(1)) / (max.getDouble(1) - min.getDouble(1)))
        input.putScalar(Array[Int](j, 2), (Low - min.getDouble(2)) / (max.getDouble(2) - min.getDouble(2)))
        input.putScalar(Array[Int](j, 3), (High - min.getDouble(3)) / (max.getDouble(3) - min.getDouble(3)))
        input.putScalar(Array[Int](j, 4), (Volume - min.getInt(4).toDouble) / (max.getInt(4).toDouble - min.getInt(4).toDouble))

        label.putScalar(Array[Int](j, 0), (NextClose - min.getDouble(1)) / (max.getDouble(1) - min.getDouble(1)))
      }

      start += 1
      end += 1

      result += new DataSet(input, label)

    } while (end < rows.size)

    result
  }
}
