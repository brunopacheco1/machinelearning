package com.dev.bruno.ml.deeplearning.lstm

import java.util

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ListBuffer

object DataSetBuilder {

  def build(
             features: Array[String],
             labels: Array[String],
             miniBatchSize: Int,
             timeFrameSize: Int,
             min: Row,
             max: Row,
             rows: util.List[Row],
             splitRatio: Double,
             sc: SparkContext
           ): (RDD[DataSet], ListBuffer[DataSet], ListBuffer[DataSet]) = {

    val list = createTimeFrameList(timeFrameSize, rows)

    val splitAt = list.size * splitRatio

    val (trainingList, testList) = list.splitAt(splitAt.intValue())

    val trainingSet = createTrainingSet(features, labels, miniBatchSize, timeFrameSize, min, max, trainingList)

    val trainingTestSet = createTestSet(features, labels, timeFrameSize, min, max, trainingList)

    val testSet = createTestSet(features, labels, timeFrameSize, min, max, testList)

    val trainingSetRDD = sc.parallelize(trainingSet)

    (trainingSetRDD, trainingTestSet, testSet)
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
