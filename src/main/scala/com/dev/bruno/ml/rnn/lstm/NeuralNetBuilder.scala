package com.dev.bruno.ml.rnn.lstm

import java.util.Collections

import org.apache.spark.SparkContext
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{BackpropType, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.ui.stats.StatsListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

object NeuralNetBuilder {

  private val learningRate = 0.05
  private val iterations = 1
  private val seed = 12345

  private val lstmLayerSize = 512
  private val denseLayerSize = 32
  private val dropoutRatio = 0.2

  //Set up the Spark-specific configuration
  /* How frequently should we average parameters (in number of minibatches)?
  Averaging too frequently can be slow (synchronization + serialization costs) whereas too infrequently can result
  learning difficulties (i.e., network may not converge) */
  private val averagingFrequency = 4

  def build(nIn: Int, nOut: Int, timeFrameSize: Int, batchSize: Int, sc: SparkContext): SparkDl4jMultiLayer = {

    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .learningRate(learningRate)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.RMSPROP)
      .regularization(true)
      .l2(1e-4)
      .list
      .layer(0, new GravesLSTM.Builder()
        .nIn(nIn)
        .nOut(lstmLayerSize)
        .activation(Activation.TANH)
        .gateActivationFunction(Activation.HARDSIGMOID)
        .dropOut(dropoutRatio).build)
      .layer(1, new DenseLayer.Builder()
        .nIn(lstmLayerSize)
        .nOut(denseLayerSize)
        .activation(Activation.RELU).build)
      .layer(2, new RnnOutputLayer.Builder()
        .nIn(denseLayerSize)
        .nOut(nOut)
        .activation(Activation.IDENTITY)
        .lossFunction(LossFunctions.LossFunction.MSE).build
      )
      .backpropType(BackpropType.TruncatedBPTT)
      .tBPTTForwardLength(timeFrameSize)
      .tBPTTBackwardLength(timeFrameSize)
      .pretrain(false)
      .backprop(true)
      .build

    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
      .averagingFrequency(averagingFrequency)
      .workerPrefetchNumBatches(0)
      .saveUpdater(true) //save things like adagrad squared gradient histories
      .batchSizePerWorker(batchSize).build()


    val net = new SparkDl4jMultiLayer(sc, conf, trainingMaster)
    net.setCollectTrainingStats(true)

    val remoteUIRouter = new RemoteUIStatsStorageRouter("http://localhost:9000")
    net.setListeners(remoteUIRouter, Collections.singletonList(new StatsListener(null)))

    net
  }
}