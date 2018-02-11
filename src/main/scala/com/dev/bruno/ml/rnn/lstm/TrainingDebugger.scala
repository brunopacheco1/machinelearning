package com.dev.bruno.ml.rnn.lstm

import org.deeplearning4j.ui.api.UIServer

object TrainingDebugger {

  def main(args: Array[String]): Unit = {

    val uiServer = UIServer.getInstance

    uiServer.enableRemoteListener()
  }
}