package com.dev.bruno.ml.deeplearning.lstm

import org.deeplearning4j.ui.api.UIServer

object TrainingDebugger {

  def main(args: Array[String]): Unit = {

    val uiServer = UIServer.getInstance

    uiServer.enableRemoteListener()
  }
}