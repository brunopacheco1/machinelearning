name := "machinelearning"

version := "1.0"

scalaVersion := "2.11.12"

classpathTypes += "maven-plugin"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.2"

libraryDependencies += "org.apache.spark" %% "spark-hive" % "2.1.2"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.1.2"

libraryDependencies += "org.deeplearning4j" %% "dl4j-spark-ml" % "0.9.1_spark_2"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.9.1"

libraryDependencies += "org.slf4j" % "slf4j-simple" % "1.7.21"

libraryDependencies += "co.theasi" %% "plotly" % "0.2.0"