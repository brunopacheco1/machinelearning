package com.dev.bruno.ml

import org.apache.spark.SparkConf
import org.apache.spark.graphx.{Graph, VertexId}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.hashing.MurmurHash3


object SparkApp {

  def main(args: Array[String]): Unit = {
    val time = System.currentTimeMillis()

    val config = new SparkConf().setAppName("SparkApp").setMaster("local[*]")

    val warehouseLocation = "file:${system:user.dir}/spark-warehouse"

    val spark = SparkSession
      .builder
      .config(config)
      .config("spark.sql.warehouse.dir", warehouseLocation)
      .enableHiveSupport()
      .getOrCreate()

    val relationships = spark.sql("SELECT father, child FROM relationship").dropDuplicates().rdd.map(row => (row(0).toString, row(1).toString))

    val documents = relationships.groupByKey()

    val edges: RDD[(VertexId, VertexId)] = relationships.map(line => (MurmurHash3.stringHash(line._1), MurmurHash3.stringHash(line._2)))

    println("Total de relacionamentos: " + edges.count())

    val graph = Graph.fromEdgeTuples(edges, 1)

    val ranks = graph.pageRank(0.0001).vertices

    val vertices = documents.map(value => (MurmurHash3.stringHash(value._1).asInstanceOf[org.apache.spark.graphx.VertexId], value._1))

    val rankByVertices = vertices.join(ranks).map(el => Map("id" -> el._2._1, "pagerank" -> el._2._2)).filter(value => value("pagerank") != null && value("pagerank").toString.toDouble > 0)

    println("Total de documentos rankeados: " + rankByVertices.count())

    println("Tempo total gasto: " + (System.currentTimeMillis() - time) + "ms")
  }
}