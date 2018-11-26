import functions._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.{DataFrame, SparkSession}
import java.lang.Math.sqrt

import scala.collection.mutable

object MainSimulationOne {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession
      .builder
      .appName("OSELM-Simulation-One")
      .master("local[*]")
      .getOrCreate()

    var training1 = withInputColumns(spark.read.format("csv")
      .load("src/main/resources/simulationOne/train_XY_1.csv"))
    var training2 = withInputColumns(spark.read.format("csv")
      .load("src/main/resources/simulationOne/train_XY_2.csv"))
    var training3 = withInputColumns(spark.read.format("csv")
      .load("src/main/resources/simulationOne/train_XY_3.csv"))

    var testing1 = withInputColumns(spark.read.format("csv")
      .load("src/main/resources/simulationOne/test_XY_1.csv"))
    var testing2 = withInputColumns(spark.read.format("csv")
      .load("src/main/resources/simulationOne/test_XY_2.csv"))
    var testing3 = withInputColumns(spark.read.format("csv")
      .load("src/main/resources/simulationOne/test_XY_3.csv"))

    var training_small = withInputColumns(spark.read.format("csv")
      .load("src/main/resources/simulationOne/train_small.csv"))

    val oselm = new OSELM()
      .setHiddenNodes(380)
      .setActivationFunction(ELM.SIGNUM)

    var model: ELMModel = null
    var predictions: DataFrame = null

//    time("Train on some initial data to allow spark to 'warm-up': ", {
//      model = oselm.fit(training_small)
//    })

    /*
     * Begin Running Simulation
     */
    Array((training1, testing1), (training2, testing2), (training3, testing3)).foreach({ dfs =>
      time("Training Time:", {
        model = oselm.fit(dfs._1)
      })

      time("Testing Time:", {
        predictions = model.transform(dfs._2)
        predictions.collect()
      })

      val errDistances = predictions.select("predictions").rdd.zip(dfs._2.select("output").rdd).map(r => {
        val firstIsh = r._1.get(0).asInstanceOf[mutable.WrappedArray[Double]]
        val secondIsh = r._2.get(0).asInstanceOf[mutable.WrappedArray[Double]]

        firstIsh.zip(secondIsh).map(z => {
          val difference = z._2 - z._1
          difference * difference
        }).reduce((a, b) => {
          sqrt(a + b)
        })
      })

      println("Error: " + (errDistances.sum() / errDistances.count()))
    })

    spark.stop()
  }

  def withInputColumns(dataFrame: DataFrame): DataFrame = {
    val featuresUDF = udf { features: Any =>
      features.asInstanceOf[GenericRowWithSchema].toSeq.toArray
        .map(x => x.asInstanceOf[String].toDouble)
    }

    dataFrame
      .withColumn("input", featuresUDF(struct(dataFrame.select("_c0", "_c1", "_c2", "_c3").columns.map(dataFrame(_)): _*)))
      .withColumn("output", featuresUDF(struct(dataFrame.select("_c4", "_c5").columns.map(dataFrame(_)): _*)))
      .select("input", "output")
      .cache()
  }

  def withOutputColumns(dataframe: DataFrame): DataFrame = {
    dataframe.withColumn("actualX", getNth(0)(col("output")))
      .withColumn("actualY", getNth(1)(col("output")))
      .withColumn("predictionX", getNth(0)(col("predictions")))
      .withColumn("predictionY", getNth(1)(col("predictions")))
      .select("actualX", "actualY", "predictionX", "predictionY")
  }
}
