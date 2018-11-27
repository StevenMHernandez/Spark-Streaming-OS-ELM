import functions._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.{DataFrame, SparkSession}

object MainSimulationFinal {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession
      .builder
      .appName("OSELM-Simulation-One")
      .master("local[*]")
      .getOrCreate()

    var training1 = withInputColumns(spark.read.format("csv")
      .load("src/main/resources/simulationFinal/train_XY_1.csv"))
      .orderBy(rand())

    val oselm = new OSELM()
      .setHiddenNodes(380)
      .setActivationFunction(ELM.SIGNUM)

    var model: ELMModel = null
    var predictions: DataFrame = null

    /*
     * Begin Running Simulation
     */
    val range = 2.to(250002).by(1000)
    range.foreach({ i =>
      spark.sqlContext.clearCache()
      time(s"Training Time ($i):", {
        model = oselm.fit(training1.limit(i))
      })

      time("Testing Time:", {
        predictions = model.transform(training1.limit(i))
        predictions.collect()
      })
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
