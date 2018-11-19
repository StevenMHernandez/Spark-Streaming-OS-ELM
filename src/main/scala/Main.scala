import functions._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Main {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession
      .builder
      .appName("ELM-Example")
      .master("local[*]")
      .getOrCreate()

    val featuresUDF = udf { (features: Any) =>
      features.asInstanceOf[GenericRowWithSchema].toSeq.toArray
        .map(x => x.asInstanceOf[String].toDouble)
    }

    var training = spark.read.format("csv")
      .load("src/main/resources/train_XY.csv")

    training = training
      .withColumn("input", featuresUDF(struct(training.select("_c0", "_c1", "_c2", "_c3", "_c4").columns.map(training(_)): _*)))
      .withColumn("output", featuresUDF(struct(training.select("_c5", "_c6").columns.map(training(_)): _*)))
      .select("input", "output")

    training.show()

    var testing = spark.read.format("csv")
      .load("src/main/resources/test_XY.csv")

    testing = testing
      .withColumn("input", featuresUDF(struct(testing.select("_c0", "_c1", "_c2", "_c3", "_c4").columns.map(testing(_)): _*)))
      .withColumn("output", featuresUDF(struct(testing.select("_c5", "_c6").columns.map(testing(_)): _*)))
      .select("input", "output")

    val elm = new OSELM()
      .setHiddenNodes(5)
      .setActivationFunction(ELM.SIGMOID)

    var model: ELMModel = null
    var predictions: DataFrame = null

    time("Training Time:", {
      println("got it!")
      model = elm.fit(training.limit(5))
    })

    time("Testing Time:", {
      predictions = model.transform(testing)
        .withColumn("actualX", getNth(0)(col("output")))
        .withColumn("actualY", getNth(1)(col("output")))
        .withColumn("predictionX", getNth(0)(col("predictions")))
        .withColumn("predictionY", getNth(1)(col("predictions")))
        .select("actualX", "actualY", "predictionX", "predictionY")
      predictions.collect()
    })

    evaluate(predictions)

    predictions.show()


    time("Incremental Training Time:", {
      println("got it!")
      model = elm.fit(training.limit(10))
    })

    time("Testing Time:", {
      predictions = model.transform(testing)
        .withColumn("actualX", getNth(0)(col("output")))
        .withColumn("actualY", getNth(1)(col("output")))
        .withColumn("predictionX", getNth(0)(col("predictions")))
        .withColumn("predictionY", getNth(1)(col("predictions")))
        .select("actualX", "actualY", "predictionX", "predictionY")
      predictions.collect()
    })

    evaluate(predictions)

    predictions.show()


    time("Incremental Training Time:", {
      println("got it!")
      model = elm.fit(training)
    })

    time("Testing Time:", {
      predictions = model.transform(testing)
        .withColumn("actualX", getNth(0)(col("output")))
        .withColumn("actualY", getNth(1)(col("output")))
        .withColumn("predictionX", getNth(0)(col("predictions")))
        .withColumn("predictionY", getNth(1)(col("predictions")))
        .select("actualX", "actualY", "predictionX", "predictionY")
      predictions.collect()
    })

    evaluate(predictions)

    predictions.show()
  }
}
