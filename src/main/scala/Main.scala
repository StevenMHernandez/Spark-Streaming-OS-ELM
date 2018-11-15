import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.functions.{udf, _}

object Main {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession
      .builder
      .appName("ELM-Example")
      .master("local")
      .getOrCreate()

    val featuresUDF = udf { (features: Any) =>
      features.asInstanceOf[GenericRowWithSchema].toSeq.toArray
        .map(x => x.asInstanceOf[String].toDouble)
    }

    var training = spark.read.format("csv")
      .load("src/main/resources/train_XY.csv")

    val t = training.select("_c0", "_c1", "_c2", "_c3", "_c4")

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

    val elm = new ELM()
      .setHiddenNodes(50)

    val model = elm.fit(training)

    model.transform(testing).show()
  }
}
