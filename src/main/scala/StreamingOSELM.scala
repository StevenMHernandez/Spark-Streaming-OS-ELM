import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.dstream.DStream

class StreamingOSELM extends OSELM {
  var model: ELMModel = _

  def trainOn(data: DStream[(Seq[Double], Seq[Double])]): Unit = {
    data.foreachRDD { (rdd, time) =>
      if (!rdd.isEmpty) {
        val spark = SparkSession.builder.config(rdd.sparkContext.getConf).getOrCreate()
        import spark.implicits._

        val dataset = rdd.toDF("input", "output")

        this.model = fit(dataset)
      }
    }
  }

  def predictOn(data: DStream[(Seq[Double], Seq[Double])]): DStream[Seq[Double]] = {
    data.map(x => this.model.transform(x))
  }
}
