import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.ml.Model
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class ELMModel(
                val a: DenseMatrix[Double],
                val bias: DenseVector[Double],
                val beta: DenseMatrix[Double],
                val activationFunctionType: String,
                val maxX: DenseVector[Double],
                val minX: DenseVector[Double],
                val maxY: DenseVector[Double],
                val minY: DenseVector[Double]
              ) extends Model[ELMModel] with ELMTraits {

  override def transform(dataset: Dataset[_]): DataFrame = {
    val predictUDF = udf { (features: Any) =>
      val arr = features.asInstanceOf[GenericRowWithSchema].toSeq.toArray
        .map(x => x.asInstanceOf[String].toDouble)

      val (m, _, _) = normalizeMatrix(new DenseMatrix[Double](1, 5, arr), maxX, minX)

      predict(m)
    }

    dataset.withColumn("predictions", predictUDF(struct(dataset.columns.map(dataset(_)): _*)))
  }

  protected def predict(features: DenseMatrix[Double]): Array[Double] = {
    val output = buildHiddenLayer(a, bias, features) * beta
    denormalizeMatrix(output, maxY, minY).toArray
  }

  override def copy(extra: ParamMap): ELMModel = {
    defaultCopy(extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override val uid: String = Identifiable.randomUID("elm-model")
}
