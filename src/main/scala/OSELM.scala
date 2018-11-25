import breeze.linalg.pinv
import org.apache.spark.ml.util._
import org.apache.spark.sql._

class OSELM(override val uid: String) extends ELM {
  def this() = this(Identifiable.randomUID("oselm"))

  private var model: ELMModel = _

  override def fit(dataset: Dataset[_]): ELMModel = {
    if (model != null) {
      this.model = incrementalLearn(dataset)
    } else {
      this.model = batchLearn(dataset)
    }

    this.model
  }

  def incrementalLearn(dataset: Dataset[_]): ELMModel = {
    val l = dataset.collect().length

    val input = dataset.select("input")
    val output = dataset.select("output")

    val (inputMatrix, maxX, minX) = normalizeMatrix(datasetToDenseMatrix(input, l))
    val (outputMatrix, maxY, minY) = normalizeMatrix(datasetToDenseMatrix(output, l))

    val T_k1 = outputMatrix

    val a = model.a
    val bias = model.bias

    val H_k1 = buildHiddenLayer(activationFunction, a, bias, inputMatrix)

    val K_k1 = model.K + H_k1.t * H_k1
    val beta = model.beta + pinv(K_k1) * H_k1.t * (T_k1 - H_k1 * model.beta)

    new ELMModel(a, bias, beta, K_k1, activationFunction, maxX, minX, maxY, minY)
  }
}
