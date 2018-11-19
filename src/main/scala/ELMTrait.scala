import breeze.linalg.{*, DenseMatrix, DenseVector, max, min}
import breeze.numerics.{sigmoid, signum, tanh}
import org.apache.spark.SparkException

trait ELMTrait {
  def G(activationFunction: String, a: DenseVector[Double], b: Double, x: DenseVector[Double]) = {
    val in = a.t * x + b

    activationFunction match {
      case ELM.SIGMOID => sigmoid(in)
      case ELM.SIGNUM => signum(in)
      case ELM.TANH => tanh(in)
      case _ => throw new SparkException(s"Unknown ELM Activation Function: '$activationFunction'")
    }
  }

  def buildHiddenLayer(activationFunction: String, a: DenseMatrix[Double], bias: DenseVector[Double], input: DenseMatrix[Double]): DenseMatrix[Double] = {
    val l = input.rows
    val num_hidden_nodes = a.rows

    val rangeInputs = 0 until l
    val rangeHiddenLayers = 0 until num_hidden_nodes

    val H = DenseMatrix.zeros[Double](l, num_hidden_nodes)

    rangeInputs.foreach(x => {
      rangeHiddenLayers.foreach(y => {
        H(x, y) = G(activationFunction, a(y, ::).t, bias(y), input(x, ::).t)
      })
    })

    H
  }

  def normalizeMatrix(m: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {
    val maxM = max(m(::, *)).t
    val minM = min(m(::, *)).t

    normalizeMatrix(m, maxM, minM)
  }

  def normalizeMatrix(m: DenseMatrix[Double], maxM: DenseVector[Double], minM: DenseVector[Double]): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {
    val rangeCol = 0 until m.cols
    val rangeRow = 0 until m.rows

    rangeRow.foreach(i => {
      rangeCol.foreach(j => {
        m(i, j) = (m(i, j) - minM(j)) / (maxM(j) - minM(j))
      })
    })

    (m, maxM, minM)
  }

  def denormalizeMatrix(m: DenseMatrix[Double], maxM: DenseVector[Double], minM: DenseVector[Double]): DenseMatrix[Double] = {
    val rangeCol = 0 until m.cols
    val rangeRow = 0 until m.rows

    rangeRow.foreach(i => {
      rangeCol.foreach(j => {
        m(i, j) = (m(i, j) * (maxM(j) - minM(j))) + minM(j)
      })
    })

    m
  }
}
