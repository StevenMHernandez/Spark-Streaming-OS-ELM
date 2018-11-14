import breeze.linalg.{*, DenseMatrix, DenseVector, max, min}
import breeze.numerics.sigmoid

trait ELMTraits {
  def G(a: DenseVector[Double], b: Double, x: DenseVector[Double]) = {
    sigmoid(a.t * x + b)
//    signum(a.t * x + b)
//    tanh(a.t * x + b)
  }

  def buildHiddenLayer(a: DenseMatrix[Double], bias: DenseVector[Double], input: DenseMatrix[Double]): DenseMatrix[Double] = {
    val l = input.rows
    val num_hidden_nodes = a.rows

    val rangeInputs = 0 until l
    val rangeHiddenLayers = 0 until num_hidden_nodes

    val H = DenseMatrix.zeros[Double](l, num_hidden_nodes)

    rangeInputs.foreach(x => {
      rangeHiddenLayers.foreach(y => {
        H(x, y) = G(a(y, ::).t, bias(y), input(x, ::).t)
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
