object SimpleCsvParser {
  def parse(s: String): Array[Double] = {
    s.split(',').map(_.toDouble)
  }
}
