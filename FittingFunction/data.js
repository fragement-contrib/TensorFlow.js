export function generateData(numPoints, coeff, sigma = 0.04) {
  return tf.tidy(() => {

    // 函数四个系数，变成张量
    let a = tf.scalar(coeff.a), b = tf.scalar(coeff.b), c = tf.scalar(coeff.c), d = tf.scalar(coeff.d)

    // 得到随机的x坐标
    const xs = tf.randomUniform([numPoints], -1, 1)

    // 根据xs计算出ys
    // y = a * x ^ 3 + b * x ^ 2 + c * x + d
    const three = tf.scalar(3, 'int32')
    const ys = a.mul(xs.pow(three)).add(b.mul(xs.square())).add(c.mul(xs)).add(d)

      // 补充随机
      .add(tf.randomNormal([numPoints], 0, sigma))

    // 序列化
    // 也就是把ys的值规范在0～1
    const ymin = ys.min()
    const ymax = ys.max()
    const yrange = ymax.sub(ymin)
    const ysNormalized = ys.sub(ymin).div(yrange)

    return {
      xs,
      ys: ysNormalized
    }
  })
}
