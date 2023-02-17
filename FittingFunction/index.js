import { generateData } from './data.js'

export default function () {

    // 第1步：设置变量

    const a = tf.variable(tf.scalar(Math.random()))
    const b = tf.variable(tf.scalar(Math.random()))
    const c = tf.variable(tf.scalar(Math.random()))
    const d = tf.variable(tf.scalar(Math.random()))

    // 第2步：建立模型

    function predict(x) {
        // y = a * x ^ 3 + b * x ^ 2 + c * x + d
        return tf.tidy(() => {
            return a.mul(x.pow(tf.scalar(3))) // a * x^3
                .add(b.mul(x.square())) // + b * x ^ 2
                .add(c.mul(x)) // + c * x
                .add(d) // + d
        })
    }

    // 第3步：训练模型

    // 定义损失函数
    function loss(predictions, labels) {
        // 将labels（实际的值）进行抽象
        // 然后获取平均数.
        const meanSquareError = predictions.sub(labels).square().mean()
        return meanSquareError
    }

    // 定义训练循环
    function train(xs, ys, numIterations = 75) {

        const learningRate = 0.5;
        const optimizer = tf.train.sgd(learningRate);

        for (let iter = 0; iter < numIterations; iter++) {
            optimizer.minimize(() => {
                const predsYs = predict(xs)
                return loss(predsYs, ys)
            })
        }
    }

    let input = generateData(100, { a: -.8, b: -.2, c: .9, d: .5 })

    train(input.xs, input.ys)

    // 第4步：绘图查看效果

    let painter = document.getElementById('mycanvas').getContext('2d')

    // 先绘制点
    let xsArray = input.xs.dataSync()
    let ysArray = input.ys.dataSync()

    painter.fillStyle = "blue"
    for (let i = 0; i < 100; i++) {
        painter.beginPath()
        painter.rect((xsArray[i] + 1) * 0.5 * 400, 400 - ysArray[i] * 400, 5, 5)
        painter.fill()
    }

    // 再绘制连线

    let _a = a.dataSync()[0]
    let _b = b.dataSync()[0]
    let _c = c.dataSync()[0]
    let _d = d.dataSync()[0]

    painter.strokeStyle = "red"
    painter.beginPath()
    for (let i = -1; i < 1; i += 0.01) {

        // y = a * x ^ 3 + b * x ^ 2 + c * x + d
        let y = _a * i * i * i + _b * i * i + _c * i + _d
        painter.lineTo((i + 1) * 0.5 * 400, 400 - y * 400)
    }
    painter.stroke()
} 