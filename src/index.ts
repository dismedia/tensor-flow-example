import {layers, sequential, Sequential} from "@tensorflow/tfjs-layers";
import {
    tensor2d,
    Tensor,
    losses,
    variableGrads,
    tidy,
    train,
    NamedTensorMap,
    stack,
    mean,
    concat
} from "@tensorflow/tfjs";
import {ActivationIdentifier} from "@tensorflow/tfjs-layers/src/keras_format/activation_config";
import {NamedTensor} from "@tensorflow/tfjs-core/dist/tensor_types";
import {InitializerIdentifier} from "@tensorflow/tfjs-layers/src/initializers";
import {addHeatmap} from "./vis/heatmap";

const func = (...x) => {

    const y1 = x[0] * x[1] * 0.9 + (1 - x[0]) * (1 - x[1]) * 0.9;
    //const y1 = x[0]*x[1];
    //const y1 = Math.sin(x[0] * 4) * Math.cos(x[1] * 6 ) * 0.5 + 0.5;
    //const y1 = x[0]>0.1&&x[0]<0.9 && x[1]>0.3&&x[1]<0.7
    //const y1 = x[0]>0.5?1:0
    //const y1= x[1]>0.3&&x[1]<0.7;
    return tensor2d([y1], [1, 1])
}

const activation: ActivationIdentifier = "tanh"
const kernelInitializer: InitializerIdentifier = null
const model: Sequential = sequential();

const inputLayer = layers.dense({
    units: 2,
    inputShape: [2],
    kernelInitializer,

});

const hiddenLayer1 = layers.dense({
    units: 16,
    activation: activation,
    //kernelInitializer,
    useBias: true
});

const outputLayer = layers.dense({
    units: 1,
    activation: "sigmoid",
    kernelInitializer,
    useBias: true

});

const dim = 10; // error sampling density

model.add(inputLayer);
model.add(hiddenLayer1);
model.add(outputLayer);

const optimizer = train.adam(0.1);

const calculateGradient = () => {

    return tidy(() => {
        const vGrads = variableGrads(() => tidy(() => {
            const x1 = Math.random();
            const x2 = Math.random();
            const labels = func(x1, x2)
            const input = tensor2d([x1, x2], [1, 2])

            return losses.meanSquaredError(
                labels,
                model.predict(input) as Tensor
            ).asScalar();

        }));
        return vGrads.grads;
    })
}

const createBatch = (n: number) => {

    return tidy(() => {
        const gradientsArrays = {}

        for (let i = 0; i < n; i++) {
            const gradient = calculateGradient();
            Object.keys(gradient).forEach((entry) => {
                gradientsArrays[entry] ? gradientsArrays[entry].push(gradient[entry]) : gradientsArrays[entry] = [gradient[entry]]
            })
        }

        const gradientsMeans = {}

        Object.keys(gradientsArrays).forEach(key => {
            gradientsMeans[key] = mean(stack(gradientsArrays[key], 0))
        })

        return gradientsMeans;
    })
}

const epoch = (iterations: number) => {

    for (let i = 0; i < iterations; i++) {
        let batch = createBatch(16);
        optimizer.applyGradients(batch)
    }

}

const calculateDesiredOutputs = () => {
    const desiredOutputs = [];
    for (let y = 0; y < 1; y += 1 / dim) {
        for (let x = 0; x < 1; x += 1 / dim) {
            desiredOutputs.push({x, y, value: func(x, y).dataSync()[0]});
        }
    }

    return desiredOutputs;
}

const calculateNetOutputs = () => {
    const netOutputs = [];

    for (let y = 0; y < 1; y += 1 / dim) {
        for (let x = 0; x < 1; x += 1 / dim) {
            const value = (<any>model.predict(tensor2d([x, y], [1, 2]))).dataSync()[0];
            netOutputs.push({x, y, value});
        }
    }

    return netOutputs
}

const calculateError = (a: { value: number }[], b: { value: number }[]) => {

    let error = 0;

    for (let i = 0; i < a.length; i++) {
        let e = a[i].value - b[i].value;
        error += e * e
    }

    return Math.sqrt(error) / (dim * dim);

}

const run = async () => {

    const desiredOutputs = calculateDesiredOutputs();

    const desiredOutputsHeatmap = addHeatmap({dim});
    desiredOutputsHeatmap.update(desiredOutputs)
    const netOutputHeatmap = addHeatmap({dim});

    let i = 0;

    while (i < 256) {

        epoch(20);

        let netOutputs = calculateNetOutputs();
        console.log("epoch: ", i)
        console.log(calculateError(desiredOutputs, netOutputs))

        netOutputHeatmap.update(netOutputs);
        await new Promise((r) => setTimeout(() => r(), 100));

        i++;
    }

}

run();
