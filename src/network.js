// Et lite nettverk fra scratch. Ingen bibliotek som skjuler regningen for oss.
// To innganger → åtte tanh-nevroner → én sigmoid-utgang.

export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

export function createNetwork({ seed = 9173, hiddenSize = 8 } = {}) {
  const random = seededRandom(seed);
  return {
    w1: Array.from({ length: hiddenSize }, () => [random() * 2 - 1, random() * 2 - 1]),
    b1: Array.from({ length: hiddenSize }, () => random() * 0.4 - 0.2),
    w2: Array.from({ length: hiddenSize }, () => random() * 2 - 1),
    b2: 0,
    steps: 0,
  };
}

function sigmoid(z) {
  // Denne varianten unngår å regne exp() av store positive tall.
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const exp = Math.exp(z);
  return exp / (1 + exp);
}

export function forward(network, x, y) {
  const hidden = network.w1.map((weights, i) =>
    Math.tanh(weights[0] * x + weights[1] * y + network.b1[i])
  );
  const logit = hidden.reduce((sum, value, i) => sum + value * network.w2[i], network.b2);
  return { inputs: [x, y], hidden, logit, probability: sigmoid(logit) };
}

export function loss(network, samples) {
  if (!samples.length) throw new Error('Vi trenger minst ett treningspunkt.');
  let sum = 0;
  for (const [x, y, target] of samples) {
    const { logit } = forward(network, x, y);
    // Binary cross-entropy skrevet med logiten. Samme loss, færre avrundingsproblemer.
    sum += Math.max(logit, 0) - logit * target + Math.log1p(Math.exp(-Math.abs(logit)));
  }
  return sum / samples.length;
}

export function gradients(network, samples) {
  if (!samples.length) throw new Error('Vi trenger minst ett treningspunkt.');
  const hiddenSize = network.w1.length;
  const gradient = {
    w1: Array.from({ length: hiddenSize }, () => [0, 0]),
    b1: Array(hiddenSize).fill(0),
    w2: Array(hiddenSize).fill(0),
    b2: 0,
  };

  for (const [x, y, target] of samples) {
    const { hidden, probability } = forward(network, x, y);
    // Sigmoid + BCE gir denne fine, enkle deriverte ved utgangen.
    const outputDelta = (probability - target) / samples.length;
    gradient.b2 += outputDelta;

    for (let i = 0; i < hiddenSize; i++) {
      gradient.w2[i] += outputDelta * hidden[i];
      // Vi sender feilen bakover gjennom vekten og den deriverte til tanh.
      const hiddenDelta = outputDelta * network.w2[i] * (1 - hidden[i] ** 2);
      gradient.w1[i][0] += hiddenDelta * x;
      gradient.w1[i][1] += hiddenDelta * y;
      gradient.b1[i] += hiddenDelta;
    }
  }
  return gradient;
}

export function trainStep(network, samples, learningRate = 0.2) {
  const gradient = gradients(network, samples);
  // Viktig: alle gradientene er regnet med de gamle vektene først.
  for (let i = 0; i < network.w1.length; i++) {
    network.w1[i][0] -= learningRate * gradient.w1[i][0];
    network.w1[i][1] -= learningRate * gradient.w1[i][1];
    network.b1[i] -= learningRate * gradient.b1[i];
    network.w2[i] -= learningRate * gradient.w2[i];
  }
  network.b2 -= learningRate * gradient.b2;
  network.steps += 1;
  return network;
}

export function accuracy(network, samples) {
  return samples.filter(([x, y, target]) => Number(forward(network, x, y).probability >= 0.5) === target).length / samples.length;
}
