const corners = [
  [-1, -1],
  [-1, 1],
  [1, -1],
  [1, 1],
];

function cornerSamples(targets) {
  return corners.map(([x, y], i) => [x, y, targets[i]]);
}

const circleSamples = [];
for (let row = 0; row < 9; row++) {
  for (let column = 0; column < 9; column++) {
    const x = -1 + column / 4;
    const y = -1 + row / 4;
    circleSamples.push([x, y, Number(x * x + y * y < 0.58 ** 2)]);
  }
}

// Hjørnene bruker -1 og +1 i modellen; i de boolske uttrykkene vises de som 0 og 1.
export const DATASETS = {
  xor: {
    label: { nb: 'XOR · én av dem', en: 'XOR · one of them' },
    description: { nb: 'Én av inngangene er 1, men ikke begge. En rett linje klarer ikke å skille disse punktene.', en: 'One input is 1, but not both. A straight line cannot separate these points.' },
    samples: cornerSamples([0, 1, 1, 0]),
    symbol: '⊕',
    learningRate: 0.2,
    maxSteps: 2400,
  },
  and: {
    label: { nb: 'AND · begge to', en: 'AND · both of them' },
    description: { nb: 'Begge inngangene må være 1. Dette går fint an å skille med en rett linje. Se forskjellen fra XOR!', en: 'Both inputs must be 1. A straight line can separate these points. Compare it with XOR!' },
    samples: cornerSamples([0, 0, 0, 1]),
    symbol: '∧',
    learningRate: 0.2,
    maxSteps: 1600,
  },
  or: {
    label: { nb: 'OR · minst én', en: 'OR · at least one' },
    description: { nb: 'Minst én inngang må være 1. Samme nettverk, andre svar å lære.', en: 'At least one input must be 1. The same network, different answers to learn.' },
    samples: cornerSamples([0, 1, 1, 1]),
    symbol: '∨',
    learningRate: 0.2,
    maxSteps: 1600,
  },
  circle: {
    label: { nb: 'Sirkel · innenfor eller utenfor?', en: 'Circle · inside or outside?' },
    description: { nb: '81 punkter. Nettverket får bare koordinatene og svaret: innenfor eller utenfor sirkelen. Vi gir det ingen formel for en sirkel.', en: '81 points. The network only gets the coordinates and the answer: inside or outside the circle. We give it no formula for a circle.' },
    samples: circleSamples,
    symbol: null,
    learningRate: 0.2,
    maxSteps: 6500,
  },
};

export function getDataset(id) {
  const dataset = DATASETS[id];
  if (!dataset) throw new Error(`Ukjent datasett: ${id}`);
  return { ...dataset, id, samples: dataset.samples.map(sample => [...sample]) };
}
