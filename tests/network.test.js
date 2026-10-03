import test from 'node:test';
import assert from 'node:assert/strict';
import { createNetwork, forward, loss, gradients, trainStep, accuracy } from '../src/network.js';
import { getDataset, DATASETS } from '../src/datasets.js';

test('Samme seed og data gir nøyaktig samme vekter og prediksjoner', () => {
  const samples = getDataset('xor').samples;
  const a = createNetwork();
  const b = createNetwork();
  for (let step = 0; step < 150; step++) {
    trainStep(a, samples);
    trainStep(b, samples);
  }
  assert.deepEqual(a, b);
  assert.deepEqual(forward(a, 0.23, -0.42), forward(b, 0.23, -0.42));
  assert.notDeepEqual(a, createNetwork({ seed: 42 }));
});

test('Backprop stemmer med finite differences for hver eneste parameter', () => {
  const network = createNetwork({ seed: 123 });
  const samples = getDataset('xor').samples;
  const analytical = gradients(network, samples);
  const epsilon = 1e-5;

  function check(object, key, expected, label) {
    const original = object[key];
    object[key] = original + epsilon;
    const plus = loss(network, samples);
    object[key] = original - epsilon;
    const minus = loss(network, samples);
    object[key] = original;
    const numerical = (plus - minus) / (2 * epsilon);
    assert.ok(Math.abs(numerical - expected) < 1e-7, `${label}: ${numerical} != ${expected}`);
  }

  for (let i = 0; i < network.w1.length; i++) {
    check(network.w1[i], 0, analytical.w1[i][0], `w1[${i}][0]`);
    check(network.w1[i], 1, analytical.w1[i][1], `w1[${i}][1]`);
    check(network.b1, i, analytical.b1[i], `b1[${i}]`);
    check(network.w2, i, analytical.w2[i], `w2[${i}]`);
  }
  check(network, 'b2', analytical.b2, 'b2');
});

for (const id of Object.keys(DATASETS)) {
  test(`Det lille nettverket lærer ${id}`, () => {
    const dataset = getDataset(id);
    const network = createNetwork();
    const initialLoss = loss(network, dataset.samples);
    for (let step = 0; step < dataset.maxSteps; step++) {
      trainStep(network, dataset.samples, dataset.learningRate);
    }
    const finalLoss = loss(network, dataset.samples);
    assert.ok(finalLoss < initialLoss * 0.2, `loss ${initialLoss} → ${finalLoss}`);
    assert.equal(accuracy(network, dataset.samples), 1);
    assert.ok(finalLoss < (id === 'circle' ? 0.1 : 0.02), `Slutt-loss: ${finalLoss}`);
    assert.equal(network.steps, dataset.maxSteps);
  });
}

test('BCE holder seg endelig også ved ekstreme logits', () => {
  const network = createNetwork();
  network.b2 = 1000;
  assert.ok(Number.isFinite(loss(network, [[0, 0, 0]])));
  assert.equal(forward(network, 0, 0).probability, 1);
  network.b2 = -1000;
  assert.ok(Number.isFinite(loss(network, [[0, 0, 1]])));
  assert.equal(forward(network, 0, 0).probability, 0);
});
