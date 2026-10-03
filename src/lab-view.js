import { createNetwork, forward, loss, trainStep, accuracy } from './network.js';
import { DATASETS, getDataset } from './datasets.js';

const CYAN = [20, 168, 179];
const RED = [233, 35, 64];
const INK = [17, 23, 22];
const MAP_SIZE = 144;
const MAP_EXTENT = 1.12;

function blend(from, to, amount) {
  return from.map((value, i) => Math.round(value + (to[i] - value) * amount));
}

function signed(value) {
  return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(2)}`;
}

function networkMarkup(t) {
  const inputNodes = [{ x: 48, y: 115, name: 'x' }, { x: 48, y: 220, name: 'y' }];
  const hiddenNodes = Array.from({ length: 8 }, (_, i) => ({ x: 210, y: 38 + i * 35, name: `h${i + 1}` }));
  const output = { x: 365, y: 163, name: 'p' };
  const wires = [];
  for (let i = 0; i < hiddenNodes.length; i++) {
    const hidden = hiddenNodes[i];
    for (let input = 0; input < 2; input++) {
      const start = inputNodes[input];
      wires.push(`<path class="lab-wire" data-wire="in-${i}-${input}" d="M ${start.x + 17} ${start.y} C 121 ${start.y}, 151 ${hidden.y}, ${hidden.x - 15} ${hidden.y}"><title></title></path>`);
    }
    wires.push(`<path class="lab-wire" data-wire="out-${i}" d="M ${hidden.x + 15} ${hidden.y} C 272 ${hidden.y}, 293 ${output.y}, ${output.x - 20} ${output.y}"><title></title></path>`);
  }
  function node(type, i, point, radius) {
    return `<g class="lab-neuron" data-neuron="${type}-${i}" transform="translate(${point.x} ${point.y})" tabindex="0" role="button" aria-label="${t('Vis regningen for', 'Show the calculation for')} ${point.name}"><title></title><circle class="lab-neuron-ring" r="${radius + 4}"></circle><circle class="lab-neuron-fill" r="${radius}"></circle><text class="lab-neuron-value" y="3">0.00</text><text class="lab-neuron-name" x="${radius + 9}" y="3">${point.name}</text></g>`;
  }
  return `<svg class="lab-network" viewBox="0 0 420 330" aria-label="${t('Nettverk med to innganger, åtte skjulte nevroner og én utgang. Velg et nevron for å se regningen.', 'Network with two inputs, eight hidden neurons and one output. Select a neuron to see its calculation.')}"><text class="lab-layer-label" x="48" y="12" text-anchor="middle">${t('INN', 'INPUT')}</text><text class="lab-layer-label" x="210" y="12" text-anchor="middle">tanh</text><text class="lab-layer-label" x="365" y="12" text-anchor="middle">sigmoid</text><g class="lab-wires">${wires.join('')}</g>${inputNodes.map((point, i) => node('input', i, point, 17)).join('')}${hiddenNodes.map((point, i) => node('hidden', i, point, 14)).join('')}${node('output', 0, output, 20)}<text class="lab-layer-note" x="210" y="320" text-anchor="middle">2 → 8 → 1</text></svg>`;
}

// Nettstedet og den frittstående laben monterer nøyaktig samme UI.
export function mountLabUI(host, language = 'nb', { onStep, eventsURL = null } = {}) {
  const t = (nb, en) => language === 'nb' ? nb : en;
  const locale = language === 'nb' ? 'nb' : 'en';
  let dataset = getDataset('xor');
  let network = createNetwork();
  let point = [-1, 1];
  let selectedNeuron = 'hidden-0';
  let running = false;
  let disposed = false;
  let frameId = 0;
  let lastDraw = 0;
  let lastSignalStep = 0;
  let trainingLimit = dataset.maxSteps;
  let lossHistory = [{ step: 0, value: loss(network, dataset.samples) }];

  host.classList.add('maki-lab');
  host.innerHTML = `<div class="lab-toolbar"><label class="lab-dataset-label" for="lab-dataset">${t('Hva skal vi lære?', 'What shall we learn?')}<select id="lab-dataset" data-dataset>${Object.entries(DATASETS).map(([id, item]) => `<option value="${id}">${item.label[locale]}</option>`).join('')}</select></label><div class="lab-controls"><button type="button" data-train>${t('Tren nettverket', 'Train the network')} →</button><button type="button" data-one-step aria-label="${t('Tren ett steg', 'Train one step')}" title="${t('Tren ett steg', 'Train one step')}">+1</button><button type="button" data-reset aria-label="${t('Nullstill vekter og trening', 'Reset weights and training')}" title="${t('Nullstill vekter og trening', 'Reset weights and training')}">↺</button></div></div>
    <p class="lab-task" data-task>${dataset.description[locale]}</p>
    <div class="lab-neural-panel"><div class="lab-panel-heading"><span>${t('Dette er selve nettverket.', 'This is the actual network.')}</span><span data-selected-point>x −1.00 · y +1.00</span></div>${networkMarkup(t)}<div class="lab-neuron-detail" data-neuron-detail></div><div class="lab-neural-legend"><span><i class="lab-key cyan"></i>${t('Cyan: positiv vekt / aktivering', 'Cyan: positive weight / activation')}</span><span><i class="lab-key red"></i>${t('Rød: negativ vekt / aktivering', 'Red: negative weight / activation')}</span><span>${t('Tykkere kobling = større |vekt|.', 'Thicker connection = larger |weight|.')}</span></div><p class="lab-note">${t('Innganger og tanh-nevroner går fra −1 til +1. Utgangen går fra 0 til 1. Lysere nevron = større |aktivering|. Klikk på et nevron for å se regningen.', 'Inputs and tanh neurons range from −1 to +1. The output ranges from 0 to 1. Brighter neuron = larger |activation|. Click a neuron to see its calculation.')}</p></div>
    <div class="lab-grid"><div class="lab-map-panel"><div class="lab-panel-heading"><span>${t('Hva ser nettverket?', 'What does the network see?')}</span><span data-sample-count>4 ${t('treningspunkter', 'training points')}</span></div><div class="lab-field"><canvas class="lab-map" width="${MAP_SIZE}" height="${MAP_SIZE}" tabindex="0" role="button" aria-label="${t('Beslutningskart. Klikk for å velge inngangspunkt. Piltaster flytter punktet, Enter velger sentrum.', 'Decision map. Click to select an input point. Arrow keys move the point, Enter selects the centre.')}"></canvas><span class="lab-axis-y">y</span><span class="lab-axis-x">x</span><span class="lab-axis-max">+1</span><span class="lab-axis-min">−1</span></div><div class="lab-probability-scale"><span>0</span><i></i><span>1</span></div><p class="lab-note">${t('Flaten: modellens anslag på klasse 1. Små ringer: fasiten i treningsdataene (cyan = 1, mørk = 0). Kryss: punktet du undersøker.', 'Surface: the model’s estimate for class 1. Small rings: the training labels (cyan = 1, dark = 0). Cross: the point you are inspecting.')}</p><p class="lab-map-prompt">${t('Klikk et sted. Se hva hvert nevron gjør.', 'Click anywhere. See what each neuron does.')}</p></div><div class="lab-readout"><div class="lab-panel-heading"><span>${t('Vi trener faktisk.', 'We are actually training.')}</span></div><div class="lab-stats"><div><b data-steps>0</b><span>${t('steg', 'steps')}</span></div><div><b data-loss>—</b><span>loss · BCE</span></div><div><b data-accuracy>—</b><span>${t('riktige treningspunkter', 'correct training points')}</span></div></div><svg class="lab-loss-chart" viewBox="0 0 240 70" role="img" aria-label="${t('Faktisk loss gjennom treningen, med logaritmisk y-akse.', 'Actual loss during training, with a logarithmic y-axis.')}"><path class="lab-loss-guide" d="M 0 8 H 240 M 0 35 H 240 M 0 62 H 240"></path><path data-loss-line class="lab-loss-line" d=""></path></svg><div class="lab-chart-labels"><span>${t('loss ↓ · log-skala', 'loss ↓ · log scale')}</span><span data-chart-end>0 ${t('steg', 'steps')}</span></div><div class="lab-results" data-results></div><p class="lab-note" data-results-note></p></div></div>
    <p class="lab-status" data-status role="status">${t('Klar. Vektene er tilfeldige. Prøv å trene!', 'Ready. The weights are random. Try training!')}</p><p class="lab-explanation">${t('Dette kjører her i nettleseren: egne vekter, egne gradienter, ekte backprop. Ingen KI-tjeneste bak. Tallene i nevronene gjelder akkurat punktet du har valgt.', 'This runs here in your browser: real weights, real gradients, actual backprop. No AI service behind it. The numbers in the neurons are for the exact point you selected.')}</p><div class="lab-footer"><a href="https://github.com/AndersErikstad/maki-fra-scratch" target="_blank" rel="noopener noreferrer">${t('Se koden. Bygg videre.', 'See the code. Build on it.')} ↗</a>${eventsURL ? `<a href="${eventsURL}" target="_blank" rel="noopener noreferrer">${t('Finn arrangementer på IFI', 'Find events at IFI')} ↗</a>` : `<button type="button" data-lab-events>${t('Bygg fra scratch med oss', 'Build from scratch with us')} →</button>`}</div>`;

  const canvas = host.querySelector('.lab-map');
  const context = canvas.getContext('2d', { alpha: false });
  const pixels = context.createImageData(MAP_SIZE, MAP_SIZE);
  const trainButton = host.querySelector('[data-train]');
  const stepButton = host.querySelector('[data-one-step]');
  const status = host.querySelector('[data-status]');
  const nodes = [...host.querySelectorAll('[data-neuron]')];
  const wires = [...host.querySelectorAll('[data-wire]')];
  const stepsReadout = host.querySelector('[data-steps]');
  const lossReadout = host.querySelector('[data-loss]');
  const accuracyReadout = host.querySelector('[data-accuracy]');
  const lossLine = host.querySelector('[data-loss-line]');
  const results = host.querySelector('[data-results]');
  const detail = host.querySelector('[data-neuron-detail]');
  const pointReadout = host.querySelector('[data-selected-point]');

  function renderSampleList() {
    const shown = dataset.symbol ? dataset.samples : [dataset.samples[0], dataset.samples[40], dataset.samples[80]];
    results.innerHTML = shown.map(([x, y, target]) => `<div><span>${dataset.symbol ? `${Number(x > 0)} ${dataset.symbol} ${Number(y > 0)} = ${target}` : `(${signed(x)}, ${signed(y)}) → ${target}`}</span><b data-result>—</b></div>`).join('');
    host.querySelector('[data-results-note]').textContent = dataset.symbol ? t('0 og 1 i uttrykkene mates inn som −1 og +1.', '0 and 1 in the expressions are fed in as −1 and +1.') : t('Tre punkter vist her; loss og treff regnes over alle 81.', 'Three points shown here; loss and accuracy use all 81.');
    host.querySelector('[data-sample-count]').textContent = `${dataset.samples.length} ${t('treningspunkter', 'training points')}`;
  }

  function showCalculation(values) {
    const [type, rawIndex] = selectedNeuron.split('-');
    const i = Number(rawIndex);
    if (type === 'input') {
      detail.textContent = `${i === 0 ? 'x' : 'y'} = ${signed(point[i])}. ${t('Koordinaten du valgte. Her er det ingen lært vekt.', 'The coordinate you selected. No learned weight here.')}`;
    } else if (type === 'hidden') {
      const z = network.w1[i][0] * point[0] + network.w1[i][1] * point[1] + network.b1[i];
      detail.textContent = `h${i + 1}: tanh(${signed(z)}) = ${signed(values.hidden[i])} · bias ${signed(network.b1[i])}`;
    } else {
      detail.textContent = `p = sigmoid(${signed(values.logit)}) = ${values.probability.toFixed(3)} · ${t('anslag på klasse 1', 'estimate for class 1')}`;
    }
  }

  function drawNetwork() {
    const values = forward(network, ...point);
    const allWeights = [...network.w1.flat(), ...network.w2];
    const maxWeight = Math.max(1, ...allWeights.map(Math.abs));

    for (const wire of wires) {
      const [type, index, input] = wire.dataset.wire.split('-');
      const i = Number(index);
      const weight = type === 'in' ? network.w1[i][Number(input)] : network.w2[i];
      const strength = Math.abs(weight) / maxWeight;
      const connected = type === 'in'
        ? selectedNeuron === `hidden-${i}` || selectedNeuron === `input-${input}`
        : selectedNeuron === `hidden-${i}` || selectedNeuron === 'output-0';
      wire.setAttribute('stroke', weight >= 0 ? '#14a8b3' : '#e92340');
      wire.setAttribute('stroke-width', (0.65 + 2.8 * strength).toFixed(2));
      wire.setAttribute('opacity', ((connected ? 0.42 : 0.16) + 0.46 * strength).toFixed(2));
      wire.querySelector('title').textContent = `${t('Vekt', 'Weight')} ${signed(weight)}`;
    }

    for (const node of nodes) {
      const [type, index] = node.dataset.neuron.split('-');
      const i = Number(index);
      const value = type === 'input' ? point[i] : type === 'hidden' ? values.hidden[i] : values.probability;
      const color = value >= 0 ? CYAN : RED;
      const brightness = Math.min(1, Math.abs(value));
      node.querySelector('.lab-neuron-fill').setAttribute('fill', `rgb(${blend(INK, color, brightness * 0.92).join(',')})`);
      node.querySelector('.lab-neuron-fill').setAttribute('stroke', `rgb(${blend([99, 117, 116], color, brightness).join(',')})`);
      node.querySelector('.lab-neuron-value').textContent = type === 'output' ? value.toFixed(2) : signed(value);
      node.classList.toggle('is-selected', node.dataset.neuron === selectedNeuron);
      const name = type === 'input' ? (i ? 'y' : 'x') : type === 'hidden' ? `h${i + 1}` : 'p';
      const description = `${name}: ${value.toFixed(3)}`;
      node.querySelector('title').textContent = description;
      node.setAttribute('aria-label', `${description}. ${t('Vis regningen.', 'Show the calculation.')}`);
    }
    pointReadout.textContent = `x ${signed(point[0])} · y ${signed(point[1])}`;
    showCalculation(values);
  }

  function screen(value) {
    return (value / MAP_EXTENT + 1) / 2 * (MAP_SIZE - 1);
  }

  function drawMap(recalculate = true) {
    if (recalculate) {
      for (let row = 0; row < MAP_SIZE; row++) {
        const y = (1 - row / (MAP_SIZE - 1) * 2) * MAP_EXTENT;
        for (let column = 0; column < MAP_SIZE; column++) {
          const x = (column / (MAP_SIZE - 1) * 2 - 1) * MAP_EXTENT;
          const probability = forward(network, x, y).probability;
          const offset = (row * MAP_SIZE + column) * 4;
          const color = blend(INK, CYAN, probability);
          pixels.data[offset] = color[0];
          pixels.data[offset + 1] = color[1];
          pixels.data[offset + 2] = color[2];
          pixels.data[offset + 3] = 255;
        }
      }
    }
    context.putImageData(pixels, 0, 0);
    // Hver ring viser én fasit fra treningsdataene. Bakgrunnen viser prediksjonen.
    for (const [x, y, target] of dataset.samples) {
      context.beginPath();
      context.arc(screen(x), screen(-y), dataset.id === 'circle' ? 2.1 : 4.5, 0, Math.PI * 2);
      context.fillStyle = target ? '#14a8b3' : '#111716';
      context.fill();
      context.lineWidth = 0.9;
      context.strokeStyle = '#f3f0e8';
      context.stroke();
    }
    const x = screen(point[0]);
    const y = screen(-point[1]);
    context.beginPath();
    context.arc(x, y, 7, 0, Math.PI * 2);
    context.lineWidth = 3;
    context.strokeStyle = '#111716';
    context.stroke();
    context.beginPath();
    context.moveTo(x - 5, y);
    context.lineTo(x + 5, y);
    context.moveTo(x, y - 5);
    context.lineTo(x, y + 5);
    context.lineWidth = 1.4;
    context.strokeStyle = '#f3f0e8';
    context.stroke();
  }

  function drawReadout(recordLoss = true) {
    const currentLoss = loss(network, dataset.samples);
    if (recordLoss && lossHistory.at(-1).step !== network.steps) lossHistory.push({ step: network.steps, value: currentLoss });
    stepsReadout.textContent = network.steps.toLocaleString(locale);
    lossReadout.textContent = currentLoss.toFixed(3);
    accuracyReadout.textContent = `${Math.round(accuracy(network, dataset.samples) * 100)}%`;
    const shown = dataset.symbol ? dataset.samples : [dataset.samples[0], dataset.samples[40], dataset.samples[80]];
    results.querySelectorAll('[data-result]').forEach((element, i) => {
      element.textContent = forward(network, shown[i][0], shown[i][1]).probability.toFixed(2);
    });
    // Log-skala gjør at vi fortsatt ser framgang når loss allerede er liten.
    const top = Math.log10(Math.max(1, lossHistory[0].value));
    const bottom = -3.3;
    const end = Math.max(1, network.steps);
    const curve = lossHistory.map((item, i) => {
      const x = item.step / end * 240;
      const y = 8 + Math.max(0, Math.min(1, (top - Math.log10(Math.max(0.00001, item.value))) / (top - bottom))) * 54;
      return `${i ? 'L' : 'M'}${x.toFixed(2)} ${y.toFixed(2)}`;
    }).join(' ');
    lossLine.setAttribute('d', curve);
    host.querySelector('[data-chart-end]').textContent = `${network.steps.toLocaleString(locale)} ${t('steg', 'steps')}`;
  }

  function draw() {
    if (disposed) return;
    drawNetwork();
    drawMap();
    drawReadout();
  }

  function updateControls() {
    trainButton.textContent = running
      ? t('Pause treningen', 'Pause training')
      : network.steps === 0 ? `${t('Tren nettverket', 'Train the network')} →`
        : network.steps >= trainingLimit ? `${t('Tren litt mer', 'Train a little more')} →`
          : `${t('Fortsett', 'Continue')} →`;
    stepButton.disabled = running;
    host.classList.toggle('is-training', running);
  }

  function pause() {
    running = false;
    cancelAnimationFrame(frameId);
    updateControls();
  }

  function frame(time) {
    if (!running || disposed) return;
    for (let i = 0; i < 30 && network.steps < trainingLimit; i++) {
      trainStep(network, dataset.samples, dataset.learningRate);
    }
    if (time - lastDraw >= 80 || network.steps >= trainingLimit) {
      draw();
      lastDraw = time;
    }
    if (network.steps - lastSignalStep >= 150) {
      onStep?.();
      lastSignalStep = network.steps;
    }
    if (network.steps >= trainingLimit) {
      pause();
      status.textContent = t('Ferdig for nå. Prøv andre punkter, eller tren litt mer!', 'Done for now. Try other points, or train a little more!');
      return;
    }
    frameId = requestAnimationFrame(frame);
  }

  trainButton.addEventListener('click', () => {
    if (running) {
      pause();
      draw();
      status.textContent = t('Pause. Du kan fortsatt undersøke punkter og nevroner.', 'Paused. You can still inspect points and neurons.');
    } else {
      if (network.steps >= trainingLimit) trainingLimit += dataset.maxSteps;
      running = true;
      lastDraw = 0;
      updateControls();
      status.textContent = t('Trener. Se hvordan vektene, nevronene og kartet endrer seg.', 'Training. Watch the weights, neurons and map change.');
      onStep?.();
      frameId = requestAnimationFrame(frame);
    }
  });

  stepButton.addEventListener('click', () => {
    trainStep(network, dataset.samples, dataset.learningRate);
    draw();
    updateControls();
    onStep?.();
    status.textContent = t('Ett steg: regn framover, finn gradientene, oppdater vektene.', 'One step: calculate forward, find the gradients, update the weights.');
  });

  function reset() {
    pause();
    network = createNetwork();
    trainingLimit = dataset.maxSteps;
    lastSignalStep = 0;
    lossHistory = [{ step: 0, value: loss(network, dataset.samples) }];
    renderSampleList();
    draw();
    updateControls();
    status.textContent = t('Nullstilt. Samme startvekter, så vi kan sammenligne forsøkene.', 'Reset. The same starting weights, so we can compare runs.');
  }
  host.querySelector('[data-reset]').addEventListener('click', reset);
  host.querySelector('[data-dataset]').addEventListener('change', event => {
    dataset = getDataset(event.target.value);
    point = dataset.id === 'circle' ? [0.25, 0.25] : [-1, 1];
    host.querySelector('[data-task]').textContent = dataset.description[locale];
    reset();
  });

  function selectPoint(x, y) {
    point = [x, y].map(value => Math.max(-1, Math.min(1, value)));
    drawNetwork();
    drawMap(false);
  }
  canvas.addEventListener('pointerdown', event => {
    const bounds = canvas.getBoundingClientRect();
    selectPoint(((event.clientX - bounds.left) / bounds.width * 2 - 1) * MAP_EXTENT, (1 - (event.clientY - bounds.top) / bounds.height * 2) * MAP_EXTENT);
    canvas.focus({ preventScroll: true });
    onStep?.();
  });
  canvas.addEventListener('keydown', event => {
    const directions = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] };
    if (directions[event.key]) {
      event.preventDefault();
      const [x, y] = directions[event.key];
      selectPoint(point[0] + x, point[1] + y);
    } else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      selectPoint(0, 0);
    }
  });
  for (const node of nodes) {
    const select = () => { selectedNeuron = node.dataset.neuron; drawNetwork(); };
    node.addEventListener('click', select);
    node.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); }
    });
  }
  renderSampleList();
  draw();

  return () => {
    disposed = true;
    pause();
    // Ingen bakgrunnstrening fortsetter etter at dialogen lukkes.
  };
}
