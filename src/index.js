/**
 * @license
 * Copyright 2021 Google LLC
 * ...
 */

// ... (existing imports remain the same) ...

import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';

import * as mpPose from '@mediapipe/pose';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import * as tf from '@tensorflow/tfjs-core';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';

import {Camera} from './camera';
import {RendererWebGPU} from './renderer_webgpu';
import {RendererCanvas2d} from './renderer_canvas2d';
import {setupDatGui} from './option_panel';
import {STATE} from './params';
import {setupStats} from './stats_panel';
import {setBackendAndEnvFlags} from './util';

// Configuration parameters
const DATA_RETENTION_WINDOW = 5000; // 5 seconds in ms
const GRAPH_UPDATE_INTERVAL = 100;   // Time interval in ms for graph updates

// Threshold & alert control
let threshold = 300;       // Default threshold
let alertEnabled = false;  // Whether the audible alert is ON/OFF
let lastBeepTime = 0;      // Keep track of the last beep to avoid spamming

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
let renderer = null;
let useGpuRenderer = false;

// Keep arrays for keypoints 0..4
let keypointsData = {
  0: [],
  1: [],
  2: [],
  3: [],
  4: []
};

let chart;

/**
 * Simple beep function using the Web Audio API.
 * Adjust volume (vol), frequency (freq), and duration (ms) as needed.
 */
function beep(vol, freq, duration) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const oscillator = audioCtx.createOscillator();
  const gainNode = audioCtx.createGain();
  oscillator.connect(gainNode);
  gainNode.connect(audioCtx.destination);

  gainNode.gain.value = vol;         // volume
  oscillator.frequency.value = freq; // frequency
  oscillator.type = 'square';

  oscillator.start();
  oscillator.stop(audioCtx.currentTime + duration * 0.001); // duration in seconds
}

function updateKeypointsData(poses) {
  const currentTime = performance.now();
  if (poses && poses.length > 0) {
    const keypoints = poses[0].keypoints;
    [0, 1, 2, 3, 4].forEach(point => {
      if (keypoints[point]) {
        keypointsData[point].push({ time: currentTime, y: keypoints[point].y });
        // Keep data for the configured retention window
        keypointsData[point] = keypointsData[point].filter(
          data => currentTime - data.time < DATA_RETENTION_WINDOW
        );
      }
    });
  }
  updateGraph();
}

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: {width: 500, height: 500},
        multiplier: 0.75
      });
    case posedetection.SupportedModels.BlazePose:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return posedetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath:
              `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return posedetection.createDetector(
            STATE.model, {runtime, modelType: STATE.modelConfig.type});
      }
    case posedetection.SupportedModels.MoveNet:
      let modelType;
      if (STATE.modelConfig.type == 'lightning') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
      } else if (STATE.modelConfig.type == 'thunder') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      } else if (STATE.modelConfig.type == 'multipose') {
        modelType = posedetection.movenet.modelType.MULTIPOSE_LIGHTNING;
      }
      const modelConfig = {modelType};

      if (STATE.modelConfig.customModel !== '') {
        modelConfig.modelUrl = STATE.modelConfig.customModel;
      }
      if (STATE.modelConfig.type === 'multipose') {
        modelConfig.enableTracking = STATE.modelConfig.enableTracking;
      }
      return posedetection.createDetector(STATE.model, modelConfig);
  }
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let poses = null;
  let canvasInfo = null;

  if (detector != null) {
    beginEstimatePosesStats();
    try {
      if (useGpuRenderer && STATE.model !== 'PoseNet') {
        throw new Error('Only PoseNet supports GPU renderer!');
      }
      if (useGpuRenderer) {
        const [posesTemp, canvasInfoTemp] = await detector.estimatePosesGPU(
            camera.video,
            {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false},
            true
        );
        poses = posesTemp;
        canvasInfo = canvasInfoTemp;
      } else {
        poses = await detector.estimatePoses(
            camera.video,
            {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false}
        );
      }
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }
    endEstimatePosesStats();
    updateKeypointsData(poses);
  }
  
  const rendererParams = useGpuRenderer
    ? [camera.video, poses, canvasInfo, STATE.modelConfig.scoreThreshold]
    : [camera.video, poses, STATE.isModelChanged];
  renderer.draw(rendererParams);
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(renderPrediction);
}

async function app() {
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }
  await setupDatGui(urlParams);

  stats = setupStats();
  const isWebGPU = STATE.backend === 'tfjs-webgpu';
  const importVideo = (urlParams.get('importVideo') === 'true') && isWebGPU;

  camera = await Camera.setup(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);
  await tf.ready();
  detector = await createDetector();
  const canvas = document.getElementById('output');
  canvas.width = camera.video.width;
  canvas.height = camera.video.height;

  useGpuRenderer = (urlParams.get('gpuRenderer') === 'true') && isWebGPU;
  if (useGpuRenderer) {
    renderer = new RendererWebGPU(canvas, importVideo);
  } else {
    renderer = new RendererCanvas2d(canvas);
  }

  // Set up our Chart.js chart
  setupChart();

  // Wire up our new slider and toggle button
  setupThresholdControls();

  renderPrediction();
}

app();

if (useGpuRenderer) {
  renderer.dispose();
}

function setupChart() {
  const ctx = document.getElementById('keypointGraph').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['t', 't-1', 't-2', 't-3', 't-4'],
      datasets: [{
        label: 'Average Y',
        data: [],
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: false,
          min: 0,
          max: 480,
          reverse: true
        },
        x: {
          reverse: true
        }
      }
    }
  });
}

/**
 * Sets up the slider and alert button UI
 */
function setupThresholdControls() {
  const slider = document.getElementById('thresholdSlider');
  const thresholdValueSpan = document.getElementById('thresholdValue');
  const toggleAlertBtn = document.getElementById('toggleAlertBtn');

  // Update threshold when slider changes
  slider.addEventListener('input', (e) => {
    threshold = Number(e.target.value);
    thresholdValueSpan.textContent = threshold.toString();
  });

  // Toggle the audible alert on/off
  toggleAlertBtn.addEventListener('click', () => {
    alertEnabled = !alertEnabled;
    toggleAlertBtn.textContent = alertEnabled ? 'Alert: ON' : 'Alert: OFF';
  });
}

/**
 * Updates the chart data each frame and checks threshold over last 1 second.
 */
function updateGraph() {
  const currentTime = performance.now();

  // Calculate average Y for each of the last 5 keypoints over the entire 5s window
  const graphData = [0, 1, 2, 3, 4].map(point => {
    const filteredData = keypointsData[point].filter(
      d => currentTime - d.time < DATA_RETENTION_WINDOW
    );
    const avgY = filteredData.length === 0 
      ? 0 
      : filteredData.reduce((acc, d) => acc + d.y, 0) / filteredData.length;
    return avgY;
  });

  // Update Chart.js dataset (the array is just the 5 current average points)
  drawGraph(graphData);

  // Also compute average Y across all keypoints in the last 1 second,
  // then see if it exceeds the threshold
  const dataLast1Sec = [];
  [0, 1, 2, 3, 4].forEach(point => {
    keypointsData[point].forEach(entry => {
      if (currentTime - entry.time < 1000) {
        dataLast1Sec.push(entry.y);
      }
    });
  });
  if (dataLast1Sec.length > 0) {
    const avgLast1Sec = dataLast1Sec.reduce((a, b) => a + b, 0) / dataLast1Sec.length;

    if (alertEnabled && avgLast1Sec > threshold) {
      // Simple anti-spam: beep only if at least 1 second has passed since last beep
      if (currentTime - lastBeepTime > 1000) {
        beep(0.2, 440, 300); // volume=0.2, freq=440 Hz, duration=300 ms
        lastBeepTime = currentTime;
      }
    }
  }
}

function drawGraph(data) {
  chart.data.datasets[0].data = data;  // data is an array of 5 average points
  chart.update();
}
