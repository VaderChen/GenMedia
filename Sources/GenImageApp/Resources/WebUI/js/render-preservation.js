// 全量重繪時要救回來的東西，全部集中在這裡。
//
// render() 會把整棵 DOM 換掉，游標、選取範圍、捲動位置、正在播放的音訊／影片節點與 Web Audio
// 視覺化圖，都無法自己活過這個動作。過去這些補救散落在 render() 裡的八次呼叫，順序錯了就會
// 出現游標跳掉或播放中斷，而且每加一個元件都要記得再補一次。現在只有 withPreservedView 一個
// 進入點：呼叫端負責畫，這裡負責前後把狀態接回去。

const AUDIO_FFT_SIZE = 8192;
const AUDIO_MIN_FREQUENCY_HZ = 18;
const AUDIO_MAX_FREQUENCY_HZ = 24_000;
const AUDIO_MAX_WAVEFORM_POINTS = 1_600;

const audioVisualizerBindings = new Map();
let audioVisualizerContext = null;

/// 在保住可見狀態的前提下重畫 root：paint() 只要把新的 DOM 放進去即可。
export function withPreservedView(root, paint) {
  const viewState = captureViewState(root);
  const preservedMedia = preservePlayingMedia(root);
  const preservedAudio = new Set(
    preservedMedia
      .filter(({ media }) => media.tagName === "AUDIO")
      .map(({ media }) => media),
  );
  disposeAudioVisualizers(preservedAudio);
  paint();
  restoreViewState(root, viewState);
  restorePlayingMedia(root, preservedMedia);
  setupAudioVisualizers(root);
}

function captureViewState(root) {
  const active = document.activeElement;
  const focusKey = active?.dataset?.preserveFocus;
  const focus = focusKey
    ? { key: focusKey, start: active.selectionStart, end: active.selectionEnd }
    : null;
  const scroll = {};
  root.querySelectorAll("[data-scroll-id]").forEach((element) => {
    scroll[element.dataset.scrollId] = { top: element.scrollTop, left: element.scrollLeft };
  });
  return { focus, scroll };
}

function restoreViewState(root, viewState) {
  Object.entries(viewState.scroll).forEach(([key, value]) => {
    const element = root.querySelector(`[data-scroll-id="${key}"]`);
    if (element) {
      element.scrollTop = value.top;
      element.scrollLeft = value.left;
    }
  });
  if (!viewState.focus) return;
  const element = root.querySelector(`[data-preserve-focus="${viewState.focus.key}"]`);
  if (!element) return;
  element.focus({ preventScroll: true });
  if (typeof element.setSelectionRange === "function") {
    element.setSelectionRange(viewState.focus.start, viewState.focus.end);
  }
}

function preservePlayingMedia(root) {
  const occurrenceByIdentity = new Map();
  const preservedMedia = [];
  root.querySelectorAll("audio[data-asset-id], video[data-asset-id]").forEach((media) => {
    const identity = mediaRenderIdentity(media);
    const occurrence = occurrenceByIdentity.get(identity) || 0;
    occurrenceByIdentity.set(identity, occurrence + 1);
    if (media.paused || media.ended) return;

    const host = media.tagName === "AUDIO"
      ? media.closest("[data-audio-visualizer]")
      : media;
    if (!host) return;
    host.remove();
    preservedMedia.push({ host, identity, media, occurrence });
  });
  return preservedMedia;
}

function restorePlayingMedia(root, preservedMedia) {
  if (!preservedMedia.length) return;
  const candidatesByIdentity = new Map();
  root.querySelectorAll("audio[data-asset-id], video[data-asset-id]").forEach((media) => {
    const identity = mediaRenderIdentity(media);
    const candidates = candidatesByIdentity.get(identity) || [];
    candidates.push(media);
    candidatesByIdentity.set(identity, candidates);
  });

  preservedMedia.forEach((preserved) => {
    const replacement = candidatesByIdentity.get(preserved.identity)?.[preserved.occurrence];
    const replacementHost = replacement?.tagName === "AUDIO"
      ? replacement.closest("[data-audio-visualizer]")
      : replacement;
    if (replacementHost) {
      replacementHost.replaceWith(preserved.host);
      return;
    }
    if (preserved.media.tagName === "AUDIO") {
      const binding = audioVisualizerBindings.get(preserved.media);
      if (binding) disposeAudioVisualizerBinding(binding);
      else preserved.media.pause();
    } else {
      preserved.media.pause();
    }
  });
}

function mediaRenderIdentity(media) {
  const region = media.closest(".preview-panel")
    ? "preview"
    : media.closest(".inspector-panel")
      ? "inspector"
      : "other";
  return [
    media.tagName,
    media.dataset.assetId,
    region,
    media.controls ? "controls" : "no-controls",
    media.muted ? "muted" : "audible",
  ].join(":");
}

function disposeAudioVisualizers(preservedAudio = new Set()) {
  Array.from(audioVisualizerBindings.values()).forEach((binding) => {
    if (!preservedAudio.has(binding.audio)) disposeAudioVisualizerBinding(binding);
  });
}

function disposeAudioVisualizerBinding(binding) {
  cancelAnimationFrame(binding.frameID);
  binding.audio.pause();
  binding.audio.removeEventListener("play", binding.onPlay);
  binding.audio.removeEventListener("pause", binding.onPause);
  binding.audio.removeEventListener("ended", binding.onPause);
  binding.source?.disconnect();
  binding.analyser?.disconnect();
  audioVisualizerBindings.delete(binding.audio);
}

function setupAudioVisualizers(root) {
  root.querySelectorAll("[data-audio-visualizer]").forEach((container) => {
    const audio = container.querySelector("[data-audio-visualizer-audio]");
    const canvas = container.querySelector("[data-audio-visualizer-canvas]");
    if (!audio || !canvas) return;
    if (audioVisualizerBindings.has(audio)) return;

    const binding = {
      audio,
      canvas,
      source: null,
      analyser: null,
      frequencyData: null,
      waveformData: null,
      frameID: 0,
      onPlay: null,
      onPause: null,
    };
    binding.onPlay = () => {
      connectAudioVisualizer(binding);
      const resumePromise = audioVisualizerContext?.resume?.();
      resumePromise?.catch?.(() => {});
      drawAudioVisualizer(binding);
    };
    binding.onPause = () => {
      cancelAnimationFrame(binding.frameID);
      binding.frameID = 0;
      drawAudioVisualizer(binding);
    };
    audio.addEventListener("play", binding.onPlay);
    audio.addEventListener("pause", binding.onPause);
    audio.addEventListener("ended", binding.onPause);
    audioVisualizerBindings.set(audio, binding);
    drawAudioVisualizer(binding);
  });
}

function connectAudioVisualizer(binding) {
  if (binding.source || binding.canvas.dataset.visualizerUnavailable === "true") return;
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) {
    binding.canvas.dataset.visualizerUnavailable = "true";
    return;
  }
  try {
    audioVisualizerContext ||= new AudioContext();
    binding.source = audioVisualizerContext.createMediaElementSource(binding.audio);
    binding.analyser = audioVisualizerContext.createAnalyser();
    binding.analyser.fftSize = AUDIO_FFT_SIZE;
    binding.analyser.smoothingTimeConstant = 0.82;
    binding.frequencyData = new Uint8Array(binding.analyser.frequencyBinCount);
    binding.waveformData = new Uint8Array(binding.analyser.fftSize);
    binding.source.connect(binding.analyser);
    binding.analyser.connect(audioVisualizerContext.destination);
  } catch (error) {
    const fallbackSource = binding.source;
    fallbackSource?.disconnect();
    binding.analyser?.disconnect();
    binding.source = null;
    binding.analyser = null;
    binding.frequencyData = null;
    binding.waveformData = null;
    try {
      fallbackSource?.connect(audioVisualizerContext.destination);
    } catch {
      // If the source itself could not be created, the native audio element remains untouched.
    }
    console.debug("Audio visualizer unavailable; native audio playback remains enabled.", error);
    binding.canvas.dataset.visualizerUnavailable = "true";
  }
}

function drawAudioVisualizer(binding) {
  const context = binding.canvas.getContext("2d");
  if (!context) return;
  const displayWidth = Math.max(320, binding.canvas.clientWidth || 0);
  const displayHeight = Math.max(180, binding.canvas.clientHeight || 0);
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.floor(displayWidth * pixelRatio);
  const height = Math.floor(displayHeight * pixelRatio);
  if (binding.canvas.width !== width || binding.canvas.height !== height) {
    binding.canvas.width = width;
    binding.canvas.height = height;
  }
  context.clearRect(0, 0, width, height);

  const idle = !binding.analyser || binding.audio.paused || binding.audio.ended;
  const frequencyData = binding.frequencyData;
  const waveformData = binding.waveformData;
  if (idle) {
    frequencyData?.fill(0);
    waveformData?.fill(128);
  } else {
    binding.analyser.getByteFrequencyData(frequencyData);
    binding.analyser.getByteTimeDomainData(waveformData);
  }

  const barCount = 72;
  const gap = Math.max(2 * pixelRatio, width * 0.0025);
  const barWidth = Math.max(1, (width - gap * (barCount - 1)) / barCount);
  const amplitudes = calculateLogSpectrumAmplitudes(
    frequencyData,
    audioVisualizerContext?.sampleRate || 44_100,
    binding.analyser?.fftSize || AUDIO_FFT_SIZE,
    barCount,
    idle,
  );
  const barGradient = context.createLinearGradient(0, height, 0, 0);
  barGradient.addColorStop(0, "rgba(110, 231, 183, 0.48)");
  barGradient.addColorStop(0.55, "rgba(96, 165, 250, 0.86)");
  barGradient.addColorStop(1, "rgba(216, 180, 254, 0.96)");
  context.fillStyle = barGradient;
  for (let index = 0; index < barCount; index += 1) {
    const barHeight = Math.max(3 * pixelRatio, amplitudes[index] * height * 0.78);
    const x = index * (barWidth + gap);
    context.fillRect(x, height - barHeight, barWidth, barHeight);
  }

  const maximumWaveformPoints = Math.min(
    AUDIO_MAX_WAVEFORM_POINTS,
    Math.max(320, Math.floor(displayWidth * 1.5)),
  );
  const waveformStride = Math.max(1, Math.ceil((waveformData?.length || AUDIO_FFT_SIZE) / maximumWaveformPoints));
  const waveformPointCount = Math.ceil((waveformData?.length || AUDIO_FFT_SIZE) / waveformStride);
  context.beginPath();
  context.lineWidth = Math.max(2 * pixelRatio, 1.5);
  context.strokeStyle = idle ? "rgba(220, 240, 255, 0.26)" : "rgba(235, 248, 255, 0.92)";
  context.shadowColor = "rgba(120, 190, 255, 0.48)";
  context.shadowBlur = 10 * pixelRatio;
  for (let index = 0; index < waveformPointCount; index += 1) {
    const sampleIndex = Math.min(
      (waveformData?.length || AUDIO_FFT_SIZE) - 1,
      index * waveformStride,
    );
    const x = index * width / Math.max(1, waveformPointCount - 1);
    const sample = idle ? 128 : (waveformData?.[sampleIndex] ?? 128);
    const y = idle ? height * 0.46 : (sample / 255) * height * 0.54 + height * 0.18;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.stroke();
  context.shadowBlur = 0;

  if (binding.analyser && !binding.audio.paused && !binding.audio.ended) {
    binding.frameID = requestAnimationFrame(() => drawAudioVisualizer(binding));
  } else {
    binding.frameID = 0;
  }
}

function calculateLogSpectrumAmplitudes(frequencyData, sampleRate, fftSize, barCount, idle) {
  if (barCount <= 0) return [];
  if (idle) {
    return Array.from({ length: barCount }, (_, index) => 0.025 + 0.018 * Math.sin(index * 0.55) ** 2);
  }
  if (!frequencyData?.length) return Array.from({ length: barCount }, () => 0);

  const maximumFrequency = Math.max(
    AUDIO_MIN_FREQUENCY_HZ,
    Math.min(AUDIO_MAX_FREQUENCY_HZ, sampleRate / 2),
  );
  const frequencyRatio = maximumFrequency / AUDIO_MIN_FREQUENCY_HZ;
  const binFrequency = sampleRate / fftSize;
  const rawAmplitudes = Array.from({ length: barCount }, (_, index) => {
    // Sample the logarithmic band's centre and interpolate between adjacent
    // FFT bins. At the low end one log band can be narrower than a single
    // FFT bin; aggregating [startBin, endBin] would then reuse the same bins
    // for several bars and create visible flat plateaus.
    const centreFrequency = AUDIO_MIN_FREQUENCY_HZ
      * frequencyRatio ** ((index + 0.5) / barCount);
    const fractionalBin = Math.max(1, Math.min(
      frequencyData.length - 1,
      centreFrequency / binFrequency,
    ));
    const lowerBin = Math.floor(fractionalBin);
    const upperBin = Math.min(frequencyData.length - 1, lowerBin + 1);
    const interpolation = fractionalBin - lowerBin;
    const lowerValue = frequencyData[lowerBin] / 255;
    const upperValue = frequencyData[upperBin] / 255;
    return lowerValue + (upperValue - lowerValue) * interpolation;
  });
  const framePeak = Math.max(0.18, ...rawAmplitudes);
  const automaticGain = Math.min(1.6, 0.92 / framePeak);
  return rawAmplitudes.map((amplitude) => Math.min(1, (amplitude * automaticGain) ** 0.68));
}
