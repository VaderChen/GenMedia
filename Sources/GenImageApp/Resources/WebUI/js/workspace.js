import {
  actionLabel,
  escapeHTML,
  jobLabel,
  kindLabel,
  percent,
} from "./format.js";
import { t } from "./i18n.js";

const toolMeta = {
  textToImage: { titleKey: "cap.textToImage" },
  imageToText: { titleKey: "cap.imageToText" },
  imageToImage: { titleKey: "cap.imageToImage" },
  textToVideo: { titleKey: "cap.textToVideo" },
  imageToVideo: { titleKey: "cap.imageToVideo" },
  textToMusic: { titleKey: "cap.textToMusic" },
  upscale: { titleKey: "cap.upscale" },
};

const aspectRatios = [
  { label: "1:1", width: 1, height: 1 },
  { label: "2:3", width: 2, height: 3 },
  { label: "3:4", width: 3, height: 4 },
  { label: "9:16", width: 9, height: 16 },
  { label: "3:2", width: 3, height: 2 },
  { label: "4:3", width: 4, height: 3 },
  { label: "16:9", width: 16, height: 9 },
];

const creationTabsByGenerationType = {
  image: [
    { id: "prompt", labelKey: "workspace.prompt" },
    { id: "negative", labelKey: "workspace.negative" },
    { id: "imageOutput", labelKey: "workspace.imageOutput" },
  ],
  video: [
    { id: "prompt", labelKey: "workspace.prompt" },
    { id: "negative", labelKey: "workspace.negative" },
    { id: "videoOutput", labelKey: "workspace.videoOutput" },
  ],
  music: [
    { id: "prompt", labelKey: "workspace.prompt" },
    { id: "lyrics", labelKey: "workspace.lyrics" },
    { id: "musicOutput", labelKey: "workspace.musicOutput" },
  ],
};

const musicStyles = [
  ["pop", "workspace.musicStyle.pop"],
  ["rock", "workspace.musicStyle.rock"],
  ["hipHop", "workspace.musicStyle.hipHop"],
  ["rnb", "workspace.musicStyle.rnb"],
  ["electronic", "workspace.musicStyle.electronic"],
  ["jazz", "workspace.musicStyle.jazz"],
  ["classical", "workspace.musicStyle.classical"],
  ["folk", "workspace.musicStyle.folk"],
  ["country", "workspace.musicStyle.country"],
  ["blues", "workspace.musicStyle.blues"],
  ["reggae", "workspace.musicStyle.reggae"],
  ["metal", "workspace.musicStyle.metal"],
  ["ambient", "workspace.musicStyle.ambient"],
  ["cinematic", "workspace.musicStyle.cinematic"],
  ["anime", "workspace.musicStyle.anime"],
  ["lofi", "workspace.musicStyle.lofi"],
];

const MUSIC_DURATION_MIN_SECONDS = 5;
const MUSIC_DURATION_MAX_SECONDS = 300;

const jobTimingEstimators = new Map();
const ETA_SAMPLE_WINDOW_MS = 3 * 60 * 1_000;
const ETA_MIN_SAMPLE_SPAN_MS = 5_000;
const ETA_MIN_PROGRESS_DELTA = 0.001;
const ETA_STEADY_PROGRESS_START = 0.25;
const ETA_MIN_ESTIMATE_PROGRESS = 0.35;
const ETA_MIN_ESTIMATE_ELAPSED_MS = 15_000;
const ETA_MIN_ESTIMATE_DELTA = 0.05;
const ETA_MAX_DURATION_MS = 7 * 24 * 60 * 60 * 1_000;

export function renderWorkspace(state, ui) {
  return `
    <section class="workspace-shell">
      ${renderWorkspaceTabs(state, ui)}
      <section class="workspace-grid">
        ${renderPreviewPanel(state, ui)}
        ${renderCreationPanel(state, ui)}
        ${renderInspector(state, ui)}
      </section>
    </section>
  `;
}

function renderWorkspaceTabs(state, ui) {
  const tabs = ui.workspaceTabs || [];
  return `
    <div class="workspace-tabs" role="tablist" aria-label="${t("workspace.tabsLabel")}">
      <div class="workspace-tab-list">
        ${tabs
          .map((tab, index) => {
            const active = tab.id === ui.activeWorkspaceTabID;
            return `<div class="workspace-tab ${active ? "active" : ""}">
              <button
                class="workspace-tab-main"
                data-action="workspaceTab"
                data-tab-id="${escapeHTML(tab.id)}"
                role="tab"
                aria-selected="${active}"
                title="${t("workspace.renameTabHint")}"
              ><span>${escapeHTML(tab.name || t("workspace.tabName", { count: index + 1 }))}</span></button>
              <button
                class="workspace-tab-close"
                data-action="workspaceCloseTab"
                data-tab-id="${escapeHTML(tab.id)}"
                title="${t("workspace.closeTab")}"
                aria-label="${t("workspace.closeTab")}"
                ${tabs.length <= 1 || isInferenceBusy(state) ? "disabled" : ""}
              >×</button>
            </div>`;
          })
          .join("")}
      </div>
      <button class="workspace-add-tab" data-action="workspaceAddTab" title="${t("workspace.addTab")}" aria-label="${t("workspace.addTab")}">＋</button>
    </div>
  `;
}

export function renderQuickTools(state) {
  return ["imageToText", "textToImage", "imageToImage", "textToVideo", "imageToVideo", "textToMusic", "upscale"]
    .filter((capability) => hasActiveProfile(state, capability))
    .map((capability) => renderQuickTool(state, capability))
    .join("");
}

export function renderCreationPanel(state, ui) {
  const recipe = state.recipe;
  const videoOutputSettings = state.videoOutputSettings || {
    width: 1280,
    height: 720,
    steps: 8,
    outputCount: 1,
    frameCount: 97,
    frameRate: 24,
    seed: "42",
  };
  const musicOutputSettings = state.musicOutputSettings || {
    prompt: "",
    lyrics: "",
    style: "pop",
    durationSeconds: 10,
    steps: 30,
    seed: "42",
    format: "mp3",
  };
  const collapsed = ui.creationCollapsed;
  const descriptionBusy = isImageDescriptionBusy(state);
  const inferenceBusy = isInferenceBusy(state);
  const inferenceDisabledAttribute = inferenceBusy ? "disabled aria-busy=\"true\"" : "";
  const generationType = creationTabsByGenerationType[ui.generationType] ? ui.generationType : "image";
  const creationTabs = creationTabsByGenerationType[generationType];
  const promptTab = creationTabs.some((tab) => tab.id === ui.promptTab)
    ? ui.promptTab
    : creationTabs[0].id;
  const toggleLabel = collapsed ? t("workspace.expandCreation") : t("workspace.collapseCreation");
  const showGenerateText = hasActiveProfile(state, "imageToText");
  const showGenerateImage = generationType === "image";
  const showGenerateVideo = generationType === "video";
  const showGenerateMusic = generationType === "music";
  return `
    <aside class="creation-panel ${collapsed ? "collapsed" : ""}">
      <div class="creation-header">
        <button
          class="ghost-button compact creation-toggle"
          data-action="toggleCreationPanel"
          aria-expanded="${collapsed ? "false" : "true"}"
          title="${toggleLabel}"
        ><span aria-hidden="true">${collapsed ? "▴" : "▾"}</span>${toggleLabel}</button>
        <div class="toolbar-spacer"></div>
        ${
          showGenerateText
            ? `<button
                class="secondary-button creation-generate-button creation-generate-text"
                data-action="describe"
                ${inferenceDisabledAttribute}
              >⌕ ${t("workspace.generateText")}</button>`
            : ""
        }
        ${
          showGenerateImage
            ? `<button
                class="primary-button creation-generate-button creation-generate-image"
                data-action="generate"
                ${inferenceDisabledAttribute}
              >✦ ${t("workspace.generate")}</button>`
            : ""
        }
        ${
          showGenerateVideo
            ? `<button
                class="secondary-button creation-generate-button creation-generate-video"
                data-action="generateVideo"
                ${inferenceDisabledAttribute}
              >▶ ${t("workspace.generateVideo")}</button>`
            : ""
        }
        ${
          showGenerateMusic
            ? `<button
                class="secondary-button creation-generate-button creation-generate-music"
                data-action="generateMusic"
                ${inferenceDisabledAttribute}
              >♫ ${t("workspace.generateMusic")}</button>`
            : ""
        }
      </div>

      ${
        collapsed
          ? ""
          : `
      <div class="creation-scroll" data-scroll-id="creation">
        <div class="prompt-tab-header">
          <div class="creation-tab-controls">
            ${renderGenerationTypeSelect(generationType)}
            <div
              class="prompt-tabs"
              role="tablist"
              aria-label="${t("workspace.creationTabsLabel")}"
              style="--creation-tab-count: ${creationTabs.length}"
            >
              ${creationTabs.map((tab) => promptTabButton(tab.id, t(tab.labelKey), promptTab)).join("")}
            </div>
            ${promptTab === "imageOutput" ? renderInlineAspectRatios(recipe, "image") : ""}
            ${promptTab === "videoOutput" ? renderInlineAspectRatios(videoOutputSettings, "video") : ""}
          </div>
          <button class="ghost-button compact" data-action="applyProfileDefaults">${t("workspace.applyDefaults")}</button>
        </div>
        <div class="prompt-tab-panel" role="tabpanel">
          ${renderCreationTab(state, generationType, promptTab, descriptionBusy)}
        </div>
      </div>
      `
      }
    </aside>
  `;
}

function activeProfile(state, capability) {
  const activeProfileID = state.activeProfileIDs?.[capability];
  if (!activeProfileID) return null;
  const disabledProfileIDs = new Set(state.disabledProfileIDs || []);
  return state.profiles.find(
    (profile) =>
      profile.id === activeProfileID
      && profile.capability === capability
      && !disabledProfileIDs.has(profile.id),
  ) || null;
}

function hasActiveProfile(state, capability) {
  return activeProfile(state, capability) !== null;
}

function renderGenerationTypeSelect(activeGenerationType) {
  const options = [
    ["image", "workspace.imageGeneration"],
    ["video", "workspace.videoGeneration"],
    ["music", "workspace.musicGeneration"],
  ];
  return `<select
    class="field generation-type-select"
    data-ui-field="generationType"
    aria-label="${t("workspace.generationType")}"
    title="${t("workspace.generationType")}"
  >${options.map(([value, labelKey]) =>
    `<option value="${value}" ${value === activeGenerationType ? "selected" : ""}>${t(labelKey)}</option>`,
  ).join("")}</select>`;
}

function promptTabButton(tab, label, activeTab) {
  const active = tab === activeTab;
  return `<button
    class="prompt-tab ${active ? "active" : ""}"
    data-action="promptTab"
    data-tab="${tab}"
    role="tab"
    aria-selected="${active}"
  >${label}</button>`;
}

function renderCreationTab(state, generationType, promptTab, descriptionBusy) {
  if (promptTab === "imageOutput") return renderOutputSettings(state.recipe, "image");
  if (promptTab === "videoOutput") {
    return renderOutputSettings(state.videoOutputSettings, "video");
  }
  if (promptTab === "lyrics") return renderLyricsEditor(state.musicOutputSettings, descriptionBusy);
  if (promptTab === "musicOutput") return renderMusicOutputSettings(state);
  if (generationType === "music") return renderMusicPromptEditor(state.musicOutputSettings);
  return renderPromptEditor(state.recipe, promptTab, descriptionBusy);
}

function renderMusicPromptEditor(settings) {
  return `<textarea
    id="music-prompt"
    class="prompt-area"
    data-music-field="prompt"
    data-preserve-focus="music-prompt"
    placeholder="${t("workspace.musicPromptPlaceholder")}"
  >${escapeHTML(settings.prompt)}</textarea>`;
}

function renderLyricsEditor(settings, disabled) {
  const disabledAttribute = disabled ? "disabled aria-busy=\"true\"" : "";
  return `<textarea
    id="music-lyrics"
    class="prompt-area"
    data-music-field="lyrics"
    data-preserve-focus="music-lyrics"
    placeholder="${t("workspace.lyricsPlaceholder")}"
    ${disabledAttribute}
  >${escapeHTML(settings.lyrics)}</textarea>`;
}

function renderMusicOutputSettings(state) {
  const settings = state.musicOutputSettings;
  const musicConfiguration = activeProfile(state, "textToMusic")?.music || {};
  const minimumDurationSeconds = Number.isFinite(musicConfiguration.minimumDurationSeconds)
    ? musicConfiguration.minimumDurationSeconds
    : MUSIC_DURATION_MIN_SECONDS;
  const maximumDurationSeconds = Number.isFinite(musicConfiguration.maximumDurationSeconds)
    ? musicConfiguration.maximumDurationSeconds
    : MUSIC_DURATION_MAX_SECONDS;
  const maximumDuration = musicConfiguration.durationSemantics === "maximum";
  return `<div class="music-output-panel">
    <section class="output-setting-group music-output-setting-card">
      <div class="output-setting-heading">
        <strong>${t("workspace.musicParameters")}</strong>
      </div>
      <div class="music-output-fields">
        <div class="field-group">
          <label for="music-style">${t("workspace.musicStyle")}</label>
          <select id="music-style" class="field" data-music-field="style" data-preserve-focus="music-style">
            ${musicStyles.map(([style, labelKey]) =>
              `<option value="${style}" ${settings.style === style ? "selected" : ""}>${t(labelKey)}</option>`,
            ).join("")}
          </select>
        </div>
        ${musicNumberField(
          t(maximumDuration ? "workspace.maximumDuration" : "workspace.duration"),
          "durationSeconds",
          settings.durationSeconds,
          minimumDurationSeconds,
          maximumDurationSeconds,
        )}
        ${musicNumberField(t("workspace.steps"), "steps", settings.steps, 1, 100)}
        <div class="field-group">
          <label for="music-format">${t("workspace.audioFormat")}</label>
          <select id="music-format" class="field" data-music-field="format" data-preserve-focus="music-format">
            ${["mp3", "m4a", "aac", "flac"].map((format) =>
              `<option value="${format}" ${settings.format === format ? "selected" : ""}>${format.toUpperCase()}</option>`,
            ).join("")}
          </select>
        </div>
        <div class="field-group seed-field">
          <label for="music-seed">${t("workspace.seed")}</label>
          <div class="seed-input-row">
            <input
              id="music-seed"
              class="field"
              type="text"
              inputmode="numeric"
              data-music-field="seed"
              data-preserve-focus="music-seed"
              value="${escapeHTML(settings.seed)}"
            />
            <button
              class="icon-button compact seed-random-button"
              data-action="randomizeSeed"
              data-output-kind="music"
              title="${t("workspace.randomSeed")}"
              aria-label="${t("workspace.randomSeed")}"
            >↻</button>
          </div>
        </div>
      </div>
      <p class="section-note music-output-note">${t(
        maximumDuration ? "workspace.musicMaximumDurationNote" : "workspace.musicTargetDurationNote",
      )}</p>
    </section>
  </div>`;
}

function musicNumberField(label, field, value, min, max) {
  return `<div class="field-group">
    <label for="music-${field}">${label}</label>
    <input
      id="music-${field}"
      class="field"
      type="number"
      min="${min}"
      max="${max}"
      step="1"
      data-music-field="${field}"
      data-preserve-focus="music-${field}"
      value="${value}"
    />
  </div>`;
}

function renderInlineAspectRatios(settings, outputKind) {
  const activeRatio = closestAspectRatio(settings.width, settings.height);
  const label = outputKind === "video" ? t("workspace.videoAspectRatio") : t("workspace.aspectRatio");
  return `<div class="inline-aspect-ratios" aria-label="${label}">
    ${aspectRatios.map((ratio) => aspectRatioChip(ratio, activeRatio, outputKind)).join("")}
  </div>`;
}

function renderPromptEditor(recipe, promptTab, disabled) {
  const disabledAttribute = disabled ? "disabled aria-busy=\"true\"" : "";
  if (promptTab === "negative") {
    return `<textarea
      id="recipe-negative"
      class="prompt-area"
      data-recipe-field="negativePrompt"
      data-preserve-focus="recipe-negative"
      placeholder="${t("workspace.negativePlaceholder")}"
      ${disabledAttribute}
    >${escapeHTML(recipe.negativePrompt)}</textarea>`;
  }

  return `<textarea
    id="recipe-prompt"
    class="prompt-area"
    data-recipe-field="prompt"
    data-preserve-focus="recipe-prompt"
    placeholder="${t("workspace.promptPlaceholder")}"
    ${disabledAttribute}
  >${escapeHTML(recipe.prompt)}</textarea>`;
}

function renderOutputSettings(settings, outputKind) {
  const isVideo = outputKind === "video";
  const ratio = closestAspectRatio(settings.width, settings.height);
  const widthBounds = resolutionBounds("width", ratio);
  const heightBounds = resolutionBounds("height", ratio);
  const duration = isVideo
    ? Number(settings.frameCount) / Math.max(1, Number(settings.frameRate))
    : 0;
  return `<div class="output-tab-panel">
    <div class="output-detail-grid">
      <section class="output-setting-group resolution-setting-card">
        <div class="output-setting-heading">
          <strong>${t("workspace.resolution")}</strong>
          <span data-resolution-summary data-output-kind="${outputKind}">${settings.width} × ${settings.height} px</span>
        </div>
        <div class="resolution-grid">
          ${resolutionSlider(t("workspace.width"), "width", settings.width, widthBounds, ratio, outputKind)}
          ${resolutionSlider(t("workspace.height"), "height", settings.height, heightBounds, ratio, outputKind)}
        </div>
      </section>

      <section class="output-setting-group output-parameter-card ${isVideo ? "video-output-parameter-card" : ""}">
        <div class="output-setting-heading">
          <strong>${isVideo ? t("workspace.videoParameters") : t("workspace.parameters")}</strong>
          ${
            isVideo
              ? `<span>${duration.toFixed(1)} ${t("workspace.seconds")} · ${settings.frameRate} FPS</span>`
              : ""
          }
        </div>
        <div class="output-parameter-list">
          <div class="output-parameter-primary-row">
            ${numberField(t("workspace.steps"), "steps", settings.steps, 1, 100, outputKind)}
            ${numberField(t("workspace.count"), "outputCount", settings.outputCount, 1, 8, outputKind)}
          </div>
          ${
            isVideo
              ? `<div class="output-parameter-primary-row">
                  ${numberField(t("workspace.frameCount"), "frameCount", settings.frameCount, 1, 512, outputKind, 8)}
                  ${numberField(t("workspace.frameRate"), "frameRate", settings.frameRate, 1, 120, outputKind)}
                </div>`
              : ""
          }
          <div class="field-group seed-field">
            <label for="${outputKind}-seed">${t("workspace.seed")}</label>
            <div class="seed-input-row">
              <input
                id="${outputKind}-seed"
                class="field"
                type="text"
                inputmode="numeric"
                ${settingsFieldAttribute(outputKind, "seed")}
                data-preserve-focus="${outputKind}-seed"
                value="${escapeHTML(settings.seed)}"
              />
              <button
                class="icon-button compact seed-random-button"
                data-action="randomizeSeed"
                data-output-kind="${outputKind}"
                title="${t("workspace.randomSeed")}"
                aria-label="${t("workspace.randomSeed")}"
              ><svg class="seed-random-icon" viewBox="0 0 24 24" aria-hidden="true">
                <rect x="4" y="4" width="16" height="16" rx="3"></rect>
                <circle cx="8.5" cy="8.5" r="1"></circle>
                <circle cx="15.5" cy="8.5" r="1"></circle>
                <circle cx="12" cy="12" r="1"></circle>
                <circle cx="8.5" cy="15.5" r="1"></circle>
                <circle cx="15.5" cy="15.5" r="1"></circle>
              </svg></button>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>`;
}

function closestAspectRatio(width, height) {
  const value = width / Math.max(1, height);
  return aspectRatios.reduce((best, ratio) =>
    Math.abs(ratio.width / ratio.height - value) < Math.abs(best.width / best.height - value) ? ratio : best,
  );
}

function aspectRatioChip(ratio, activeRatio, outputKind) {
  const active = ratio.label === activeRatio.label;
  return `<button
    class="chip aspect-ratio-chip ${active ? "active" : ""}"
    data-action="aspectRatio"
    data-ratio-width="${ratio.width}"
    data-ratio-height="${ratio.height}"
    data-output-kind="${outputKind}"
  >${ratio.label}</button>`;
}

function resolutionBounds(dimension, ratio) {
  const ratioValue = dimension === "width"
    ? ratio.width / ratio.height
    : ratio.height / ratio.width;
  return {
    min: Math.max(64, Math.ceil((64 * ratioValue) / 16) * 16),
    max: Math.min(4096, Math.floor((4096 * ratioValue) / 16) * 16),
  };
}

function resolutionSlider(label, field, value, bounds, ratio, outputKind) {
  return `<div class="resolution-control">
    <div class="resolution-label">
      <label for="${outputKind}-${field}">${label}</label>
      <output data-dimension-value="${field}" data-output-kind="${outputKind}">${value} px</output>
    </div>
    <input
      id="${outputKind}-${field}"
      class="resolution-slider"
      type="range"
      min="${bounds.min}"
      max="${bounds.max}"
      step="16"
      value="${value}"
      data-dimension-field="${field}"
      data-output-kind="${outputKind}"
      data-ratio-width="${ratio.width}"
      data-ratio-height="${ratio.height}"
    />
    <div class="resolution-range"><span>${bounds.min} px</span><span>${bounds.max} px</span></div>
  </div>`;
}

function renderQuickTool(state, capability) {
  const meta = toolMeta[capability];
  const disabledProfileIDs = new Set(state.disabledProfileIDs || []);
  const profiles = state.profiles.filter(
    (profile) => profile.capability === capability && !disabledProfileIDs.has(profile.id),
  );
  const activeID = state.activeProfileIDs[capability];
  return `
    <div class="tool-card">
      <div class="tool-copy">
        <strong>${t(meta.titleKey)}</strong>
        <select
          class="profile-select"
          data-profile-capability="${capability}"
          aria-label="${t(meta.titleKey)} Profile"
          ${profiles.length ? "" : "disabled"}
        >
          ${profiles.length ? "" : `<option value="">${t("profile.noProfile")}</option>`}
          ${profiles.length && !activeID ? `<option value="" selected disabled>${t("profile.select")}</option>` : ""}
          ${profiles
            .map(
              (profile) => `<option value="${profile.id}" ${profile.id === activeID ? "selected" : ""}>
                ${escapeHTML(profile.name)}
              </option>`,
            )
            .join("")}
          </select>
          ${capability === "textToImage" ? renderLoRAControl(state) : ""}
      </div>
    </div>
  `;
}

function renderLoRAControl(state) {
  const loras = Array.isArray(state.loras)
    ? state.loras.filter((lora) =>
      !Array.isArray(lora.compatibleCapabilities)
        || lora.compatibleCapabilities.includes("textToImage"))
    : [];
  const selectedID = state.recipe.loraID || "";
  const scale = Number.isFinite(Number(state.recipe.loraScale))
    ? Math.min(1, Math.max(0, Number(state.recipe.loraScale)))
    : 1;
  return `<div class="lora-control">
    <label for="recipe-lora">${t("lora.label")}</label>
    <select
      id="recipe-lora"
      class="profile-select lora-select"
      data-lora-select
      aria-label="${t("lora.label")}"
      ${loras.length ? "" : "disabled"}
    >
      <option value="">${loras.length ? t("lora.none") : t("lora.noModels")}</option>
      ${loras
        .map(
          (lora) => `<option value="${escapeHTML(lora.id)}" ${lora.id === selectedID ? "selected" : ""}>
            ${escapeHTML(lora.displayName)}
          </option>`,
        )
        .join("")}
    </select>
    ${
      selectedID
        ? `<div class="lora-scale-row">
            <span>${t("lora.scale")}</span>
            <input type="range" min="0" max="1" step="0.05" value="${scale}" data-lora-scale />
            <output data-lora-scale-value>${Math.round(scale * 100)}%</output>
          </div>`
        : ""
    }
  </div>`;
}

function renderPreviewPanel(state, ui) {
  const fixedKindAsset = ui.previewMode === "single" ? selectedAsset(state) : null;
  const zoomDisabled = ui.generationType === "music"
    || selectedAsset(state)?.kind === "generatedAudio";
  const previewUI = zoomDisabled ? { ...ui, zoom: 1 } : ui;
  return `
    <main class="preview-panel">
      <div class="preview-toolbar">
        ${previewModeButton("grid", `▦ ${t("preview.grid")}`, ui)}
        ${previewModeButton("single", `▭ ${t("preview.single")}`, ui)}
        ${previewModeButton("compare", `◫ ${t("preview.compare")}`, ui)}
        <div class="toolbar-spacer"></div>
        ${
          ui.previewMode === "grid"
            ? ""
            : `
              <button class="icon-button compact" data-action="zoomOut" ${zoomDisabled ? "disabled" : ""}>−</button>
              <input class="zoom-range" type="range" min="0.25" max="2.5" step="0.25" value="${previewUI.zoom}" data-ui-field="zoom" ${zoomDisabled ? "disabled" : ""} />
              <span class="section-note">${Math.round(previewUI.zoom * 100)}%</span>
              <button class="icon-button compact" data-action="zoomIn" ${zoomDisabled ? "disabled" : ""}>＋</button>
              <button
                class="icon-button compact"
                data-action="fitPreview"
                title="${t("preview.fit")}"
                aria-label="${t("preview.fit")}"
                ${zoomDisabled ? "disabled" : ""}
              >
                <svg class="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M9 4H4v5M15 4h5v5M20 15v5h-5M9 20H4v-5" />
                </svg>
              </button>
              <button
                class="icon-button compact"
                data-action="revealOutputDirectory"
                title="${t("preview.openOutputDirectory")}"
                aria-label="${t("preview.openOutputDirectory")}"
              >
                <svg class="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M3.5 6.5h6l2 2h9v9a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z" />
                </svg>
              </button>
            `
        }
      </div>
      <div class="preview-stage" data-scroll-id="preview" data-pan-enabled="${ui.previewMode !== "grid"}" data-zoom-enabled="${!zoomDisabled}">
        ${renderPreviewContent(state, previewUI)}
      </div>
      ${fixedKindAsset ? `<span class="asset-kind preview-fixed-kind">${kindLabel(fixedKindAsset.kind)}</span>` : ""}
      ${renderFilmstrip(state)}
    </main>
  `;
}

function renderPreviewContent(state, ui) {
  if (!state.assets.length) return emptyState(t("preview.noImages"), t("preview.noImagesDetail"), "▧");
  if (ui.previewMode === "single") return renderSingle(state, ui);
  if (ui.previewMode === "compare") return renderCompare(state, ui);
  return `<div class="asset-grid">${state.assets.map((asset) => renderAssetCard(asset, state)).join("")}</div>`;
}

function renderAssetCard(asset, state) {
  const selection = assetSelection(asset, state);
  return `
    <button
      class="asset-card ${selection.classes}"
      data-action="selectAsset"
      data-asset-id="${asset.id}"
      aria-pressed="${selection.selected}"
    >
      ${renderArtwork(asset)}
      ${selection.badge}
      <div class="asset-caption">
        <div class="asset-caption-copy">
          <strong>${escapeHTML(asset.title)}</strong>
          <span>${assetSummary(asset)}</span>
        </div>
        <span class="muted">›</span>
      </div>
    </button>
  `;
}

function renderSingle(state, ui) {
  const asset = selectedAsset(state);
  if (!asset) return emptyState(t("preview.noSelection"), t("preview.selectFilmstrip"), "▧");
  const size = Math.round(ui.zoom * 100);
  return `
    <div class="single-stage ${ui.zoom > 1 ? "zoomed" : ""}">
      <div class="single-asset" style="width:${size}%;height:${size}%">
        ${renderArtwork(asset, true, false)}
      </div>
    </div>
  `;
}

function renderCompare(state, ui) {
  const primary = selectedAsset(state);
  const secondary = state.assets.find((asset) => asset.id === state.comparisonAssetID);
  if (!primary || !secondary) return emptyState(t("preview.needTwo"), t("preview.needTwoDetail"), "◫");
  const width = Math.round(385 * ui.zoom);
  return `
    <div class="compare-stage">
      ${comparisonPane(t("preview.current"), primary, width)}
      ${comparisonPane(t("preview.comparison"), secondary, width)}
    </div>
  `;
}

function comparisonPane(title, asset, width) {
  return `
    <div class="compare-pane" style="width:${width}px">
      <h3><span>${title}</span><span class="muted">${assetSummary(asset)}</span></h3>
      ${renderArtwork(asset, true)}
    </div>
  `;
}

function renderArtwork(asset, controls = false, showKind = true) {
  const hasMedia = Boolean(asset.previewURL);
  const isVideo = asset.kind === "generatedVideo";
  const isAudio = asset.kind === "generatedAudio";
  const media = hasMedia
      ? isAudio
        ? controls
        ? `<div class="audio-player" data-audio-visualizer data-asset-id="${escapeHTML(asset.id)}">
            <canvas class="audio-visualizer" data-audio-visualizer-canvas aria-hidden="true"></canvas>
            <audio
              src="${escapeHTML(asset.previewURL)}"
              data-asset-id="${escapeHTML(asset.id)}"
              data-audio-visualizer-audio
              aria-label="${escapeHTML(asset.title)}"
              preload="metadata"
              controls
            ></audio>
          </div>`
        : `<span class="audio-placeholder" data-asset-id="${escapeHTML(asset.id)}" aria-label="${escapeHTML(asset.title)}">♫</span>`
      : isVideo
      ? `<video
          src="${escapeHTML(asset.previewURL)}"
          data-asset-id="${escapeHTML(asset.id)}"
          aria-label="${escapeHTML(asset.title)}"
          preload="metadata"
          playsinline
          ${controls ? "controls" : "muted"}
        ></video>`
      : `<img src="${escapeHTML(asset.previewURL)}" data-asset-id="${escapeHTML(asset.id)}" alt="${escapeHTML(asset.title)}" draggable="false" />`
    : `<span class="placeholder-icon">${asset.kind === "upscaled" ? "↗" : asset.kind === "generated" ? "✦" : asset.kind === "generatedAudio" ? "♫" : "▧"}</span>`;
  return `
    <div class="asset-artwork ${asset.kind} ${hasMedia ? "has-media" : ""}">
      ${media}
      ${showKind ? `<span class="asset-kind">${kindLabel(asset.kind)}</span>` : ""}
    </div>
  `;
}

function renderFilmstrip(state) {
  return `<div class="filmstrip" data-scroll-id="filmstrip">
    ${state.assets
      .map(
        (asset) => {
          const selection = assetSelection(asset, state);
          return `
          <div class="film-thumb-shell">
            <button
              class="film-thumb ${selection.classes}"
              data-action="selectAsset"
              data-asset-id="${asset.id}"
              title="${escapeHTML(asset.title)} · ⌘/Ctrl ${t("preview.multiSelectHint")}"
              aria-pressed="${selection.selected}"
            >${renderArtwork(asset)}${selection.badge}</button>
            <button
              class="film-thumb-remove"
              data-action="removeAsset"
              data-asset-id="${asset.id}"
              title="${t("preview.removeImage")}"
              aria-label="${t("preview.removeImage")}"
            >×</button>
          </div>
        `;
        },
      )
      .join("")}
  </div>`;
}

function assetSelection(asset, state) {
  const anchorIndex = (state.selectedAssetIDs || []).indexOf(asset.id);
  const primary = asset.id === state.selectedAssetID;
  const selected = primary || anchorIndex >= 0;
  const classes = [
    selected ? "selected" : "",
    primary ? "primary-selection" : "",
    anchorIndex >= 0 ? "video-anchor" : "",
  ].filter(Boolean).join(" ");
  const badge = anchorIndex >= 0
    ? `<span class="video-anchor-order" aria-label="${t("preview.anchorNumber", { count: anchorIndex + 1 })}">${anchorIndex + 1}</span>`
    : "";
  return { badge, classes, selected };
}

export function isInferenceBusy(state) {
  return state.jobs.some((job) => ["queued", "running", "cancelling"].includes(job.state));
}

function isImageDescriptionBusy(state) {
  return state.jobs.some((job) =>
    job.action === "describe" && ["queued", "running", "cancelling"].includes(job.state),
  );
}

function renderInspector(state, ui) {
  const activeTab = ui.inspectorTab === "jobs" ? "jobs" : "info";
  return `
    <aside class="inspector-panel">
      <div class="inspector-tabs" role="tablist" aria-label="${t("inspector.panelLabel")}">
        ${inspectorTabButton("info", t("inspector.infoTab"), activeTab)}
        ${inspectorTabButton("jobs", t("jobs.title"), activeTab)}
      </div>
      <div class="inspector-content" role="tabpanel">
        ${activeTab === "jobs" ? renderJobsPanel(state) : renderAssetInspector(state)}
      </div>
    </aside>
  `;
}

function inspectorTabButton(tab, label, activeTab) {
  const active = tab === activeTab;
  return `
    <button
      class="inspector-tab ${active ? "active" : ""}"
      data-action="inspectorTab"
      data-tab="${tab}"
      role="tab"
      aria-selected="${active}"
    >${label}</button>
  `;
}

function renderAssetInspector(state) {
  const asset = selectedAsset(state);
  if (!asset) {
    return `<div class="empty-state"><span class="empty-icon">ⓘ</span><strong>${t("inspector.noSelection")}</strong></div>`;
  }
  const operation = [...state.operations].reverse().find((item) => item.outputAssetIDs.includes(asset.id));
  const lineage = buildLineage(state.assets, asset);
  const isVideo = asset.kind === "generatedVideo";
  const isAudio = asset.kind === "generatedAudio";
  const canEdit = !isVideo && !isAudio && hasActiveProfile(state, "imageToImage");
  const inferenceDisabledAttribute = isInferenceBusy(state)
    ? "disabled aria-busy=\"true\""
    : "";
  return `
    <div class="inspector-scroll" data-scroll-id="inspector-info">
      <div class="section-heading"><h2>${escapeHTML(asset.title)}</h2></div>
      <div class="inspector-preview">${renderArtwork(asset, true)}</div>
      ${
        isVideo || isAudio
          ? ""
          : `<div class="inspector-actions">
              <button class="secondary-button compact" data-action="describe" ${inferenceDisabledAttribute}>⌕ ${t("inspector.caption")}</button>
              ${canEdit ? `<button class="secondary-button compact" data-action="imageToImage" ${inferenceDisabledAttribute}>▧ ${t("cap.imageToImage")}</button>` : ""}
              <button class="secondary-button compact" data-action="upscale" ${inferenceDisabledAttribute}>↗ ${t("inspector.upscale")}</button>
            </div>`
      }
      <div class="inspector-group">
        <h3>${t(isAudio ? "inspector.audioInfo" : isVideo ? "inspector.videoInfo" : "inspector.imageInfo")}</h3>
        ${isAudio ? "" : detailRow(t("inspector.dimensions"), `${asset.pixelWidth} × ${asset.pixelHeight}`)}
        ${isAudio ? detailRow(t("inspector.duration"), formatDuration((asset.mediaDurationSeconds || 0) * 1_000)) : ""}
        ${isAudio ? detailRow(t("inspector.audioFormat"), String(asset.audioFormat || "").toUpperCase()) : ""}
        ${isAudio ? detailRow(t("inspector.sampleRate"), `${Number(asset.sampleRate || 0).toLocaleString()} Hz`) : ""}
        ${isAudio ? detailRow(t("inspector.channels"), String(asset.channelCount || "–")) : ""}
        ${detailRow(t("inspector.kind"), kindLabel(asset.kind))}
      </div>
      ${
        operation?.profileName
          ? `<div class="inspector-group">
              <h3>${t("inspector.profileSnapshot")}</h3>
              ${detailRow(t("inspector.name"), operation.profileName)}
              ${detailRow(t("inspector.profileRevision"), `r${operation.profileRevision || 1}`)}
            </div>`
          : ""
      }
      <div class="inspector-group">
        <h3>${t("inspector.lineage")}</h3>
        <div class="lineage">
          ${lineage.map((item) => `<div class="lineage-item"><span>›</span>${escapeHTML(item.title)}</div>`).join("")}
        </div>
      </div>
      <div class="inspector-group">
        <h3>${t("inspector.currentPrompt")}</h3>
        <div class="section-note">${escapeHTML(state.recipe.prompt)}</div>
      </div>
    </div>
  `;
}

function renderJobsPanel(state) {
  const running = state.jobs.filter((job) => ["queued", "running", "cancelling"].includes(job.state)).length;
  return `
    <section class="inspector-jobs">
      <div class="job-header">
        <span class="section-note">${t("jobs.running", { count: running })}</span>
        <div class="toolbar-spacer"></div>
        <button class="ghost-button compact" data-action="clearJobs">${t("jobs.clear")}</button>
      </div>
      <div class="inspector-job-list" data-scroll-id="inspector-jobs">
        ${
          state.jobs.length
            ? state.jobs.map(renderJob).join("")
            : `<div class="empty-state"><span class="empty-icon">☷</span><span>${t("jobs.empty")}</span></div>`
        }
      </div>
    </section>
  `;
}

export function refreshJobsPanel(state, container = document) {
  const currentPanel = container.querySelector(".inspector-jobs");
  if (!currentPanel) return;
  const currentList = currentPanel.querySelector(".inspector-job-list");
  const scrollTop = currentList?.scrollTop || 0;
  const template = document.createElement("template");
  template.innerHTML = renderJobsPanel(state).trim();
  const nextPanel = template.content.firstElementChild;
  if (!nextPanel) return;
  currentPanel.replaceWith(nextPanel);
  const nextList = nextPanel.querySelector(".inspector-job-list");
  if (nextList) nextList.scrollTop = scrollTop;
}

function renderJob(job) {
  const cancellable = job.state === "running" || job.state === "queued";
  const cancelling = job.state === "cancelling";
  return `
    <div class="job-card inspector-job-card">
      <strong>${actionLabel(job.action)}</strong>
      ${
        cancellable
          ? `<button class="ghost-button compact" data-action="cancelJob" data-job-id="${job.id}">×</button>`
          : cancelling
            ? `<span class="job-cancelling-indicator" role="status" aria-label="${t("job.cancelling")}" title="${t("job.cancelling")}"></span>`
          : `<span class="section-note">${jobLabel(job.state)}</span>`
      }
      <progress class="${cancelling ? "is-cancelling" : ""}" value="${job.progress}" max="1"></progress>
      <span class="section-note job-status-line">
        ${jobLabel(job.state)} · ${percent(job.progress)}<span data-job-timing="${escapeHTML(job.id)}">${escapeHTML(jobTimingSuffix(job))}</span>
      </span>
      ${job.errorMessage ? `<span class="job-error-message" title="${escapeHTML(job.errorMessage)}">${escapeHTML(job.errorMessage)}</span>` : ""}
    </div>
  `;
}

export function refreshJobTimings(state, container = document) {
  const runningJobIDs = new Set(
    state.jobs.filter((job) => job.state === "running").map((job) => job.id),
  );
  jobTimingEstimators.forEach((_, jobID) => {
    if (!runningJobIDs.has(jobID)) jobTimingEstimators.delete(jobID);
  });
  const jobsByID = new Map(state.jobs.map((job) => [job.id, job]));
  container.querySelectorAll("[data-job-timing]").forEach((element) => {
    const job = jobsByID.get(element.dataset.jobTiming);
    element.textContent = job ? jobTimingSuffix(job) : "";
  });
}

function jobTimingSuffix(job, now = Date.now()) {
  const startedAt = parseTimestamp(job.startedAt);
  if (job.state === "running") {
    const progress = Math.min(1, Math.max(0, Number(job.progress) || 0));
    const remaining = estimateRemainingTime(job, progress, startedAt, now);
    const time = remaining === null ? t("job.estimating") : formatDuration(remaining, true);
    return ` · ${t("job.estimatedRemaining", { time })}`;
  }
  jobTimingEstimators.delete(job.id);
  if (job.state === "completed") {
    const finishedAt = parseTimestamp(job.finishedAt);
    if (startedAt === null || finishedAt === null) return "";
    return ` · ${t("job.generationTime", { time: formatDuration(finishedAt - startedAt) })}`;
  }
  return "";
}

function estimateRemainingTime(job, progress, startedAt, now) {
  if (startedAt === null || progress <= 0 || progress >= 1) return null;
  let estimator = jobTimingEstimators.get(job.id);
  if (!estimator || estimator.startedAt !== startedAt || progress < estimator.progress) {
    estimator = {
      startedAt,
      progress: 0,
      lastProgressAt: startedAt,
      samples: [],
      rate: null,
      finishAt: null,
    };
    jobTimingEstimators.set(job.id, estimator);
  }

  if (!estimator.samples.length && progress >= ETA_STEADY_PROGRESS_START) {
    estimator.progress = progress;
    estimator.lastProgressAt = now;
    estimator.samples.push({ time: now, progress });
  } else if (estimator.samples.length
      && progress - estimator.progress >= ETA_MIN_PROGRESS_DELTA) {
    recordProgressSample(estimator, progress, now);
  }
  // Loading the model and encoding the prompt use coarse milestone values
  // (2%, 12%, 18%, 24%, 25%). Do not extrapolate those milestones as if they
  // were steady denoising progress; wait for a stable post-initialization
  // sample window before showing a numeric ETA.
  if (progress < ETA_MIN_ESTIMATE_PROGRESS
      || now - startedAt < ETA_MIN_ESTIMATE_ELAPSED_MS) {
    return null;
  }
  const fallbackRemaining = fallbackRemainingTime(progress, startedAt, now);
  if (!estimator.samples.length
      || progress - estimator.samples[0].progress < ETA_MIN_ESTIMATE_DELTA
      || !Number.isFinite(estimator.finishAt)) {
    return fallbackRemaining;
  }

  const updateIntervals = estimator.samples.slice(1).map((sample, index) =>
    sample.time - estimator.samples[index].time,
  ).filter((duration) => duration > 0);
  const typicalInterval = median(updateIntervals) ?? 15_000;
  const staleLimit = Math.min(3 * 60 * 1_000, Math.max(30_000, typicalInterval * 3));
  if (now - estimator.lastProgressAt > staleLimit) return fallbackRemaining;

  const countdownRemaining = estimator.finishAt - now;
  const rateRemaining = Number.isFinite(estimator.rate) && estimator.rate > 0
    ? (1 - progress) / estimator.rate
    : null;
  const remaining = countdownRemaining > 0 ? countdownRemaining : rateRemaining;
  return Number.isFinite(remaining) && remaining > 0 && remaining <= ETA_MAX_DURATION_MS
    ? remaining
    : fallbackRemaining;
}

function fallbackRemainingTime(progress, startedAt, now) {
  const elapsed = now - startedAt;
  if (elapsed <= 0 || progress <= 0 || progress >= 1) return null;
  const rate = progress / elapsed;
  const remaining = (1 - progress) / rate;
  return Number.isFinite(remaining) && remaining > 0 && remaining <= ETA_MAX_DURATION_MS
    ? remaining
    : null;
}

function recordProgressSample(estimator, progress, now) {
  const lastSample = estimator.samples.at(-1);
  if (lastSample && now - lastSample.time < 1_000 && estimator.samples.length > 1) {
    lastSample.progress = progress;
  } else {
    estimator.samples.push({ time: now, progress });
  }
  estimator.progress = progress;
  estimator.lastProgressAt = now;

  const cutoff = now - ETA_SAMPLE_WINDOW_MS;
  while (estimator.samples.length > 2 && estimator.samples[1].time < cutoff) {
    estimator.samples.shift();
  }
  if (estimator.samples.length > 32) {
    estimator.samples.splice(0, estimator.samples.length - 32);
  }

  const rates = [];
  for (let currentIndex = 1; currentIndex < estimator.samples.length; currentIndex += 1) {
    const current = estimator.samples[currentIndex];
    for (let previousIndex = 0; previousIndex < currentIndex; previousIndex += 1) {
      const previous = estimator.samples[previousIndex];
      const duration = current.time - previous.time;
      const completed = current.progress - previous.progress;
      if (duration >= ETA_MIN_SAMPLE_SPAN_MS && completed > 0) {
        rates.push(completed / duration);
      }
    }
  }
  const measuredRate = median(rates);
  if (!Number.isFinite(measuredRate) || measuredRate <= 0) return;

  if (Number.isFinite(estimator.rate)) {
    const boundedRate = Math.min(estimator.rate * 4, Math.max(estimator.rate / 4, measuredRate));
    estimator.rate = estimator.rate * 0.65 + boundedRate * 0.35;
  } else {
    estimator.rate = measuredRate;
  }
  const remaining = (1 - progress) / estimator.rate;
  estimator.finishAt = now + remaining;
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function parseTimestamp(value) {
  if (!value) return null;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : null;
}

function formatDuration(milliseconds, roundUp = false) {
  const secondsValue = Math.max(0, milliseconds / 1_000);
  const totalSeconds = roundUp ? Math.ceil(secondsValue) : Math.round(secondsValue);
  const hours = Math.floor(totalSeconds / 3_600);
  const minutes = Math.floor(totalSeconds % 3_600 / 60);
  const seconds = totalSeconds % 60;
  const twoDigits = (value) => String(value).padStart(2, "0");
  return hours > 0
    ? `${hours}:${twoDigits(minutes)}:${twoDigits(seconds)}`
    : `${twoDigits(minutes)}:${twoDigits(seconds)}`;
}

function numberField(label, field, value, min, max, outputKind, step = 1) {
  return `<div class="field-group">
    <label for="${outputKind}-${field}">${label}</label>
    <input
      id="${outputKind}-${field}"
      class="field"
      type="number"
      min="${min}"
      max="${max}"
      step="${step}"
      ${settingsFieldAttribute(outputKind, field)}
      data-preserve-focus="${outputKind}-${field}"
      value="${value}"
    />
  </div>`;
}

function settingsFieldAttribute(outputKind, field) {
  return outputKind === "video" ? `data-video-field="${field}"` : `data-recipe-field="${field}"`;
}

function previewModeButton(mode, title, ui) {
  return `<button class="mode-button compact ${ui.previewMode === mode ? "active" : ""}" data-action="previewMode" data-mode="${mode}">${title}</button>`;
}

function detailRow(label, value) {
  return `<div class="detail-row"><span>${escapeHTML(label)}</span><span>${escapeHTML(value)}</span></div>`;
}

function assetSummary(asset) {
  if (asset.kind === "generatedAudio") {
    const format = String(asset.audioFormat || "").toUpperCase();
    const duration = formatDuration((asset.mediaDurationSeconds || 0) * 1_000);
    return [format, duration].filter(Boolean).join(" · ");
  }
  return `${asset.pixelWidth} × ${asset.pixelHeight}`;
}

function selectedAsset(state) {
  return state.assets.find((asset) => asset.id === state.selectedAssetID);
}

function buildLineage(assets, start) {
  const byID = new Map(assets.map((asset) => [asset.id, asset]));
  const result = [];
  const visited = new Set();
  let current = start;
  while (current && !visited.has(current.id)) {
    visited.add(current.id);
    result.unshift(current);
    current = current.parentAssetID ? byID.get(current.parentAssetID) : null;
  }
  return result;
}

function emptyState(title, description, icon) {
  return `<div class="empty-state"><span class="empty-icon">${icon}</span><strong>${title}</strong><span>${description}</span></div>`;
}
