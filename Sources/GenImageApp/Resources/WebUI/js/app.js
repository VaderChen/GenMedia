import { invoke, onClipboardImage, onState } from "./bridge.js";
import { renderDownloads } from "./downloads.js";
import { escapeHTML, gigabytes, percent, phaseLabel } from "./format.js";
import { getLocale, setLocale, t } from "./i18n.js";
import { renderModels } from "./models.js";
import { appendProfileLoRARow, removeProfileLoRARow, renderProfiles } from "./profiles.js";
import { renderSettings } from "./settings.js";
import { getTheme, setTheme } from "./themes.js";
import {
  isInferenceBusy,
  refreshJobsPanel,
  refreshJobTimings,
  renderCreationPanel,
  renderQuickTools,
  renderWorkspace,
  selectedSourceImage,
  sourceImageDimensions,
} from "./workspace.js";

const root = document.querySelector("#app");
const WORKSPACE_TABS_KEY = "genimage.workspaceTabs";
const STATUS_MESSAGE_DURATION_MS = 5_000;
const JOB_TIMING_REFRESH_MS = 1_000;
const PROMPT_TABS = new Set(["prompt", "negative", "lyrics", "imageOutput", "videoOutput", "musicOutput"]);
const GENERATION_TYPES = new Set(["image", "video", "music"]);
const PROMPT_TABS_BY_GENERATION_TYPE = {
  image: new Set(["prompt", "negative", "imageOutput"]),
  video: new Set(["prompt", "negative", "videoOutput"]),
  music: new Set(["prompt", "lyrics", "musicOutput"]),
};
const AUDIO_FFT_SIZE = 8192;
const AUDIO_MIN_FREQUENCY_HZ = 18;
const AUDIO_MAX_FREQUENCY_HZ = 24_000;
const AUDIO_MAX_WAVEFORM_POINTS = 1_600;
const SUPPORTED_IMAGE_FILE_PATTERN = /\.(png|jpe?g|webp|gif|tiff?|heic|heif)$/i;

let state = null;
let recipeTimer = null;
let statusMessageTimer = null;
let activeEditableField = null;
let composingEditableField = null;
let renderDeferredDuringEditing = false;
let deferredRenderTimer = null;
let previewPan = null;
let imageDragDepth = 0;
let imageDropOverlay = null;
let stateContentSignature = null;
const pendingOutputs = [];
const pasteState = { image: null };
const audioVisualizerBindings = new Map();
let audioVisualizerContext = null;
const savedWorkspaceTabs = loadWorkspaceTabs();
const savedPromptTab = normalizeLegacyPromptTab(localStorage.getItem("genimage.promptTab"));
const savedGenerationType = normalizeGenerationType(
  localStorage.getItem("genimage.generationType"),
  savedPromptTab,
);
const ui = {
  route: "workspace",
  previewMode: "single",
  zoom: 1,
  creationCollapsed: localStorage.getItem("genimage.creationCollapsed") === "true",
  generationType: savedGenerationType,
  promptTab: normalizePromptTab(savedPromptTab, savedGenerationType),
  inspectorTab: localStorage.getItem("genimage.inspectorTab") === "jobs" ? "jobs" : "info",
  workspaceTabs: savedWorkspaceTabs.tabs,
  activeWorkspaceTabID: savedWorkspaceTabs.activeTabID,
  renameWorkspaceTabID: null,
  renameWorkspaceTabValue: "",
  pasteDialogOpen: false,
  modelFilter: "all",
  profileFilter: "all",
  modelSearch: "",
  language: getLocale(),
  theme: getTheme(),
  profileHintVisible: localStorage.getItem("genimage.profileHintSeen") !== "true",
};

onClipboardImage((image) => {
  if (!state || typeof image?.dataURL !== "string" || !image.dataURL.startsWith("data:image/")) return;
  handleClipboardImage(image.dataURL, image.name).catch(showBridgeError);
});

onState((nextState) => {
  if (nextState.schemaVersion !== 1) {
    showBridgeError(new Error(`Unsupported Bridge schema: ${nextState.schemaVersion}`));
    return;
  }
  const statusMessageChanged = nextState.statusMessage !== state?.statusMessage;
  const nextContentSignature = contentSignature(nextState);
  if (state && activeEditableField) {
    const contentChanged = contentSignatureWithoutEditableSettings(nextState)
      !== contentSignatureWithoutEditableSettings(state);
    const localRecipe = state.recipe;
    const localVideoOutputSettings = state.videoOutputSettings;
    const localMusicOutputSettings = state.musicOutputSettings;
    reconcileWorkspaceTabs(nextState);
    state = {
      ...nextState,
      recipe: localRecipe,
      videoOutputSettings: localVideoOutputSettings,
      musicOutputSettings: localMusicOutputSettings,
    };
    stateContentSignature = nextContentSignature;
    renderDeferredDuringEditing = renderDeferredDuringEditing || contentChanged;
    if (statusMessageChanged) scheduleStatusMessageDismiss(nextState.statusMessage);
    refreshModelProgressDOM();
    updateSystemMetricsDOM();
    refreshJobsPanel(workspaceStateForActiveTab(state), root);
    refreshToastDOM();
    return;
  }
  if (state && modelProgressOnlyChanged(state, nextState)) {
    state = nextState;
    stateContentSignature = nextContentSignature;
    refreshModelProgressDOM();
    updateSystemMetricsDOM();
    refreshJobsPanel(workspaceStateForActiveTab(state), root);
    if (statusMessageChanged) scheduleStatusMessageDismiss(nextState.statusMessage);
    refreshToastDOM();
    return;
  }
  if (state && nextContentSignature === stateContentSignature) {
    const localRecipe = state.recipe;
    const localVideoOutputSettings = state.videoOutputSettings;
    const localMusicOutputSettings = state.musicOutputSettings;
    state = {
      ...nextState,
      recipe: localRecipe,
      videoOutputSettings: localVideoOutputSettings,
      musicOutputSettings: localMusicOutputSettings,
    };
    if (statusMessageChanged) scheduleStatusMessageDismiss(nextState.statusMessage);
    updateSystemMetricsDOM();
    refreshJobsPanel(workspaceStateForActiveTab(state), root);
    refreshToastDOM();
    return;
  }
  const descriptionUpdated = state
    && nextState.recipe.prompt !== state.recipe.prompt
    && nextState.operations.slice(state.operations.length).some((operation) => operation.action === "describe");
  if (descriptionUpdated) {
    setGenerationType("image", "prompt");
    ui.creationCollapsed = false;
    localStorage.setItem("genimage.creationCollapsed", "false");
  }
  reconcileWorkspaceTabs(nextState);
  state = nextState;
  stateContentSignature = nextContentSignature;
  if (statusMessageChanged) scheduleStatusMessageDismiss(nextState.statusMessage);
  render();
});

invoke("bootstrap").catch(showBridgeError);

setInterval(() => {
  if (state) refreshJobTimings(state, root);
}, JOB_TIMING_REFRESH_MS);

root.addEventListener("click", async (event) => {
  const target = event.target.closest("[data-action]");
  if (!target || !state) return;
  const action = target.dataset.action;

  try {
    switch (action) {
      case "navigate":
        ui.route = target.dataset.route;
        if (ui.route === "profiles") dismissProfileHint();
        render();
        break;
      case "dismissProfileHint":
        dismissProfileHint();
        render();
        break;
      case "openAvailableUpdate":
        await invoke("openAvailableUpdate");
        break;
      case "dismissAvailableUpdate":
        await invoke("dismissAvailableUpdate");
        break;
      case "generate":
        await syncRecipe();
        ui.route = "workspace";
        ui.previewMode = "single";
        setInspectorTab("jobs");
        await invokeTrackedOutput("generate", undefined, "generate");
        break;
      case "generateVideo":
        await Promise.all([syncRecipe(), syncVideoOutputSettings()]);
        ui.route = "workspace";
        ui.previewMode = "single";
        setInspectorTab("jobs");
        await invokeTrackedOutput(
          "generateVideo",
          { sourceAssetIDs: selectedVideoSourceAssetIDs() },
          "generateVideo",
        );
        break;
      case "generateMusic":
        await Promise.all([syncRecipe(), syncMusicOutputSettings()]);
        ui.route = "workspace";
        ui.previewMode = "single";
        setInspectorTab("jobs");
        await invokeTrackedOutput("generateMusic", undefined, "generateMusic");
        break;
      case "describe":
        ui.route = "workspace";
        setInspectorTab("jobs");
        await invoke("describe");
        break;
      case "imageToImage":
        await syncRecipe();
        ui.route = "workspace";
        ui.previewMode = "single";
        setInspectorTab("jobs");
        await invokeTrackedOutput("imageToImage", undefined, "imageToImage");
        break;
      case "upscale":
        ui.route = "workspace";
        ui.previewMode = "single";
        setInspectorTab("jobs");
        await invokeTrackedOutput("upscale", undefined, "upscale");
        break;
      case "importImage":
        ui.route = "workspace";
        ui.previewMode = "single";
        setInspectorTab("info");
        await invokeTrackedOutput("importImage", undefined, "importImage");
        break;
      case "selectAsset":
        ui.route = "workspace";
        ui.previewMode = "single";
        {
          const primaryAssetID = setActiveTabSelection(target.dataset.assetId, event);
          if (!primaryAssetID) break;
          setInspectorTab("info");
          await invoke("selectAsset", { assetID: primaryAssetID });
        }
        break;
      case "removeAsset": {
        const assetID = target.dataset.assetId;
        const replacementAssetID = replacementAssetIDAfterRemoval(assetID);
        await invoke("removeAsset", { assetID, replacementAssetID });
        removeAssetFromWorkspaceTabs(assetID, replacementAssetID);
        break;
      }
      case "workspaceTab":
        if (target.dataset.tabId === ui.activeWorkspaceTabID) {
          openWorkspaceTabRename(target.dataset.tabId);
        } else {
          await activateWorkspaceTab(target.dataset.tabId);
        }
        break;
      case "workspaceAddTab":
        addWorkspaceTab();
        break;
      case "workspaceCloseTab":
        await closeWorkspaceTab(target.dataset.tabId);
        break;
      case "workspaceRenameCancel":
        closeWorkspaceTabRename();
        break;
      case "workspaceRenameSave":
        saveWorkspaceTabRename();
        break;
      case "pasteImageDecision": {
        const pastedImage = pasteState.image;
        if (!pastedImage) break;
        const describe = target.dataset.describe === "true";
        pasteState.image = null;
        ui.pasteDialogOpen = false;
        ui.route = "workspace";
        ui.previewMode = "single";
        if (describe) {
          setGenerationType("image", "prompt");
          setInspectorTab("jobs");
        } else {
          setInspectorTab("info");
        }
        await invoke("pasteImage", { dataURL: pastedImage.dataURL, describe });
        break;
      }
      case "inspectorTab":
        setInspectorTab(target.dataset.tab);
        break;
      case "previewMode":
        ui.previewMode = target.dataset.mode;
        render();
        break;
      case "toggleCreationPanel":
        ui.creationCollapsed = !ui.creationCollapsed;
        localStorage.setItem("genimage.creationCollapsed", String(ui.creationCollapsed));
        refreshCreationPanelDOM();
        break;
      case "promptTab":
        ui.promptTab = normalizePromptTab(target.dataset.tab, ui.generationType);
        localStorage.setItem("genimage.promptTab", ui.promptTab);
        refreshCreationPanelDOM();
        break;
      case "zoomIn":
        if (isPreviewZoomDisabled()) break;
        ui.zoom = clampPreviewZoom(ui.zoom + 0.25);
        render();
        break;
      case "zoomOut":
        if (isPreviewZoomDisabled()) break;
        ui.zoom = clampPreviewZoom(ui.zoom - 0.25);
        render();
        break;
      case "fitPreview":
        if (isPreviewZoomDisabled()) break;
        ui.zoom = 1;
        render();
        break;
      case "randomizeSeed":
        await invoke(
          target.dataset.outputKind === "music"
            ? "randomizeMusicSeed"
            : target.dataset.outputKind === "video"
              ? "randomizeVideoSeed"
              : "randomizeSeed",
        );
        break;
      case "applyProfileDefaults":
        await invoke("applyProfileDefaults", {
          outputKind: ui.generationType,
        });
        break;
      case "cancelJob":
        await invoke("cancelJob", { jobID: target.dataset.jobId });
        break;
      case "releaseMemory":
        await invoke("releaseMemory");
        break;
      case "clearJobs":
        await invoke("clearJobs");
        break;
      case "modelFilter":
        ui.modelFilter = target.dataset.filter;
        render();
        break;
      case "profileFilter":
        ui.profileFilter = target.dataset.filter;
        render();
        break;
      case "chooseModelRoot":
        await invoke("chooseModelRoot");
        break;
      case "chooseOutputDirectory":
      case "revealOutputDirectory":
        await invoke(action);
        break;
      case "installModel":
      case "pauseModel":
      case "removeModel":
      case "repairModel":
        await invoke(action, { modelID: target.dataset.modelId });
        break;
      case "installProfileModels":
        await invoke("installProfileModels", { profileID: target.dataset.profileId });
        break;
      case "duplicateProfile":
        await invoke("duplicateProfile", { profileID: target.dataset.profileId });
        break;
      case "activateProfile":
        await invoke("selectProfile", {
          profileID: target.dataset.profileId,
          capability: target.dataset.capability,
        });
        break;
      case "deactivateProfile":
        await invoke("deactivateProfile", {
          profileID: target.dataset.profileId,
          capability: target.dataset.capability,
        });
        break;
      case "saveProfile":
        await saveProfile(target.dataset.profileId);
        break;
      case "addProfileLoRA":
        appendProfileLoRARow(target);
        break;
      case "removeProfileLoRA":
        removeProfileLoRARow(target);
        break;
      case "deleteProfile":
        await invoke("deleteProfile", { profileID: target.dataset.profileId });
        break;
      case "clearStatus":
        await invoke("clearStatus");
        break;
      case "selectTheme":
        ui.theme = target.dataset.theme;
        setTheme(ui.theme);
        render();
        break;
    }
  } catch (error) {
    showBridgeError(error);
  }
});

document.addEventListener("contextmenu", (event) => {
  const media = event.target.closest?.(".asset-artwork [data-asset-id]");
  if (!media) return;
  event.preventDefault();
  event.stopPropagation();
  openAssetContextMenu(event.clientX, event.clientY, media.dataset.assetId);
});

document.addEventListener("pointerdown", (event) => {
  if (!event.target.closest?.(".image-context-menu")) closeImageContextMenu();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeImageContextMenu();
});

root.addEventListener("change", async (event) => {
  if (event.target.dataset.uiField === "generationType") {
    setGenerationType(event.target.value);
    refreshCreationPanelDOM();
    return;
  }

  if (event.target.matches("[data-model-root]")) {
    const path = event.target.value.trim();
    if (!path || path === state.modelRootPath) return;
    try {
      await invoke("setModelRoot", { path });
    } catch (error) {
      showBridgeError(error);
    }
    return;
  }

  if (event.target.matches("[data-output-directory]")) {
    const path = event.target.value.trim();
    if (!path || path === state.outputDirectoryPath) return;
    try {
      await invoke("setOutputDirectory", { path });
    } catch (error) {
      showBridgeError(error);
    }
    return;
  }

  if (event.target.dataset.setting === "language") {
    ui.language = event.target.value;
    setLocale(ui.language);
    render();
    return;
  }

  if (event.target.matches("[data-aspect-ratio-select]")) {
    try {
      const outputKind = event.target.dataset.outputKind === "video" ? "video" : "image";
      const settings = outputKind === "video" ? state.videoOutputSettings : state.recipe;
      const option = event.target.selectedOptions[0];
      let dimensions;
      if (option?.value === "original") {
        const sourceAsset = selectedSourceImage(workspaceStateForActiveTab(state));
        if (!sourceAsset) return;
        dimensions = sourceImageDimensions(sourceAsset);
      } else {
        const ratioWidth = Number(option?.dataset.ratioWidth);
        const ratioHeight = Number(option?.dataset.ratioHeight);
        if (!Number.isFinite(ratioWidth) || !Number.isFinite(ratioHeight) || ratioWidth <= 0 || ratioHeight <= 0) return;
        dimensions = outputKind === "video"
          ? defaultVideoDimensionsForAspect(ratioWidth, ratioHeight)
          : dimensionsForAspect("width", settings.width, ratioWidth, ratioHeight);
      }
      settings.width = dimensions.width;
      settings.height = dimensions.height;
      render();
      await (outputKind === "video" ? syncVideoOutputSettings() : syncRecipe());
    } catch (error) {
      showBridgeError(error);
    }
    return;
  }

  if (event.target.matches("[data-lora-select]")) {
    state.recipe.loraID = event.target.value || null;
    if (!Number.isFinite(Number(state.recipe.loraScale))) state.recipe.loraScale = 1;
    render();
    try {
      await syncRecipe();
    } catch (error) {
      showBridgeError(error);
    }
    return;
  }

  if (event.target.matches("[data-lora-scale]")) {
    try {
      await syncRecipe();
    } catch (error) {
      showBridgeError(error);
    }
    return;
  }

  const dimensionField = event.target.dataset.dimensionField;
  if (dimensionField) {
    try {
      await (event.target.dataset.outputKind === "video"
        ? syncVideoOutputSettings()
        : syncRecipe());
    } catch (error) {
      showBridgeError(error);
    }
    return;
  }

  if (event.target.dataset.videoField) {
    try {
      await syncVideoOutputSettings();
    } catch (error) {
      showBridgeError(error);
    }
    return;
  }

  if (event.target.dataset.musicField) {
    state.musicOutputSettings[event.target.dataset.musicField] = event.target.value;
    try {
      await syncMusicOutputSettings();
    } catch (error) {
      showBridgeError(error);
    }
    return;
  }

  if (event.target.dataset.recipeField && event.target.tagName !== "TEXTAREA") {
    try {
      await syncRecipe();
    } catch (error) {
      showBridgeError(error);
    }
    return;
  }

  const profileSelect = event.target.closest("[data-profile-capability]");
  if (profileSelect) {
    if (!profileSelect.value) return;
    try {
      await invoke("selectProfile", {
        profileID: profileSelect.value,
        capability: profileSelect.dataset.profileCapability,
      });
    } catch (error) {
      showBridgeError(error);
    }
  }
});

root.addEventListener("input", (event) => {
  if (!state) return;

  if (event.target.matches("[data-lora-scale]")) {
    state.recipe.loraScale = Math.min(1, Math.max(0, Number(event.target.value)));
    const output = root.querySelector("[data-lora-scale-value]");
    if (output) output.textContent = `${Math.round(state.recipe.loraScale * 100)}%`;
    return;
  }

  const dimensionField = event.target.dataset.dimensionField;
  if (dimensionField) {
    const outputKind = event.target.dataset.outputKind === "video" ? "video" : "image";
    const settings = outputKind === "video" ? state.videoOutputSettings : state.recipe;
    const dimensions = dimensionsForAspect(
      dimensionField,
      Number(event.target.value),
      Number(event.target.dataset.ratioWidth),
      Number(event.target.dataset.ratioHeight),
    );
    settings.width = dimensions.width;
    settings.height = dimensions.height;
    updateResolutionControls(dimensions, outputKind);
    return;
  }

  const editableField = editableFieldFor(event.target);
  if (editableField) {
    updateEditableFieldValue(editableField, event.target.value);
    if (
      event.isComposing
      || composingEditableField === editableField.key
      || event.target.tagName !== "TEXTAREA"
    ) return;
    clearTimeout(recipeTimer);
    recipeTimer = setTimeout(() => syncEditableField(editableField).catch(showBridgeError), 240);
    return;
  }

  if (event.target.dataset.uiField === "zoom") {
    if (isPreviewZoomDisabled()) return;
    ui.zoom = Number(event.target.value);
    render();
    return;
  }

  if (event.target.dataset.uiField === "modelSearch") {
    ui.modelSearch = event.target.value;
    render();
    return;
  }

  if (event.target.dataset.uiField === "workspaceTabName") {
    ui.renameWorkspaceTabValue = event.target.value;
  }
});

root.addEventListener("focusin", (event) => {
  const editableField = editableFieldFor(event.target);
  if (!editableField) return;
  clearTimeout(deferredRenderTimer);
  deferredRenderTimer = null;
  activeEditableField = editableField;
});

root.addEventListener("focusout", (event) => {
  const editableField = editableFieldFor(event.target);
  if (!editableField || activeEditableField?.element !== event.target) return;
  updateEditableFieldValue(editableField, event.target.value);
  activeEditableField = null;
  if (composingEditableField !== editableField.key) scheduleDeferredRender();
});

root.addEventListener("compositionstart", (event) => {
  const editableField = editableFieldFor(event.target);
  if (!editableField) return;
  activeEditableField = editableField;
  composingEditableField = editableField.key;
  clearTimeout(recipeTimer);
});

root.addEventListener("compositionend", (event) => {
  const editableField = editableFieldFor(event.target);
  if (!state || !editableField || composingEditableField !== editableField.key) return;

  updateEditableFieldValue(editableField, event.target.value);
  composingEditableField = null;
  clearTimeout(recipeTimer);
  recipeTimer = setTimeout(() => syncEditableField(editableField).catch(showBridgeError), 80);
  if (!activeEditableField) scheduleDeferredRender();
});

root.addEventListener("keydown", (event) => {
  if (!event.target.matches("[data-model-root]") || event.key !== "Enter") return;
  event.preventDefault();
  event.target.blur();
});

document.addEventListener("paste", async (event) => {
  if (!state) return;
  const imageItem = Array.from(event.clipboardData?.items || []).find((item) => item.type.startsWith("image/"));
  const imageFile = imageItem?.getAsFile()
    || Array.from(event.clipboardData?.files || []).find(isSupportedImageFile);
  if (!imageFile) return;

  event.preventDefault();
  try {
    await handleClipboardImage(await readFileAsDataURL(imageFile), imageFile.name);
  } catch (error) {
    showBridgeError(error);
  }
});

document.addEventListener("dragenter", (event) => {
  if (!dataTransferHasFiles(event.dataTransfer)) return;
  event.preventDefault();
  imageDragDepth += 1;
  showImageDropOverlay();
});

document.addEventListener("dragover", (event) => {
  if (!dataTransferHasFiles(event.dataTransfer)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";
});

document.addEventListener("dragleave", (event) => {
  if (!imageDragDepth) return;
  event.preventDefault();
  imageDragDepth = Math.max(0, imageDragDepth - 1);
  if (!imageDragDepth) hideImageDropOverlay();
});

document.addEventListener("drop", async (event) => {
  if (!state || !dataTransferHasFiles(event.dataTransfer)) return;
  event.preventDefault();
  const imageFiles = Array.from(event.dataTransfer.files || []).filter(isSupportedImageFile);
  resetImageDragState();
  if (!imageFiles.length) {
    showBridgeError(new Error(t("import.unsupportedImage")));
    return;
  }

  try {
    await importDroppedImageFiles(imageFiles);
  } catch (error) {
    showBridgeError(error);
  }
});

window.addEventListener("blur", resetImageDragState);

root.addEventListener(
  "wheel",
  (event) => {
    const stage = event.target.closest(
      ".preview-stage[data-pan-enabled=\"true\"][data-zoom-enabled=\"true\"]",
    );
    if (!stage || !state) return;
    event.preventDefault();

    const previousZoom = ui.zoom;
    const nextZoom = clampPreviewZoom(previousZoom + (event.deltaY < 0 ? 0.1 : -0.1));
    if (nextZoom === previousZoom) return;

    const bounds = stage.getBoundingClientRect();
    const offsetX = event.clientX - bounds.left;
    const offsetY = event.clientY - bounds.top;
    const contentX = stage.scrollLeft + offsetX;
    const contentY = stage.scrollTop + offsetY;
    ui.zoom = nextZoom;
    render();

    const nextStage = root.querySelector('.preview-stage[data-pan-enabled="true"]');
    if (!nextStage) return;
    const ratio = nextZoom / previousZoom;
    nextStage.scrollLeft = contentX * ratio - offsetX;
    nextStage.scrollTop = contentY * ratio - offsetY;
  },
  { passive: false },
);

root.addEventListener("pointerdown", (event) => {
  const stage = event.target.closest('.preview-stage[data-pan-enabled="true"]');
  if (!stage || event.button !== 0 || event.target.closest("video, audio")) return;
  event.preventDefault();
  previewPan = {
    pointerID: event.pointerId,
    stage,
    startX: event.clientX,
    startY: event.clientY,
    scrollLeft: stage.scrollLeft,
    scrollTop: stage.scrollTop,
  };
  stage.classList.add("panning");
  stage.setPointerCapture?.(event.pointerId);
});

root.addEventListener("pointermove", (event) => {
  if (!previewPan || previewPan.pointerID !== event.pointerId) return;
  previewPan.stage.scrollLeft = previewPan.scrollLeft - (event.clientX - previewPan.startX);
  previewPan.stage.scrollTop = previewPan.scrollTop - (event.clientY - previewPan.startY);
});

root.addEventListener("pointerup", stopPreviewPan);
root.addEventListener("pointercancel", stopPreviewPan);

function render() {
  if (!state) return;
  clearTimeout(deferredRenderTimer);
  deferredRenderTimer = null;
  renderDeferredDuringEditing = false;
  previewPan = null;
  const viewState = captureViewState();
  const preservedMedia = preservePlayingMedia();
  const preservedAudio = new Set(
    preservedMedia
      .filter(({ media }) => media.tagName === "AUDIO")
      .map(({ media }) => media),
  );
  disposeAudioVisualizers(preservedAudio);
  root.innerHTML = `
    <div class="app-shell">
      ${renderSidebar()}
      <div class="main-view">
        ${renderUpdateBanner()}
        <div class="route-view">${renderRoute()}</div>
      </div>
    </div>
    ${renderWorkspaceTabRenameDialog()}
    ${renderPasteDialog()}
    ${renderToast()}
  `;
  restoreViewState(viewState);
  restorePlayingMedia(preservedMedia);
  setupAudioVisualizers();
  refreshJobTimings(state, root);
}

function refreshCreationPanelDOM() {
  const currentPanel = root.querySelector(".creation-panel");
  if (!state || ui.route !== "workspace" || !currentPanel) {
    render();
    return;
  }

  const currentScroll = currentPanel.querySelector('[data-scroll-id="creation"]');
  const scrollPosition = currentScroll
    ? { top: currentScroll.scrollTop, left: currentScroll.scrollLeft }
    : null;
  const template = document.createElement("template");
  template.innerHTML = renderCreationPanel(workspaceStateForActiveTab(state), ui).trim();
  const nextPanel = template.content.firstElementChild;
  if (!nextPanel) {
    render();
    return;
  }

  currentPanel.replaceWith(nextPanel);
  const nextScroll = nextPanel.querySelector('[data-scroll-id="creation"]');
  if (nextScroll && scrollPosition) {
    nextScroll.scrollTop = scrollPosition.top;
    nextScroll.scrollLeft = scrollPosition.left;
  }
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

function setupAudioVisualizers() {
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

function preservePlayingMedia() {
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

function restorePlayingMedia(preservedMedia) {
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

function openAssetContextMenu(clientX, clientY, assetID) {
  closeImageContextMenu();
  const asset = state.assets.find((item) => item.id === assetID);
  if (!asset) return;
  const menu = document.createElement("div");
  menu.className = "image-context-menu";
  menu.setAttribute("role", "menu");
  menu.innerHTML = [
    ["openAsset", t("context.openAsset")],
    ["revealAsset", t("context.openDirectory")],
    ["downloadAsset", t("context.downloadAsset")],
    ...(isImageAsset(asset) ? [["copyAsset", t("context.copyImage")]] : []),
    ["shareAsset", t("context.shareAsset")],
  ].map(([action, label]) => `<button type="button" role="menuitem" data-context-action="${action}">${label}</button>`).join("");
  menu.style.left = `${Math.max(8, clientX)}px`;
  menu.style.top = `${Math.max(8, clientY)}px`;
  document.body.append(menu);
  requestAnimationFrame(() => {
    const bounds = menu.getBoundingClientRect();
    menu.style.left = `${Math.min(Math.max(8, clientX), window.innerWidth - bounds.width - 8)}px`;
    menu.style.top = `${Math.min(Math.max(8, clientY), window.innerHeight - bounds.height - 8)}px`;
  });
  menu.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-context-action]");
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    const action = button.dataset.contextAction;
    closeImageContextMenu();
    try {
      await invoke(action, { assetID });
    } catch (error) {
      showBridgeError(error);
    }
  });
}

function closeImageContextMenu() {
  document.querySelectorAll(".image-context-menu").forEach((menu) => menu.remove());
}

function setInspectorTab(tab) {
  ui.inspectorTab = tab === "jobs" ? "jobs" : "info";
  localStorage.setItem("genimage.inspectorTab", ui.inspectorTab);
  render();
}

function makeWorkspaceTab() {
  const id = globalThis.crypto?.randomUUID?.() || `tab-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return {
    id,
    name: formatWorkspaceTabName(new Date()),
    assetIDs: [],
    selectedAssetID: null,
    selectedAssetIDs: [],
    selectionAnchorID: null,
  };
}

function formatWorkspaceTabName(date) {
  const twoDigits = (value) => String(value).padStart(2, "0");
  return `${twoDigits(date.getMonth() + 1)}/${twoDigits(date.getDate())} ${twoDigits(date.getHours())}:${twoDigits(date.getMinutes())}:${twoDigits(date.getSeconds())}`;
}

function loadWorkspaceTabs() {
  try {
    const stored = JSON.parse(localStorage.getItem(WORKSPACE_TABS_KEY) || "null");
    const tabs = Array.isArray(stored?.tabs)
      ? stored.tabs
          .filter((tab) => typeof tab?.id === "string")
          .map((tab, index) => ({
            id: tab.id,
            name: typeof tab.name === "string" && tab.name.trim()
              ? tab.name.trim()
              : formatWorkspaceTabName(new Date(Date.now() + index * 1_000)),
            assetIDs: Array.isArray(tab.assetIDs) ? tab.assetIDs.filter((id) => typeof id === "string") : [],
            selectedAssetID: typeof tab.selectedAssetID === "string" ? tab.selectedAssetID : null,
            selectedAssetIDs: Array.isArray(tab.selectedAssetIDs)
              ? tab.selectedAssetIDs.filter((id) => typeof id === "string")
              : [],
            selectionAnchorID: typeof tab.selectionAnchorID === "string" ? tab.selectionAnchorID : null,
          }))
      : [];
    if (tabs.length) {
      const activeTabID = tabs.some((tab) => tab.id === stored.activeTabID) ? stored.activeTabID : tabs[0].id;
      return { tabs, activeTabID };
    }
  } catch {
    localStorage.removeItem(WORKSPACE_TABS_KEY);
  }

  const tab = makeWorkspaceTab();
  return { tabs: [tab], activeTabID: tab.id };
}

function saveWorkspaceTabs() {
  localStorage.setItem(
    WORKSPACE_TABS_KEY,
    JSON.stringify({ tabs: ui.workspaceTabs, activeTabID: ui.activeWorkspaceTabID }),
  );
}

function activeWorkspaceTab() {
  return ui.workspaceTabs.find((tab) => tab.id === ui.activeWorkspaceTabID) || ui.workspaceTabs[0];
}

function workspaceTabOwningAsset(assetID) {
  return ui.workspaceTabs.find((tab) => tab.assetIDs.includes(assetID));
}

function reconcileWorkspaceTabs(nextState) {
  if (!ui.workspaceTabs.length) {
    const tab = makeWorkspaceTab();
    ui.workspaceTabs = [tab];
    ui.activeWorkspaceTabID = tab.id;
  }

  if (!ui.workspaceTabs.some((tab) => tab.id === ui.activeWorkspaceTabID)) {
    ui.activeWorkspaceTabID = ui.workspaceTabs[0].id;
  }

  const validAssetIDs = new Set(nextState.assets.map((asset) => asset.id));
  const imageAssetIDs = new Set(
    nextState.assets
      .filter(isImageAsset)
      .map((asset) => asset.id),
  );
  const assignedAssetIDs = new Set();
  ui.workspaceTabs.forEach((tab) => {
    tab.assetIDs = tab.assetIDs.filter((id) => validAssetIDs.has(id) && !assignedAssetIDs.has(id));
    tab.assetIDs.forEach((id) => assignedAssetIDs.add(id));
    tab.selectedAssetIDs = (tab.selectedAssetIDs || []).filter(
      (id) => tab.assetIDs.includes(id) && imageAssetIDs.has(id),
    );
    if (!tab.selectionAnchorID || !tab.selectedAssetIDs.includes(tab.selectionAnchorID)) {
      tab.selectionAnchorID = tab.selectedAssetIDs.at(-1) || null;
    }
  });

  const unassignedAssetIDs = new Set(
    nextState.assets.map((asset) => asset.id).filter((id) => !assignedAssetIDs.has(id)),
  );
  bindPendingOutputJobs(nextState);

  nextState.operations.forEach((operation) => {
    const outputIDs = operation.outputAssetIDs.filter((id) => unassignedAssetIDs.has(id));
    if (!outputIDs.length) return;

    const parentTab = operation.inputAssetID ? workspaceTabOwningAsset(operation.inputAssetID) : null;
    const pendingTabID = takePendingOutputTab(operation.action, nextState);
    const targetTab = parentTab
      || ui.workspaceTabs.find((tab) => tab.id === pendingTabID)
      || activeWorkspaceTab();
    outputIDs.forEach((id) => {
      targetTab.assetIDs.push(id);
      unassignedAssetIDs.delete(id);
    });
  });

  const fallbackTab = activeWorkspaceTab();
  nextState.assets.forEach((asset) => {
    if (!unassignedAssetIDs.has(asset.id)) return;
    fallbackTab.assetIDs.push(asset.id);
    unassignedAssetIDs.delete(asset.id);
  });

  const jobStateByID = new Map(nextState.jobs.map((job) => [job.id, job.state]));
  for (let index = pendingOutputs.length - 1; index >= 0; index -= 1) {
    if (["failed", "cancelled"].includes(jobStateByID.get(pendingOutputs[index].jobID))) {
      pendingOutputs.splice(index, 1);
    }
  }

  const selectedOwner = nextState.selectedAssetID ? workspaceTabOwningAsset(nextState.selectedAssetID) : null;
  if (selectedOwner) selectedOwner.selectedAssetID = nextState.selectedAssetID;
  ui.workspaceTabs.forEach((tab) => {
    if (!tab.selectedAssetID || !tab.assetIDs.includes(tab.selectedAssetID)) {
      tab.selectedAssetID = tab.assetIDs.at(-1) || null;
    }
  });
  saveWorkspaceTabs();
}

function workspaceStateForActiveTab(sourceState) {
  const tab = activeWorkspaceTab();
  const assetIDs = new Set(tab?.assetIDs || []);
  const assets = sourceState.assets.filter((asset) => assetIDs.has(asset.id));
  const selectedAssetID = tab?.selectedAssetID && assetIDs.has(tab.selectedAssetID)
    ? tab.selectedAssetID
    : assets.at(-1)?.id || null;
  const comparisonAssetID = sourceState.comparisonAssetID && assetIDs.has(sourceState.comparisonAssetID)
    ? sourceState.comparisonAssetID
    : null;
  const operations = sourceState.operations.filter((operation) =>
    operation.outputAssetIDs.some((id) => assetIDs.has(id)),
  );
  const selectedAssetIDs = (tab?.selectedAssetIDs || []).filter((id) => assetIDs.has(id));
  return {
    ...sourceState,
    assets,
    selectedAssetID,
    selectedAssetIDs,
    comparisonAssetID,
    operations,
  };
}

function addWorkspaceTab() {
  const tab = makeWorkspaceTab();
  ui.workspaceTabs.push(tab);
  ui.activeWorkspaceTabID = tab.id;
  ui.route = "workspace";
  ui.previewMode = "single";
  ui.inspectorTab = "info";
  localStorage.setItem("genimage.inspectorTab", "info");
  saveWorkspaceTabs();
  render();
}

async function activateWorkspaceTab(tabID) {
  const tab = ui.workspaceTabs.find((item) => item.id === tabID);
  if (!tab) return;
  ui.activeWorkspaceTabID = tab.id;
  ui.route = "workspace";
  ui.previewMode = "single";
  ui.inspectorTab = "info";
  localStorage.setItem("genimage.inspectorTab", "info");
  saveWorkspaceTabs();
  render();
  if (tab.selectedAssetID) {
    await invoke("selectAsset", { assetID: tab.selectedAssetID });
  }
}

async function closeWorkspaceTab(tabID) {
  if (ui.workspaceTabs.length <= 1 || isInferenceBusy(state)) return;
  const index = ui.workspaceTabs.findIndex((tab) => tab.id === tabID);
  if (index < 0) return;

  const closingTab = ui.workspaceTabs[index];
  const replacementTab = ui.workspaceTabs[index + 1] || ui.workspaceTabs[index - 1];
  await invoke("closeWorkspaceProject", { assetIDs: closingTab.assetIDs });

  ui.workspaceTabs.splice(index, 1);
  const closedActiveTab = ui.activeWorkspaceTabID === tabID;
  if (closedActiveTab) ui.activeWorkspaceTabID = replacementTab.id;
  if (ui.renameWorkspaceTabID === tabID) closeWorkspaceTabRename(false);
  saveWorkspaceTabs();
  render();

  if (closedActiveTab && replacementTab.selectedAssetID) {
    await invoke("selectAsset", { assetID: replacementTab.selectedAssetID });
  }
}

function openWorkspaceTabRename(tabID) {
  const tab = ui.workspaceTabs.find((item) => item.id === tabID);
  if (!tab) return;
  ui.renameWorkspaceTabID = tab.id;
  ui.renameWorkspaceTabValue = tab.name;
  render();
  queueMicrotask(() => {
    const input = root.querySelector('[data-ui-field="workspaceTabName"]');
    input?.focus({ preventScroll: true });
    input?.select?.();
  });
}

function closeWorkspaceTabRename(shouldRender = true) {
  ui.renameWorkspaceTabID = null;
  ui.renameWorkspaceTabValue = "";
  if (shouldRender) render();
}

function saveWorkspaceTabRename() {
  const tab = ui.workspaceTabs.find((item) => item.id === ui.renameWorkspaceTabID);
  const name = ui.renameWorkspaceTabValue.trim();
  if (tab && name) tab.name = name;
  saveWorkspaceTabs();
  closeWorkspaceTabRename();
}

function setActiveTabSelection(assetID, event) {
  const tab = activeWorkspaceTab();
  if (!tab || !tab.assetIDs.includes(assetID)) return null;
  const asset = state.assets.find((item) => item.id === assetID);
  if (!asset) return null;

  const isImage = isImageAsset(asset);
  const additive = Boolean(event?.metaKey || event?.ctrlKey);
  const rangeSelection = Boolean(event?.shiftKey && isImage);
  let selectedAssetIDs = (tab.selectedAssetIDs || []).filter((id) => isImageAssetID(id));

  if (rangeSelection) {
    const imageIDs = tab.assetIDs.filter((id) => isImageAssetID(id));
    const anchorID = imageIDs.includes(tab.selectionAnchorID)
      ? tab.selectionAnchorID
      : imageIDs.includes(tab.selectedAssetID)
        ? tab.selectedAssetID
        : assetID;
    const start = imageIDs.indexOf(anchorID);
    const end = imageIDs.indexOf(assetID);
    const rangeIDs = imageIDs.slice(Math.min(start, end), Math.max(start, end) + 1);
    selectedAssetIDs = additive
      ? [...selectedAssetIDs, ...rangeIDs.filter((id) => !selectedAssetIDs.includes(id))]
      : rangeIDs;
  } else if (additive && isImage) {
    if (!selectedAssetIDs.length && isImageAssetID(tab.selectedAssetID)) {
      selectedAssetIDs.push(tab.selectedAssetID);
    }
    const index = selectedAssetIDs.indexOf(assetID);
    if (index >= 0 && selectedAssetIDs.length > 1) {
      selectedAssetIDs.splice(index, 1);
    } else if (index < 0) {
      selectedAssetIDs.push(assetID);
    }
    tab.selectionAnchorID = assetID;
  } else {
    selectedAssetIDs = isImage ? [assetID] : [];
    tab.selectionAnchorID = isImage ? assetID : null;
  }

  tab.selectedAssetIDs = selectedAssetIDs;
  tab.selectedAssetID = selectedAssetIDs.includes(assetID) || !isImage
    ? assetID
    : selectedAssetIDs.at(-1) || assetID;
  saveWorkspaceTabs();
  return tab.selectedAssetID;
}

function isImageAssetID(assetID) {
  return state.assets.some((asset) => asset.id === assetID && isImageAsset(asset));
}

function isImageAsset(asset) {
  return !["generatedVideo", "generatedAudio"].includes(asset.kind);
}

function isPreviewZoomDisabled() {
  if (!state || ui.generationType === "music") return true;
  const workspaceState = workspaceStateForActiveTab(state);
  return workspaceState.assets.some(
    (asset) => asset.id === workspaceState.selectedAssetID && asset.kind === "generatedAudio",
  );
}

function selectedVideoSourceAssetIDs() {
  const workspaceState = workspaceStateForActiveTab(state);
  const imageIDs = new Set(
    workspaceState.assets
      .filter(isImageAsset)
      .map((asset) => asset.id),
  );
  const selected = (workspaceState.selectedAssetIDs || []).filter((id) => imageIDs.has(id));
  if (selected.length) return selected;
  return imageIDs.has(workspaceState.selectedAssetID) ? [workspaceState.selectedAssetID] : [];
}

function replacementAssetIDAfterRemoval(assetID) {
  const tab = activeWorkspaceTab();
  if (!tab) return null;
  if (tab.selectedAssetID !== assetID) return tab.selectedAssetID;
  const index = tab.assetIDs.indexOf(assetID);
  if (index < 0) return null;
  return tab.assetIDs[index + 1] || tab.assetIDs[index - 1] || null;
}

function removeAssetFromWorkspaceTabs(assetID, activeReplacementID) {
  ui.workspaceTabs.forEach((tab) => {
    tab.assetIDs = tab.assetIDs.filter((id) => id !== assetID);
    tab.selectedAssetIDs = (tab.selectedAssetIDs || []).filter((id) => id !== assetID);
    if (tab.selectionAnchorID === assetID) {
      tab.selectionAnchorID = tab.selectedAssetIDs.at(-1) || null;
    }
    if (tab.selectedAssetID !== assetID) return;
    tab.selectedAssetID = tab.id === ui.activeWorkspaceTabID
      ? activeReplacementID
      : tab.assetIDs.at(-1) || null;
  });
  saveWorkspaceTabs();
}

function invokeTrackedOutput(method, params, action) {
  const pending = {
    action,
    tabID: ui.activeWorkspaceTabID,
    jobID: null,
    expiresAt: Date.now() + 60 * 60 * 1000,
  };
  pendingOutputs.push(pending);
  const request = params === undefined ? invoke(method) : invoke(method, params);
  return request.catch((error) => {
    const index = pendingOutputs.indexOf(pending);
    if (index >= 0) pendingOutputs.splice(index, 1);
    throw error;
  });
}

function bindPendingOutputJobs(nextState) {
  const previousJobIDs = new Set((state?.jobs || []).map((job) => job.id));
  nextState.jobs.forEach((job) => {
    if (previousJobIDs.has(job.id)) return;
    const pending = pendingOutputs.find((item) => item.action === job.action && !item.jobID);
    if (pending) pending.jobID = job.id;
  });
}

function takePendingOutputTab(action, nextState) {
  const now = Date.now();
  for (let index = pendingOutputs.length - 1; index >= 0; index -= 1) {
    if (pendingOutputs[index].expiresAt < now) pendingOutputs.splice(index, 1);
  }
  const completedJobIDs = new Set(
    nextState.jobs.filter((job) => job.state === "completed").map((job) => job.id),
  );
  let index = pendingOutputs.findIndex(
    (pending) => pending.action === action && pending.jobID && completedJobIDs.has(pending.jobID),
  );
  if (index < 0) index = pendingOutputs.findIndex((pending) => pending.action === action);
  if (index < 0) return null;
  return pendingOutputs.splice(index, 1)[0].tabID;
}

function clampPreviewZoom(value) {
  return Math.round(Math.min(2.5, Math.max(0.25, value)) * 100) / 100;
}

function quantizeDimension(value) {
  return Math.min(4096, Math.max(64, Math.round(value / 16) * 16));
}

function defaultVideoDimensionsForAspect(ratioWidth, ratioHeight) {
  const anchor = ratioWidth >= ratioHeight ? "width" : "height";
  return dimensionsForAspect(anchor, 1280, ratioWidth, ratioHeight);
}

function dimensionsForAspect(anchor, value, ratioWidth, ratioHeight) {
  const safeWidth = Math.max(1, ratioWidth);
  const safeHeight = Math.max(1, ratioHeight);
  if (anchor === "height") {
    let height = quantizeDimension(value);
    const derivedWidth = height * safeWidth / safeHeight;
    let width;
    if (derivedWidth < 64) {
      width = 64;
      height = quantizeDimension(width * safeHeight / safeWidth);
    } else if (derivedWidth > 4096) {
      width = 4096;
      height = quantizeDimension(width * safeHeight / safeWidth);
    } else {
      width = quantizeDimension(derivedWidth);
    }
    return { width, height };
  }

  let width = quantizeDimension(value);
  const derivedHeight = width * safeHeight / safeWidth;
  let height;
  if (derivedHeight < 64) {
    height = 64;
    width = quantizeDimension(height * safeWidth / safeHeight);
  } else if (derivedHeight > 4096) {
    height = 4096;
    width = quantizeDimension(height * safeWidth / safeHeight);
  } else {
    height = quantizeDimension(derivedHeight);
  }
  return { width, height };
}

function updateResolutionControls(dimensions, outputKind) {
  ["width", "height"].forEach((field) => {
    const slider = root.querySelector(
      `[data-dimension-field="${field}"][data-output-kind="${outputKind}"]`,
    );
    const output = root.querySelector(
      `[data-dimension-value="${field}"][data-output-kind="${outputKind}"]`,
    );
    if (slider) slider.value = dimensions[field];
    if (output) output.textContent = `${dimensions[field]} px`;
  });
  const summary = root.querySelector(`[data-resolution-summary][data-output-kind="${outputKind}"]`);
  if (summary) summary.textContent = `${dimensions.width} × ${dimensions.height} px`;
}

function stopPreviewPan(event) {
  if (!previewPan || previewPan.pointerID !== event.pointerId) return;
  previewPan.stage.classList.remove("panning");
  previewPan.stage.releasePointerCapture?.(event.pointerId);
  previewPan = null;
}

function renderSidebar() {
  return `
    <aside class="sidebar">
      <div class="brand">
        <img class="brand-mark" src="./assets/GenImage-AppIcon.png" alt="" aria-hidden="true" />
        <div class="brand-copy"><strong>${t("brand.name")}</strong><span>${t("brand.subtitle")}</span></div>
      </div>
      <nav class="sidebar-nav">
        ${navButton("workspace", "▦", t("nav.workspace"))}
        ${navButton("models", "⬡", t("nav.models"))}
        ${navButton("profiles", "⇄", t("nav.profiles"))}
        ${ui.profileHintVisible ? `
          <div class="profile-nav-hint" role="status">
            <span class="profile-nav-hint-icon" aria-hidden="true">✦</span>
            <span>${t("profile.firstUseHint")}</span>
            <button class="profile-nav-hint-dismiss" data-action="dismissProfileHint" aria-label="${t("profile.dismissHint")}" title="${t("profile.dismissHint")}">×</button>
          </div>
        ` : ""}
        ${navButton("downloads", "⇩", t("nav.downloads"))}
        ${navButton("settings", "⚙", t("nav.settings"))}
      </nav>
      <section class="sidebar-tool-section">
        <div class="quick-tools sidebar-quick-tools">
          ${renderQuickTools(workspaceStateForActiveTab(state))}
        </div>
      </section>
      <div class="sidebar-spacer"></div>
      ${renderSystemMetrics()}
      <div class="sidebar-project">
        <div class="sidebar-project-copy">
          <span>${t("sidebar.currentProject")}</span>
          <strong>${escapeHTML(state.projectName)}</strong>
          <span>${t("sidebar.assetsJobs", { assets: state.assets.length, jobs: state.jobs.filter((job) => ["queued", "running", "cancelling"].includes(job.state)).length })}</span>
        </div>
        <button
          class="icon-button compact memory-release-button"
          data-action="releaseMemory"
          title="${state.isReleasingMemory ? t("memory.releasing") : t("memory.release")}"
          aria-label="${state.isReleasingMemory ? t("memory.releasing") : t("memory.release")}"
          ${isInferenceBusy(state) || state.isReleasingMemory ? "disabled" : ""}
        >
          ${state.isReleasingMemory
            ? `<span class="button-spinner" aria-hidden="true"></span>`
            : `<svg class="memory-release-icon" viewBox="0 0 24 24" aria-hidden="true">
                <rect x="6" y="6.5" width="12" height="11" rx="2"></rect>
                <path d="M9 10h6M9 14h6M8 3.5v3M12 3.5v3M16 3.5v3M8 17.5v3M12 17.5v3M16 17.5v3M3 9h3M3 15h3M18 9h3M18 15h3"></path>
              </svg>`}
        </button>
      </div>
    </aside>
  `;
}

function renderRoute() {
  if (ui.route === "models") return renderModels(state, ui);
  if (ui.route === "profiles") return renderProfiles(state, ui);
  if (ui.route === "downloads") return renderDownloads(state);
  if (ui.route === "settings") return renderSettings(state, ui);
  return renderWorkspace(workspaceStateForActiveTab(state), ui);
}

function renderUpdateBanner() {
  const update = state.availableUpdate;
  if (!update) return "";
  return `<aside class="update-banner" role="status">
    <span class="update-banner-icon" aria-hidden="true">⬆</span>
    <div class="update-banner-copy">
      <strong>${t("update.available", { version: update.latestVersion })}</strong>
      <span>${t("update.current", { version: update.currentVersion })} · ${escapeHTML(update.releaseName)}</span>
    </div>
    <button class="update-banner-action" data-action="openAvailableUpdate">${t("update.openRelease")}</button>
    <button
      class="update-banner-dismiss"
      data-action="dismissAvailableUpdate"
      title="${t("update.dismiss")}"
      aria-label="${t("update.dismiss")}"
    >×</button>
  </aside>`;
}

function renderSystemMetrics() {
  const metrics = state.systemMetrics || {};
  return `<section class="sidebar-system-metrics" aria-label="${t("metrics.title")}">
    ${metricRow("ram", t("metrics.ram"), metrics.ramUsage)}
    ${metricRow("gpu", t("metrics.gpu"), metrics.gpuUsage)}
  </section>`;
}

function metricRow(key, label, value) {
  const metric = metricDisplay(value);
  return `<div class="metric-row metric-level-${metric.level}" data-metric-key="${key}">
    <div class="metric-label"><span>${label}</span><strong>${metric.label}</strong></div>
    <progress value="${metric.percent}" max="100" aria-label="${label}"></progress>
  </div>`;
}

function updateSystemMetricsDOM() {
  const metrics = state.systemMetrics || {};
  [["ram", metrics.ramUsage], ["gpu", metrics.gpuUsage]].forEach(([key, value]) => {
    const row = root.querySelector(`[data-metric-key="${key}"]`);
    if (!row) return;
    const metric = metricDisplay(value);
    const label = row.querySelector("strong");
    const progress = row.querySelector("progress");
    row.classList.remove("metric-level-low", "metric-level-medium", "metric-level-high", "metric-level-unavailable");
    row.classList.add(`metric-level-${metric.level}`);
    if (label) label.textContent = metric.label;
    if (progress) progress.value = metric.percent;
  });
}

function metricDisplay(value) {
  const valid = Number.isFinite(value);
  const percent = valid ? Math.round(Math.min(1, Math.max(0, value)) * 100) : 0;
  const level = !valid ? "unavailable" : percent >= 80 ? "high" : percent >= 60 ? "medium" : "low";
  return { percent, level, label: valid ? `${percent}%` : t("metrics.unavailable") };
}

function modelProgressOnlyChanged(previous, next) {
  if (!previous || !["models", "profiles", "downloads"].includes(ui.route)) return false;
  if (previous.models.length !== next.models.length) return false;
  const sameModelContent = previous.models.every(({ descriptor, installation }, index) => {
    const nextModel = next.models[index];
    if (!nextModel || descriptor.id !== nextModel.descriptor.id) return false;
    return JSON.stringify({ descriptor, phase: installation.phase, error: installation.errorMessage })
      === JSON.stringify({
        descriptor: nextModel.descriptor,
        phase: nextModel.installation.phase,
        error: nextModel.installation.errorMessage,
      });
  });
  if (!sameModelContent) return false;
  return contentSignature({ ...previous, models: [] })
    === contentSignature({ ...next, models: [] });
}

function refreshModelProgressDOM() {
  state.models.forEach(({ descriptor, installation }) => {
    const card = root.querySelector(`[data-model-card="${CSS.escape(descriptor.id)}"]`);
    if (!card) return;
    const progress = card.querySelector("[data-model-progress]");
    const phase = card.querySelector("[data-model-phase]");
    const size = card.querySelector("[data-model-size]");
    const label = card.querySelector("[data-model-percent]");
    if (progress) progress.value = installation.progress;
    if (phase) phase.textContent = phaseLabel(installation.phase);
    if (size) size.textContent = `${gigabytes(installation.downloadedGB)} / ${gigabytes(descriptor.approximateDownloadGB)}`;
    if (label) label.textContent = percent(installation.progress);
  });
}

function contentSignature(value) {
  const {
    systemMetrics: _systemMetrics,
    statusMessage: _statusMessage,
    ...content
  } = value;
  content.jobs = value.jobs.map(({ id, action, state }) => ({ id, action, state }));
  if (ui.route === "workspace") {
    content.models = value.models.map(({ descriptor, installation }) => ({
      descriptor,
      installationGroup: installation.phase === "installed"
        ? "installed"
        : installation.phase === "notInstalled"
          ? "notInstalled"
          : "active",
    }));
  }
  return JSON.stringify(sortForSignature(content));
}

function contentSignatureWithoutEditableSettings(value) {
  return contentSignature({
    ...value,
    recipe: {},
    videoOutputSettings: {},
    musicOutputSettings: {},
  });
}

function sortForSignature(value) {
  if (Array.isArray(value)) return value.map(sortForSignature);
  if (!value || typeof value !== "object") return value;
  return Object.keys(value)
    .sort()
    .reduce((result, key) => {
      result[key] = sortForSignature(value[key]);
      return result;
    }, {});
}

function renderWorkspaceTabRenameDialog() {
  if (!ui.renameWorkspaceTabID) return "";
  return `<div class="dialog-backdrop">
    <section class="paste-dialog" role="dialog" aria-modal="true" aria-labelledby="rename-tab-dialog-title">
      <h2 id="rename-tab-dialog-title">${t("workspace.renameTab")}</h2>
      <label class="dialog-field">${t("workspace.tabNameLabel")}
        <input
          class="field"
          data-ui-field="workspaceTabName"
          value="${escapeHTML(ui.renameWorkspaceTabValue)}"
          maxlength="80"
        />
      </label>
      <div class="dialog-actions">
        <button class="secondary-button" data-action="workspaceRenameCancel">${t("common.cancel")}</button>
        <button class="primary-button" data-action="workspaceRenameSave">${t("common.save")}</button>
      </div>
    </section>
  </div>`;
}

function renderPasteDialog() {
  if (!ui.pasteDialogOpen || !pasteState.image) return "";
  const inferenceDisabledAttribute = isInferenceBusy(state)
    ? "disabled aria-busy=\"true\""
    : "";
  return `<div class="dialog-backdrop">
    <section class="paste-dialog" role="dialog" aria-modal="true" aria-labelledby="paste-dialog-title">
      <h2 id="paste-dialog-title">${t("clipboard.imageDetected")}</h2>
      <p>${t("clipboard.describeQuestion", { name: escapeHTML(pasteState.image.name) })}</p>
      <div class="dialog-actions">
        <button class="secondary-button" data-action="pasteImageDecision" data-describe="false">${t("common.no")}</button>
        <button class="primary-button" data-action="pasteImageDecision" data-describe="true" ${inferenceDisabledAttribute}>${t("common.yes")}</button>
      </div>
    </section>
  </div>`;
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(reader.result));
    reader.addEventListener("error", () => reject(reader.error || new Error(t("import.imageReadFailed"))));
    reader.readAsDataURL(file);
  });
}

function isSupportedImageFile(file) {
  return file?.type?.startsWith("image/") || SUPPORTED_IMAGE_FILE_PATTERN.test(file?.name || "");
}

function dataTransferHasFiles(dataTransfer) {
  return Array.from(dataTransfer?.types || []).includes("Files");
}

async function importDroppedImageFiles(imageFiles) {
  ui.route = "workspace";
  ui.previewMode = "single";
  setInspectorTab("info");
  for (const imageFile of imageFiles) {
    await invoke("pasteImage", {
      dataURL: await readFileAsDataURL(imageFile),
      name: imageFile.name,
      describe: false,
    });
  }
}

function showImageDropOverlay() {
  if (imageDropOverlay) return;
  const overlay = document.createElement("div");
  overlay.className = "image-drop-overlay";
  overlay.setAttribute("role", "status");
  overlay.innerHTML = `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3.5" y="4.5" width="17" height="15" rx="2.5"></rect>
      <circle cx="8.5" cy="9" r="1.5"></circle>
      <path d="m5.5 17 4.5-4.5 3.2 3.1 2.1-2.1 3.2 3.5"></path>
      <path d="M16.5 3v5M14 5.5h5"></path>
    </svg>
    <strong></strong>
    <span></span>
  `;
  overlay.querySelector("strong").textContent = t("import.dropImages");
  overlay.querySelector("span").textContent = t("import.supportedImageFormats");
  document.body.append(overlay);
  imageDropOverlay = overlay;
}

function hideImageDropOverlay() {
  imageDropOverlay?.remove();
  imageDropOverlay = null;
}

function resetImageDragState() {
  imageDragDepth = 0;
  hideImageDropOverlay();
}

async function handleClipboardImage(dataURL, name) {
  const image = {
    dataURL,
    name: typeof name === "string" && name.trim() ? name.trim() : t("clipboard.pastedImage"),
  };
  if (!hasEnabledImageToTextProfile()) {
    pasteState.image = null;
    ui.pasteDialogOpen = false;
    ui.route = "workspace";
    ui.previewMode = "single";
    setInspectorTab("info");
    await invoke("pasteImage", { dataURL: image.dataURL, describe: false });
    return;
  }
  pasteState.image = image;
  ui.pasteDialogOpen = true;
  render();
}

function hasEnabledImageToTextProfile() {
  if (!state || !Array.isArray(state.profiles)) return false;
  const activeProfileID = state.activeProfileIDs?.imageToText;
  if (!activeProfileID) return false;
  const disabled = new Set(state.disabledProfileIDs || []);
  return state.profiles.some(
    (profile) =>
      profile.id === activeProfileID
      && profile.capability === "imageToText"
      && !disabled.has(profile.id),
  );
}

function navButton(route, icon, title) {
  return `<button class="nav-button ${ui.route === route ? "active" : ""}" data-action="navigate" data-route="${route}"><span>${icon}</span>${title}</button>`;
}

function dismissProfileHint() {
  if (!ui.profileHintVisible) return;
  ui.profileHintVisible = false;
  localStorage.setItem("genimage.profileHintSeen", "true");
}

function renderToast() {
  if (!state.statusMessage) return "";
  return `<div class="toast"><span style="color:var(--positive)">●</span><span>${escapeHTML(state.statusMessage)}</span><button data-action="clearStatus">×</button></div>`;
}

function refreshToastDOM() {
  const currentToast = root.querySelector(".toast");
  const markup = renderToast();
  if (!markup) {
    currentToast?.remove();
    return;
  }
  const template = document.createElement("template");
  template.innerHTML = markup.trim();
  const nextToast = template.content.firstElementChild;
  if (!nextToast) return;
  if (currentToast) currentToast.replaceWith(nextToast);
  else root.append(nextToast);
}

function scheduleStatusMessageDismiss(message) {
  clearTimeout(statusMessageTimer);
  statusMessageTimer = null;
  if (!message) return;

  statusMessageTimer = setTimeout(() => {
    statusMessageTimer = null;
    if (state?.statusMessage !== message) return;
    invoke("clearStatus").catch(showBridgeError);
  }, STATUS_MESSAGE_DURATION_MS);
}

function syncRecipe() {
  clearTimeout(recipeTimer);
  return invoke("updateRecipe", {
    prompt: state.recipe.prompt,
    negativePrompt: state.recipe.negativePrompt,
    width: state.recipe.width,
    height: state.recipe.height,
    steps: state.recipe.steps,
    outputCount: state.recipe.outputCount,
    seed: String(state.recipe.seed),
    loraID: state.recipe.loraID,
    loraScale: state.recipe.loraScale,
  });
}

function syncVideoOutputSettings() {
  clearTimeout(recipeTimer);
  return invoke("updateVideoOutputSettings", {
    width: state.videoOutputSettings.width,
    height: state.videoOutputSettings.height,
    steps: state.videoOutputSettings.steps,
    outputCount: state.videoOutputSettings.outputCount,
    frameCount: state.videoOutputSettings.frameCount,
    frameRate: state.videoOutputSettings.frameRate,
    seed: String(state.videoOutputSettings.seed),
  });
}

function syncMusicOutputSettings() {
  clearTimeout(recipeTimer);
  return invoke("updateMusicOutputSettings", {
    prompt: state.musicOutputSettings.prompt,
    lyrics: state.musicOutputSettings.lyrics,
    style: state.musicOutputSettings.style,
    durationSeconds: state.musicOutputSettings.durationSeconds,
    steps: state.musicOutputSettings.steps,
    seed: String(state.musicOutputSettings.seed),
    format: state.musicOutputSettings.format,
  });
}

function editableFieldFor(element) {
  const definitions = [
    ["recipeField", "recipe", "recipe"],
    ["videoField", "videoOutputSettings", "video"],
    ["musicField", "musicOutputSettings", "music"],
  ];
  for (const [datasetKey, stateKey, syncKind] of definitions) {
    const field = element?.dataset?.[datasetKey];
    if (field) {
      return {
        element,
        field,
        key: `${stateKey}:${field}`,
        stateKey,
        syncKind,
      };
    }
  }
  return null;
}

function updateEditableFieldValue(editableField, value) {
  const settings = state?.[editableField.stateKey];
  if (settings) settings[editableField.field] = value;
}

function syncEditableField(editableField) {
  if (editableField.syncKind === "video") return syncVideoOutputSettings();
  if (editableField.syncKind === "music") return syncMusicOutputSettings();
  return syncRecipe();
}

function scheduleDeferredRender() {
  if (!renderDeferredDuringEditing) return;
  clearTimeout(deferredRenderTimer);
  deferredRenderTimer = setTimeout(() => {
    deferredRenderTimer = null;
    if (activeEditableField || composingEditableField) return;
    renderDeferredDuringEditing = false;
    render();
  }, 0);
}

function normalizeLegacyPromptTab(value) {
  if (value === "output") return "imageOutput";
  return PROMPT_TABS.has(value) ? value : "prompt";
}

function normalizeGenerationType(value, promptTab = "prompt") {
  if (GENERATION_TYPES.has(value)) return value;
  if (promptTab === "videoOutput") return "video";
  if (promptTab === "lyrics" || promptTab === "musicOutput") return "music";
  return "image";
}

function normalizePromptTab(value, generationType) {
  const promptTab = normalizeLegacyPromptTab(value);
  const normalizedGenerationType = normalizeGenerationType(generationType, promptTab);
  if (PROMPT_TABS_BY_GENERATION_TYPE[normalizedGenerationType].has(promptTab)) return promptTab;
  if (promptTab.endsWith("Output")) return `${normalizedGenerationType}Output`;
  return normalizedGenerationType === "music" ? "lyrics" : "prompt";
}

function setGenerationType(value, preferredPromptTab = ui.promptTab) {
  const generationType = normalizeGenerationType(value);
  ui.generationType = generationType;
  ui.promptTab = normalizePromptTab(preferredPromptTab, generationType);
  localStorage.setItem("genimage.generationType", generationType);
  localStorage.setItem("genimage.promptTab", ui.promptTab);
}

function saveProfile(profileID) {
  const card = root.querySelector(`[data-profile-card="${CSS.escape(profileID)}"]`);
  if (!card) return Promise.resolve();
  const field = (name) => card.querySelector(`[data-profile-field="${name}"]`)?.value || "";
  const loras = Array.from(card.querySelectorAll("[data-profile-lora-row]"))
    .map((row) => {
      const loraField = (name) => row.querySelector(`[data-profile-lora-field="${name}"]`)?.value || "";
      return {
        modelID: loraField("modelID").trim(),
        scale: Number(loraField("scale") || 1),
        conditioning: loraField("conditioning") || null,
        conditioningScale: Number(loraField("conditioningScale") || 1),
      };
    })
    .filter(({ modelID }) => modelID);
  return invoke("updateProfile", {
    profileID,
    name: field("name"),
    modelID: field("modelID"),
    modelRevision: field("modelRevision"),
    architecture: field("architecture"),
    loras,
  });
}

function captureViewState() {
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

function restoreViewState(viewState) {
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

function showBridgeError(error) {
  console.error(error);
  const message = error instanceof Error
    ? error.message
    : typeof error === "string"
      ? error
      : error?.message || error?.error || "Native command failed.";
  if (!state || !message) return;
  state = { ...state, statusMessage: message };
  scheduleStatusMessageDismiss(message);
  refreshToastDOM();
}
