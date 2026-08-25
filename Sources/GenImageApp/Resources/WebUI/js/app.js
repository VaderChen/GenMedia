import { invoke, onClipboardImage, onState } from "./bridge.js";
import {
  refreshModelProgressDOM,
  refreshToastDOM,
  renderPasteDialog,
  renderRoute,
  renderSidebar,
  renderSmallOutputWarningDialog,
  renderToast,
  renderUpdateBanner,
  renderWorkspaceTabRenameDialog,
  updateSystemMetricsDOM,
} from "./chrome.js";
import { escapeHTML } from "./format.js";
import {
  defaultVideoDimensionsForAspect,
  dimensionsForAspect,
  sourceImageDimensions,
  undersizedImageOutput,
} from "./geometry.js";
import { getLocale, setLocale, t } from "./i18n.js";
import { appendProfileLoRARow, removeProfileLoRARow } from "./profiles.js";
import { withPreservedView } from "./render-preservation.js";
import { getTheme, setTheme } from "./themes.js";
import {
  activeWorkspaceTab,
  dropPendingOutput,
  formatWorkspaceTabName,
  isImageAsset,
  isImageAssetID,
  loadWorkspaceTabs,
  makeWorkspaceTab,
  reconcileWorkspaceTabs,
  removeAssetFromWorkspaceTabs,
  replacementAssetIDAfterRemoval,
  saveWorkspaceTabs,
  setActiveTabSelection,
  trackPendingOutput,
  workspaceStateForActiveTab,
  workspaceTabOwningAsset,
} from "./workspace-tabs.js";
import {
  isInferenceBusy,
  refreshJobsPanel,
  refreshJobTimings,
  renderCreationPanel,
  renderQuickTools,
  renderWorkspace,
  selectedSourceImage,
} from "./workspace.js";

const root = document.querySelector("#app");
const STATUS_MESSAGE_DURATION_MS = 5_000;
const JOB_TIMING_REFRESH_MS = 1_000;
const PROMPT_TABS = new Set(["prompt", "negative", "lyrics", "imageOutput", "videoOutput", "musicOutput"]);
const GENERATION_TYPES = new Set(["image", "video", "music"]);
const PROMPT_TABS_BY_GENERATION_TYPE = {
  image: new Set(["prompt", "negative", "imageOutput"]),
  video: new Set(["prompt", "negative", "videoOutput"]),
  music: new Set(["prompt", "lyrics", "musicOutput"]),
};
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
const pasteState = { image: null };
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
  smallOutputWarning: null,
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
    reconcileWorkspaceTabs(ui, state, nextState);
    state = {
      ...nextState,
      recipe: localRecipe,
      videoOutputSettings: localVideoOutputSettings,
      musicOutputSettings: localMusicOutputSettings,
    };
    stateContentSignature = nextContentSignature;
    renderDeferredDuringEditing = renderDeferredDuringEditing || contentChanged;
    if (statusMessageChanged) scheduleStatusMessageDismiss(nextState.statusMessage);
    refreshModelProgressDOM(state, root);
    updateSystemMetricsDOM(state, root);
    refreshJobsPanel(workspaceStateForActiveTab(ui, state), root);
    refreshToastDOM(state, root);
    return;
  }
  if (state && modelProgressOnlyChanged(state, nextState)) {
    state = nextState;
    stateContentSignature = nextContentSignature;
    refreshModelProgressDOM(state, root);
    updateSystemMetricsDOM(state, root);
    refreshJobsPanel(workspaceStateForActiveTab(ui, state), root);
    if (statusMessageChanged) scheduleStatusMessageDismiss(nextState.statusMessage);
    refreshToastDOM(state, root);
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
    updateSystemMetricsDOM(state, root);
    refreshJobsPanel(workspaceStateForActiveTab(ui, state), root);
    refreshToastDOM(state, root);
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
  reconcileWorkspaceTabs(ui, state, nextState);
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
      case "imageToImage": {
        const smallOutput = undersizedImageOutput(state);
        if (smallOutput && !smallOutputWarningAcknowledged) {
          ui.smallOutputWarning = smallOutput;
          render();
          break;
        }
        await runImageToImage();
        break;
      }
      case "smallOutputCancel":
        ui.smallOutputWarning = null;
        render();
        break;
      case "smallOutputContinue":
        smallOutputWarningAcknowledged = true;
        ui.smallOutputWarning = null;
        render();
        await runImageToImage();
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
          const primaryAssetID = setActiveTabSelection(ui, state, target.dataset.assetId, event);
          if (!primaryAssetID) break;
          setInspectorTab("info");
          await invoke("selectAsset", { assetID: primaryAssetID });
        }
        break;
      case "removeAsset": {
        const assetID = target.dataset.assetId;
        const replacementAssetID = replacementAssetIDAfterRemoval(ui, state, assetID);
        await invoke("removeAsset", { assetID, replacementAssetID });
        removeAssetFromWorkspaceTabs(ui, assetID, replacementAssetID);
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
        const sourceAsset = selectedSourceImage(workspaceStateForActiveTab(ui, state));
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
  withPreservedView(root, () => {
    root.innerHTML = `
      <div class="app-shell">
        ${renderSidebar(state, ui)}
        <div class="main-view">
          ${renderUpdateBanner(state)}
          <div class="route-view">${renderRoute(state, ui)}</div>
        </div>
      </div>
      ${renderWorkspaceTabRenameDialog(ui)}
      ${renderPasteDialog(state, ui, pasteState)}
      ${renderSmallOutputWarningDialog(ui)}
      ${renderToast(state)}
    `;
  });
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
  template.innerHTML = renderCreationPanel(workspaceStateForActiveTab(ui, state), ui).trim();
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

function addWorkspaceTab() {
  const tab = makeWorkspaceTab();
  ui.workspaceTabs.push(tab);
  ui.activeWorkspaceTabID = tab.id;
  ui.route = "workspace";
  ui.previewMode = "single";
  ui.inspectorTab = "info";
  localStorage.setItem("genimage.inspectorTab", "info");
  saveWorkspaceTabs(ui);
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
  saveWorkspaceTabs(ui);
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
  saveWorkspaceTabs(ui);
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
  saveWorkspaceTabs(ui);
  closeWorkspaceTabRename();
}

function isPreviewZoomDisabled() {
  if (!state || ui.generationType === "music") return true;
  const workspaceState = workspaceStateForActiveTab(ui, state);
  return workspaceState.assets.some(
    (asset) => asset.id === workspaceState.selectedAssetID && asset.kind === "generatedAudio",
  );
}

function selectedVideoSourceAssetIDs() {
  const workspaceState = workspaceStateForActiveTab(ui, state);
  const imageIDs = new Set(
    workspaceState.assets
      .filter(isImageAsset)
      .map((asset) => asset.id),
  );
  const selected = (workspaceState.selectedAssetIDs || []).filter((id) => imageIDs.has(id));
  if (selected.length) return selected;
  return imageIDs.has(workspaceState.selectedAssetID) ? [workspaceState.selectedAssetID] : [];
}

function invokeTrackedOutput(method, params, action) {
  const pending = trackPendingOutput(ui, action);
  const request = params === undefined ? invoke(method) : invoke(method, params);
  return request.catch((error) => {
    dropPendingOutput(pending);
    throw error;
  });
}

function clampPreviewZoom(value) {
  return Math.round(Math.min(2.5, Math.max(0.25, value)) * 100) / 100;
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

// Warn at most once per session so deliberate batches of small outputs are not nagged.
let smallOutputWarningAcknowledged = false;

async function runImageToImage() {
  await syncRecipe();
  ui.route = "workspace";
  ui.previewMode = "single";
  setInspectorTab("jobs");
  await invokeTrackedOutput("imageToImage", undefined, "imageToImage");
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

function dismissProfileHint() {
  if (!ui.profileHintVisible) return;
  ui.profileHintVisible = false;
  localStorage.setItem("genimage.profileHintSeen", "true");
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
  refreshToastDOM(state, root);
}
