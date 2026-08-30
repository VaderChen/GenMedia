import { renderAutomaticFlow } from "./automatic-flow.js";
import { renderDownloads } from "./downloads.js";
import { escapeHTML, gigabytes, percent, phaseLabel } from "./format.js";
import { RECOMMENDED_MINIMUM_SIDE } from "./geometry.js";
import { t } from "./i18n.js";
import { renderModelDetails, renderModels } from "./models.js";
import { renderProfileDetails, renderProfiles } from "./profiles.js";
import { renderSettings } from "./settings.js";
import { workspaceStateForActiveTab } from "./workspace-tabs.js";
import { isInferenceBusy, renderQuickTools, renderWorkspace } from "./workspace.js";

// App 外框：側邊欄、路由切換、更新橫幅、系統資源列、對話框與 toast。
//
// 全部是純畫面函式，`state` 與 `ui` 由呼叫端傳入，不讀取任何模組層級的可變狀態，也不觸發重繪。

export function renderSidebar(state, ui) {
  const selectedWorkspace = state.workspaces?.find(
    (workspace) => workspace.id === state.selectedWorkspaceID,
  );
  const workspaceName = selectedWorkspace?.isDefault
    ? t("workspace.defaultWorkspace")
    : state.projectName;
  return `
    <aside class="sidebar">
      <div class="brand">
        <img class="brand-mark" src="./assets/GenImage-AppIcon.png" alt="" aria-hidden="true" />
        <div class="brand-copy"><strong>${t("brand.name")}</strong><span>${t("brand.subtitle")}</span></div>
      </div>
      <nav class="sidebar-nav">
        ${navButton(ui, "workspace", "▦", t("nav.workspace"))}
        ${navButton(ui, "automaticFlow", "⎇", t("nav.automaticFlow"))}
        ${navButton(ui, "profiles", "⇄", t("nav.profiles"))}
        ${ui.profileHintVisible ? `
          <div class="profile-nav-hint" role="status">
            <span class="profile-nav-hint-icon" aria-hidden="true">✦</span>
            <span>${t("profile.firstUseHint")}</span>
            <button class="profile-nav-hint-dismiss" data-action="dismissProfileHint" aria-label="${t("profile.dismissHint")}" title="${t("profile.dismissHint")}">×</button>
          </div>
        ` : ""}
        ${navButton(ui, "models", "⬡", t("nav.models"))}
        ${navButton(ui, "downloads", "⇩", t("nav.downloads"))}
        ${navButton(ui, "settings", "⚙", t("nav.settings"))}
      </nav>
      <section class="sidebar-tool-section">
        <div class="quick-tools sidebar-quick-tools">
          ${renderQuickTools(workspaceStateForActiveTab(ui, state))}
        </div>
      </section>
      <div class="sidebar-spacer"></div>
      ${renderSystemMetrics(state)}
      <div class="sidebar-project">
        <div class="sidebar-project-copy">
          <span>${t("sidebar.currentProject")}</span>
          <strong>${escapeHTML(workspaceName)}</strong>
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

function navButton(ui, route, icon, title) {
  return `<button class="nav-button ${ui.route === route ? "active" : ""}" data-action="navigate" data-route="${route}"><span>${icon}</span>${title}</button>`;
}

export function renderRoute(state, ui) {
  if (ui.route === "automaticFlow") return renderAutomaticFlow(state);
  if (ui.route === "models") return renderModels(state, ui);
  if (ui.route === "profiles") return renderProfiles(state, ui);
  if (ui.route === "downloads") return renderDownloads(state);
  if (ui.route === "settings") return renderSettings(state, ui);
  return renderWorkspace(workspaceStateForActiveTab(ui, state), ui);
}

export function renderUpdateBanner(state) {
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

function renderSystemMetrics(state) {
  const metrics = state.systemMetrics || {};
  return `<section class="sidebar-system-metrics" aria-label="${t("metrics.title")}">
    ${metricRow("ram", t("metrics.ram"), metrics.ramUsage)}
    ${metricRow("gpu", t("metrics.gpu"), metrics.gpuUsage)}
    ${metricRow("npu", t("metrics.npu"), metrics.npuUsage, t("metrics.npuNote"))}
  </section>`;
}

function metricRow(key, label, value, note = "") {
  const metric = metricDisplay(value);
  const title = note ? ` title="${escapeHTML(note)}"` : "";
  return `<div class="metric-row metric-level-${metric.level}" data-metric-key="${key}"${title}>
    <div class="metric-label"><span>${label}</span><strong>${metric.label}</strong></div>
    <progress value="${metric.percent}" max="100" aria-label="${label}"></progress>
  </div>`;
}

function metricDisplay(value) {
  const valid = Number.isFinite(value);
  const percent = valid ? Math.round(Math.min(1, Math.max(0, value)) * 100) : 0;
  const level = !valid ? "unavailable" : percent >= 80 ? "high" : percent >= 60 ? "medium" : "low";
  return { percent, level, label: valid ? `${percent}%` : t("metrics.unavailable") };
}

export function updateSystemMetricsDOM(state, root) {
  const metrics = state.systemMetrics || {};
  [["ram", metrics.ramUsage], ["gpu", metrics.gpuUsage], ["npu", metrics.npuUsage]].forEach(([key, value]) => {
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

export function refreshModelProgressDOM(state, root) {
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

export function renderWorkspaceTabRenameDialog(ui) {
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

export function renderWorkspaceCreateDialog(ui) {
  if (!ui.workspaceCreateDialogOpen) return "";
  return `<div class="dialog-backdrop">
    <section class="paste-dialog" role="dialog" aria-modal="true" aria-labelledby="create-workspace-dialog-title">
      <h2 id="create-workspace-dialog-title">${t("workspace.createWorkspace")}</h2>
      <label class="dialog-field">${t("workspace.workspaceName")}
        <input
          class="field"
          data-ui-field="workspaceName"
          value="${escapeHTML(ui.workspaceCreateName)}"
          maxlength="80"
        />
      </label>
      <div class="dialog-actions">
        <button class="secondary-button" data-action="workspaceCreateCancel">${t("common.cancel")}</button>
        <button
          class="primary-button"
          data-action="workspaceCreateSave"
          ${ui.workspaceCreateName.trim() ? "" : "disabled"}
        >${t("workspace.createAction")}</button>
      </div>
    </section>
  </div>`;
}

export function renderWorkspaceDeleteDialog(state, ui) {
  const workspace = state.workspaces?.find(
    (item) => item.id === ui.workspaceDeleteTargetID,
  );
  if (!workspace) return "";
  const name = workspace.isDefault ? t("workspace.defaultWorkspace") : workspace.name;
  return `<div class="dialog-backdrop">
    <section class="paste-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-workspace-dialog-title">
      <h2 id="delete-workspace-dialog-title">${t("workspace.deleteWorkspaceTitle")}</h2>
      <p>${escapeHTML(t("workspace.deleteWorkspaceMessage", { name }))}</p>
      <div class="dialog-actions">
        <button class="secondary-button" data-action="workspaceDeleteCancel">${t("common.cancel")}</button>
        <button class="danger-button" data-action="workspaceDeleteConfirm">${t("workspace.deleteAction")}</button>
      </div>
    </section>
  </div>`;
}

export function renderAssetRemovalDialog(state, ui) {
  const removal = ui.assetRemovalDialog;
  if (!removal) return "";
  const asset = state.assets.find((item) => item.id === removal.assetID);
  if (!asset) return "";
  return `<div class="dialog-backdrop">
    <section class="paste-dialog" role="alertdialog" aria-modal="true" aria-labelledby="asset-removal-dialog-title">
      <h2 id="asset-removal-dialog-title">${t("asset.removeDialogTitle")}</h2>
      <p>${t("asset.removeDialogMessage", { name: escapeHTML(asset.title) })}</p>
      <div class="dialog-actions">
        <button class="danger-button" data-action="assetRemovalDelete">${t("asset.deleteFile")}</button>
        <button class="secondary-button" data-action="assetRemovalCancel">${t("common.cancel")}</button>
        <button class="primary-button" data-action="assetRemovalRemove">${t("asset.removeFromWorkspace")}</button>
      </div>
    </section>
  </div>`;
}

export function renderAssetRenameDialog(state, ui) {
  const rename = ui.assetRenameDialog;
  if (!rename) return "";
  const asset = state.assets.find((item) => item.id === rename.assetID);
  if (!asset) return "";
  return `<div class="dialog-backdrop">
    <section class="paste-dialog" role="dialog" aria-modal="true" aria-labelledby="asset-rename-dialog-title">
      <h2 id="asset-rename-dialog-title">${t("asset.renameDialogTitle")}</h2>
      <p>${t("asset.renameDialogMessage", { name: escapeHTML(asset.fileName || asset.title) })}</p>
      <label class="dialog-field">${t("asset.fileName")}
        <input
          class="field"
          type="text"
          value="${escapeHTML(ui.assetRenameValue)}"
          data-asset-rename-input
          data-preserve-focus="asset-rename"
          autocomplete="off"
          spellcheck="false"
        />
      </label>
      <div class="dialog-actions">
        <button class="secondary-button" data-action="assetRenameCancel">${t("common.cancel")}</button>
        <button class="primary-button" data-action="assetRenameSave" ${ui.assetRenameValue.trim() ? "" : "disabled"}>${t("asset.rename")}</button>
      </div>
    </section>
  </div>`;
}

export function renderModelRemovalDialog(state, ui) {
  const removal = ui.modelRemovalDialog;
  if (!removal) return "";
  const item = state.models.find(({ descriptor }) => descriptor.id === removal.modelID);
  if (!item) return "";
  const location = modelLocation(item.descriptor);
  return `<div class="dialog-backdrop">
    <section class="paste-dialog" role="alertdialog" aria-modal="true" aria-labelledby="model-removal-dialog-title">
      <h2 id="model-removal-dialog-title">${t("model.removeDialogTitle")}</h2>
      <p>${t("model.removeDialogMessage", { name: escapeHTML(item.descriptor.displayName) })}</p>
      <p class="dialog-path"><strong>${t("model.removeDialogPath")}</strong><span>${escapeHTML(location)}</span></p>
      <p>${t("model.removeDialogIrreversible")}</p>
      <div class="dialog-actions">
        <button class="secondary-button" data-action="modelRemovalCancel">${t("common.cancel")}</button>
        <button class="danger-button" data-action="modelRemovalConfirm">${t("model.remove")}</button>
      </div>
    </section>
  </div>`;
}

export function renderCivitaiTokenDialog(state, ui) {
  const prompt = ui.civitaiTokenDialog;
  if (!prompt) return "";
  const item = state.models.find(({ descriptor }) => descriptor.id === prompt.modelID);
  const modelName = item?.descriptor.displayName || prompt.modelID;
  return `<div class="dialog-backdrop">
    <section class="paste-dialog" role="dialog" aria-modal="true" aria-labelledby="civitai-token-dialog-title">
      <h2 id="civitai-token-dialog-title">${t("civitai.downloadTokenTitle")}</h2>
      <p>${t("civitai.downloadTokenMessage", { name: escapeHTML(modelName) })}</p>
      <label class="dialog-field">${t("settings.civitaiToken")}
        <input
          class="field"
          type="password"
          value="${escapeHTML(ui.civitaiTokenValue)}"
          placeholder="${t("settings.civitaiTokenPlaceholder")}"
          data-civitai-token-prompt
          data-preserve-focus="civitai-download-token"
          autocomplete="off"
          spellcheck="false"
        />
      </label>
      <p class="dialog-note">${t("civitai.downloadTokenNote")}</p>
      <div class="dialog-actions">
        <button class="secondary-button" data-action="civitaiTokenCancel">${t("common.cancel")}</button>
        <button class="primary-button" data-action="civitaiTokenConfirm" ${ui.civitaiTokenValue.trim() ? "" : "disabled"}>${t("civitai.startDownload")}</button>
      </div>
    </section>
  </div>`;
}

export function renderHuggingFaceTokenDialog(state, ui) {
  const prompt = ui.huggingFaceTokenDialog;
  if (!prompt) return "";
  const item = state.models.find(({ descriptor }) => descriptor.id === prompt.modelID);
  const modelName = item?.descriptor.displayName || prompt.modelID;
  return `<div class="dialog-backdrop">
    <section class="paste-dialog" role="dialog" aria-modal="true" aria-labelledby="huggingface-token-dialog-title">
      <h2 id="huggingface-token-dialog-title">${t("huggingface.downloadTokenTitle")}</h2>
      <p>${t("huggingface.downloadTokenMessage", { name: escapeHTML(modelName) })}</p>
      <label class="dialog-field">${t("huggingface.tokenLabel")}
        <input
          class="field"
          type="password"
          value="${escapeHTML(ui.huggingFaceTokenValue)}"
          placeholder="${t("huggingface.tokenPlaceholder")}"
          data-huggingface-token-prompt
          data-preserve-focus="huggingface-download-token"
          autocomplete="off"
          spellcheck="false"
        />
      </label>
      <p class="dialog-note">${t("huggingface.downloadTokenNote")}</p>
      <div class="dialog-actions">
        <button class="secondary-button" data-action="huggingFaceTokenCancel">${t("common.cancel")}</button>
        <button class="primary-button" data-action="huggingFaceTokenConfirm" ${ui.huggingFaceTokenValue.trim() ? "" : "disabled"}>${t("huggingface.startDownload")}</button>
      </div>
    </section>
  </div>`;
}

function modelLocation(descriptor) {
  if (!descriptor.localURL) return descriptor.id;
  try {
    return decodeURIComponent(new URL(descriptor.localURL).pathname);
  } catch {
    return descriptor.localURL;
  }
}

export function renderSmallOutputWarningDialog(ui) {
  if (!ui.smallOutputWarning) return "";
  const { width, height } = ui.smallOutputWarning;
  return `<div class="dialog-backdrop">
    <section class="paste-dialog" role="dialog" aria-modal="true" aria-labelledby="small-output-dialog-title">
      <h2 id="small-output-dialog-title">${t("imageToImage.smallOutputTitle")}</h2>
      <p>${t("imageToImage.smallOutputMessage", {
        width,
        height,
        minimum: `${RECOMMENDED_MINIMUM_SIDE} × ${RECOMMENDED_MINIMUM_SIDE}`,
      })}</p>
      <div class="dialog-actions">
        <button class="secondary-button" data-action="smallOutputCancel">${t("common.cancel")}</button>
        <button class="primary-button" data-action="smallOutputContinue">${t("imageToImage.smallOutputContinue")}</button>
      </div>
    </section>
  </div>`;
}

export function renderInfoDialog(state, ui) {
  const selection = ui.infoDialog;
  if (!selection) return "";

  if (selection.kind === "model") {
    const item = state.models.find(({ descriptor }) => descriptor.id === selection.id);
    if (!item) return "";
    return detailDialog(
      "model-info-dialog-title",
      item.descriptor.displayName,
      renderModelDetails(item.descriptor),
    );
  }

  if (selection.kind === "profile") {
    const profile = state.profiles.find((candidate) => candidate.id === selection.id);
    if (!profile) return "";
    const isActive = state.activeProfileIDs?.[profile.capability] === profile.id;
    return detailDialog(
      "profile-info-dialog-title",
      profile.name,
      renderProfileDetails(profile, isActive, state.models),
    );
  }
  return "";
}

function detailDialog(titleID, title, content) {
  return `<div class="dialog-backdrop">
    <section class="detail-dialog" role="dialog" aria-modal="true" aria-labelledby="${titleID}">
      <header class="detail-dialog-header">
        <h2 id="${titleID}">${escapeHTML(title)}</h2>
        <button
          class="icon-button detail-dialog-close"
          data-action="closeInfoDialog"
          title="${t("common.close")}"
          aria-label="${t("common.close")}"
        >×</button>
      </header>
      <div class="detail-dialog-body">${content}</div>
      <div class="dialog-actions">
        <button class="secondary-button" data-action="closeInfoDialog">${t("common.close")}</button>
      </div>
    </section>
  </div>`;
}

export function renderPasteDialog(state, ui, pasteState) {
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

export function renderToast(state) {
  if (!state.statusMessage) return "";
  return `<div class="toast"><span style="color:var(--positive)">●</span><span>${escapeHTML(state.statusMessage)}</span><button data-action="clearStatus">×</button></div>`;
}

export function refreshToastDOM(state, root) {
  const currentToast = root.querySelector(".toast");
  const markup = renderToast(state);
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
