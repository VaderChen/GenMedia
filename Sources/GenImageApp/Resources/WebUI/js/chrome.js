import { renderDownloads } from "./downloads.js";
import { escapeHTML, gigabytes, percent, phaseLabel } from "./format.js";
import { RECOMMENDED_MINIMUM_SIDE } from "./geometry.js";
import { t } from "./i18n.js";
import { renderModels } from "./models.js";
import { renderProfiles } from "./profiles.js";
import { renderSettings } from "./settings.js";
import { workspaceStateForActiveTab } from "./workspace-tabs.js";
import { isInferenceBusy, renderQuickTools, renderWorkspace } from "./workspace.js";

// App 外框：側邊欄、路由切換、更新橫幅、系統資源列、對話框與 toast。
//
// 全部是純畫面函式，`state` 與 `ui` 由呼叫端傳入，不讀取任何模組層級的可變狀態，也不觸發重繪。

export function renderSidebar(state, ui) {
  return `
    <aside class="sidebar">
      <div class="brand">
        <img class="brand-mark" src="./assets/GenImage-AppIcon.png" alt="" aria-hidden="true" />
        <div class="brand-copy"><strong>${t("brand.name")}</strong><span>${t("brand.subtitle")}</span></div>
      </div>
      <nav class="sidebar-nav">
        ${navButton(ui, "workspace", "▦", t("nav.workspace"))}
        ${navButton(ui, "models", "⬡", t("nav.models"))}
        ${navButton(ui, "profiles", "⇄", t("nav.profiles"))}
        ${ui.profileHintVisible ? `
          <div class="profile-nav-hint" role="status">
            <span class="profile-nav-hint-icon" aria-hidden="true">✦</span>
            <span>${t("profile.firstUseHint")}</span>
            <button class="profile-nav-hint-dismiss" data-action="dismissProfileHint" aria-label="${t("profile.dismissHint")}" title="${t("profile.dismissHint")}">×</button>
          </div>
        ` : ""}
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

function navButton(ui, route, icon, title) {
  return `<button class="nav-button ${ui.route === route ? "active" : ""}" data-action="navigate" data-route="${route}"><span>${icon}</span>${title}</button>`;
}

export function renderRoute(state, ui) {
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
  </section>`;
}

function metricRow(key, label, value) {
  const metric = metricDisplay(value);
  return `<div class="metric-row metric-level-${metric.level}" data-metric-key="${key}">
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
