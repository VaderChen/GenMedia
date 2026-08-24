import { capabilityLabel, escapeHTML, gigabytes, percent, phaseLabel } from "./format.js";
import { t } from "./i18n.js";

export function renderModels(state, ui) {
  const filtered = state.models.filter(({ descriptor, installation }) => {
    const matchesCapability =
      ui.modelFilter === "all" ||
      (ui.modelFilter === "installed"
        ? installation.phase === "installed"
        : descriptor.capabilities.includes(ui.modelFilter));
    const query = ui.modelSearch.trim().toLocaleLowerCase();
    const matchesSearch =
      !query ||
      descriptor.displayName.toLocaleLowerCase().includes(query) ||
      descriptor.publisher.toLocaleLowerCase().includes(query);
    return matchesCapability && matchesSearch;
  });

  return `
    <section class="page">
      <header class="page-header">
        <div class="page-header-copy">
          <h1>${t("model.title")}</h1>
          <p>${t("model.subtitle")}</p>
        </div>
        <input
          class="search-field"
          type="search"
          placeholder="${t("model.search")}"
          data-ui-field="modelSearch"
          data-preserve-focus="model-search"
          value="${escapeHTML(ui.modelSearch)}"
        />
      </header>
      <div class="page-scroll" data-scroll-id="models">
        <div class="model-toolbar-row">
          <div class="filter-row">
            ${filterChip("all", t("common.all"), ui)}
            ${filterChip("installed", t("phase.installed"), ui)}
            ${filterChip("textToImage", capabilityLabel("textToImage"), ui)}
            ${filterChip("imageToText", capabilityLabel("imageToText"), ui)}
            ${filterChip("imageToImage", capabilityLabel("imageToImage"), ui)}
            ${filterChip("textToVideo", capabilityLabel("textToVideo"), ui)}
            ${filterChip("imageToVideo", capabilityLabel("imageToVideo"), ui)}
            ${filterChip("textToMusic", capabilityLabel("textToMusic"), ui)}
            ${filterChip("upscale", capabilityLabel("upscale"), ui)}
            ${filterChip("lora", capabilityLabel("lora"), ui)}
          </div>
          <div class="model-root-control">
            <input
              class="field model-root-field"
              type="text"
              value="${escapeHTML(state.modelRootPath)}"
              placeholder="${t("model.pathPlaceholder")}"
              aria-label="${t("model.defaultPath")}"
              title="${t("model.defaultPath")}"
              data-model-root
              data-preserve-focus="model-root"
              autocomplete="off"
              spellcheck="false"
            />
            <button
              class="icon-button model-root-picker"
              data-action="chooseModelRoot"
              aria-label="${t("model.chooseDirectory")}"
              title="${t("model.chooseDirectory")}"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M3.5 6.5h6l2 2h9v9a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="card-grid">
          ${filtered.length ? filtered.map(renderModelCard).join("") : emptyModels()}
        </div>
      </div>
    </section>
  `;
}

export function renderModelCard({ descriptor, installation }) {
  const capabilities = descriptor.capabilities
    .map((capability) => `<span class="badge capability-badge">${capabilityLabel(capability)}</span>`)
    .join("");
  return `
    <article class="model-card" data-model-card="${escapeHTML(descriptor.id)}">
      <div class="card-title-row">
        <div>
          <h2>${escapeHTML(descriptor.displayName)}</h2>
          <p>${escapeHTML(descriptor.publisher)}</p>
        </div>
        ${descriptor.isRecommended ? `<span class="badge recommended">${t("common.recommended")}</span>` : ""}
      </div>
      <p>${escapeHTML(descriptor.summary)}</p>
      <div class="badge-row">${capabilities}</div>
      <div class="model-meta">
        <span>${t("model.quantization")}</span><strong>${escapeHTML(descriptor.quantization)}</strong>
        <span>${t("model.downloadSize")}</span><strong>${t("model.approx", { size: gigabytes(descriptor.approximateDownloadGB) })}</strong>
        <span>${t("model.memory")}</span><strong>${descriptor.recommendedMemoryGB} GB</strong>
        <span>${t("model.license")}</span><strong>${escapeHTML(descriptor.licenseName)}</strong>
      </div>
      ${renderInstallation(descriptor, installation)}
    </article>
  `;
}

function renderInstallation(model, installation) {
  const phase = installation.phase;
  const error = installation.errorMessage
    ? `<span class="model-installation-error">${escapeHTML(installation.errorMessage)}</span>`
    : "";
  const progress = `
    <div class="model-progress">
      <div class="detail-row"><span data-model-phase>${phaseLabel(phase)}</span><span data-model-size>${gigabytes(installation.downloadedGB)} / ${gigabytes(model.approximateDownloadGB)}</span></div>
      <progress data-model-progress value="${installation.progress}" max="1"></progress>
      <span class="section-note" data-model-percent>${percent(installation.progress)}</span>
    </div>
  `;

  if (phase === "installed") {
    return `${progress}<div class="button-row">
      <button class="secondary-button compact" data-action="repairModel" data-model-id="${escapeHTML(model.id)}">${t("model.verify")}</button>
      <button class="danger-button compact" data-action="removeModel" data-model-id="${escapeHTML(model.id)}">${t("model.remove")}</button>
    </div>`;
  }
  if (phase === "downloading" || phase === "queued" || phase === "verifying") {
    return `${progress}<button class="secondary-button compact" data-action="pauseModel" data-model-id="${escapeHTML(model.id)}" ${phase === "verifying" ? "disabled" : ""}>${t("model.pause")}</button>`;
  }
  if (phase === "paused") {
    return `${progress}<button class="install-button compact" data-action="installModel" data-model-id="${escapeHTML(model.id)}">${t("model.resume")}</button>`;
  }
  return `${error}<button class="install-button compact" data-action="installModel" data-model-id="${escapeHTML(model.id)}">${t("model.install")}</button>`;
}

function filterChip(value, label, ui) {
  return `<button class="chip ${ui.modelFilter === value ? "active" : ""}" data-action="modelFilter" data-filter="${value}">${label}</button>`;
}

function emptyModels() {
  return `<div class="empty-state"><span class="empty-icon">⌕</span><strong>${t("model.notFound")}</strong><span>${t("model.adjustFilter")}</span></div>`;
}
