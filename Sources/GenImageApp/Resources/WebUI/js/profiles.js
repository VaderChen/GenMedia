import { architectureLabel, capabilityLabel, escapeHTML } from "./format.js";
import { t } from "./i18n.js";

const primaryCapabilities = ["imageToText", "textToImage", "imageToImage", "textToVideo", "imageToVideo", "textToMusic", "upscale"];

export function renderProfiles(state, ui) {
  const selectedCapability = primaryCapabilities.includes(ui.profileFilter)
    ? ui.profileFilter
    : "all";
  const visibleCapabilities = selectedCapability === "all"
    ? primaryCapabilities
    : [selectedCapability];
  return `
    <section class="page">
      <header class="page-header">
        <div class="page-header-copy">
          <h1>${t("profile.title")}</h1>
          <p>${t("profile.subtitle")}</p>
        </div>
        <div class="filter-row" role="group" aria-label="${t("profile.title")}">
          ${profileFilterChip("all", t("common.all"), selectedCapability)}
          ${primaryCapabilities
            .map((capability) =>
              profileFilterChip(capability, capabilityLabel(capability), selectedCapability))
            .join("")}
        </div>
      </header>
      <div class="page-scroll" data-scroll-id="profiles">
        ${visibleCapabilities.map((capability) => renderProfileSection(state, capability)).join("")}
      </div>
    </section>
  `;
}

function profileFilterChip(value, label, selectedCapability) {
  const active = selectedCapability === value;
  return `<button
    class="chip ${active ? "active" : ""}"
    data-action="profileFilter"
    data-filter="${value}"
    aria-pressed="${active}"
  >${label}</button>`;
}

function renderProfileSection(state, capability) {
  const activeID = state.activeProfileIDs[capability];
  const profiles = sortedProfiles(
    state.profiles.filter((profile) => profile.capability === capability),
    activeID,
    state.models,
  );
  const disabledProfileIDs = new Set(state.disabledProfileIDs || []);
  return `
    <section style="margin-bottom:28px">
      <div class="section-heading">
        <h2>${capabilityLabel(capability)}</h2>
        <span class="section-note">${t("profile.count", { count: profiles.length })}</span>
      </div>
      <div class="card-grid">
        ${profiles
          .map((profile) =>
            renderProfileCard(
              profile,
              profile.id === activeID,
              disabledProfileIDs.has(profile.id),
              state.models,
            ),
          )
          .join("")}
      </div>
    </section>
  `;
}

function sortedProfiles(profiles, activeID, models) {
  return profiles
    .map((profile, originalIndex) => ({
      profile,
      originalIndex,
      rank: profileSortRank(profile, activeID, models),
    }))
    .sort((left, right) => left.rank - right.rank || left.originalIndex - right.originalIndex)
    .map(({ profile }) => profile);
}

function profileSortRank(profile, activeID, models) {
  if (profile.id === activeID) return 0;

  if (isProfileAvailable(profile, models)) return 1;

  const dependencies = profileDependencies(profile)
    .map((modelID) => models.find(({ descriptor }) => descriptor.id === modelID))
    .filter(Boolean);
  if (dependencies.some(({ installation }) =>
    ["queued", "downloading", "paused", "verifying"].includes(installation.phase))) {
    return 2;
  }
  return 3;
}

function renderProfileCard(profile, isActive, isDisabled, models) {
  const availabilityClass = isProfileAvailable(profile, models) ? "is-available" : "is-unavailable";
  return `
    <article class="profile-card ${availabilityClass}" data-profile-card="${profile.id}">
      <div class="card-title-row">
        <div>
          <h2>${escapeHTML(profile.name)}</h2>
          <p>${capabilityLabel(profile.capability)} · r${profile.profileRevision}</p>
        </div>
        ${isActive ? `<span class="badge active">${t("common.active")}</span>` : ""}
        ${isDisabled ? `<span class="badge">${t("profile.disabled")}</span>` : ""}
        ${profile.isBuiltIn ? `<span class="badge">${t("common.builtIn")}</span>` : ""}
      </div>
      <p>${escapeHTML(profile.notes || t("profile.noNotes"))}</p>
      <div class="profile-meta">
        <span>${t("profile.engine")}</span><strong>${architectureLabel(profile.architecture)}</strong>
        <span>${t("profile.model")}</span><strong title="${escapeHTML(profile.modelID)}">${escapeHTML(profile.modelID)}</strong>
        <span>${t("profile.modelRevision")}</span><strong>${escapeHTML(profile.modelRevision)}</strong>
        <span>${t("profile.defaults")}</span><strong>${defaultsSummary(profile.defaults)}</strong>
        <span>${t("profile.loras")}</span><strong>${profileLoRASummary(profile, models)}</strong>
      </div>
      ${profile.isBuiltIn ? renderBuiltInActions(profile, isActive, models) : renderProfileForm(profile, isActive, models)}
    </article>
  `;
}

function renderBuiltInActions(profile, isActive, models) {
  return `<div class="button-row">
    ${renderActivationButton(profile, isActive, models)}
    <button class="secondary-button compact" data-action="duplicateProfile" data-profile-id="${profile.id}">${t("profile.duplicate")}</button>
    ${renderProfileInstallButton(profile, models)}
  </div>`;
}

function renderActivationButton(profile, isActive, models = []) {
  if (isActive) {
    return `<button class="danger-button compact" data-action="deactivateProfile" data-profile-id="${profile.id}" data-capability="${profile.capability}">${t("profile.deactivate")}</button>`;
  }
  const isInstalled = isProfileAvailable(profile, models);
  return `<button class="primary-button compact" data-action="activateProfile" data-profile-id="${profile.id}" data-capability="${profile.capability}" ${isInstalled ? "" : `disabled title="${escapeHTML(t("profile.installRequired"))}"`}>${t("profile.activate")}</button>`;
}

function renderProfileForm(profile, isActive, models) {
  const architectures = ["mlxSwift", "coreML", "localService", "externalCLI"];
  return `
    <div class="profile-form">
      <label>${t("profile.name")}
        <input class="field" data-profile-field="name" value="${escapeHTML(profile.name)}" />
      </label>
      <label>${t("profile.modelID")}
        <input class="field" data-profile-field="modelID" value="${escapeHTML(profile.modelID)}" />
      </label>
      <label>${t("profile.revision")}
        <input class="field" data-profile-field="modelRevision" value="${escapeHTML(profile.modelRevision)}" />
      </label>
      <label>${t("profile.engine")}
        <select class="field" data-profile-field="architecture">
          ${architectures
            .map((architecture) => `<option value="${architecture}" ${architecture === profile.architecture ? "selected" : ""}>${architectureLabel(architecture)}</option>`)
            .join("")}
        </select>
      </label>
      <div class="profile-lora-editor">
        <div class="profile-lora-heading">
          <span>${t("profile.loras")}</span>
          <button type="button" class="secondary-button compact" data-action="addProfileLoRA">＋ ${t("profile.addLoRA")}</button>
        </div>
        <div class="profile-lora-list" data-profile-lora-list>
          ${(profile.loras || []).map(renderProfileLoRARow).join("")}
        </div>
      </div>
    </div>
    <div class="button-row">
      <button class="primary-button compact" data-action="saveProfile" data-profile-id="${profile.id}">${t("profile.save")}</button>
      ${renderActivationButton(profile, isActive, models)}
      ${renderProfileInstallButton(profile, models)}
      <button class="danger-button compact" data-action="deleteProfile" data-profile-id="${profile.id}">${t("profile.delete")}</button>
    </div>
  `;
}

function renderProfileInstallButton(profile, models = []) {
  const modelIDs = profileDependencies(profile);
  const dependencies = modelIDs
    .map((modelID) => models.find(({ descriptor }) => descriptor.id === modelID))
    .filter(Boolean);
  if (dependencies.length !== modelIDs.length) {
    return `<button class="secondary-button compact" disabled>${t("profile.dependencyMissing")}</button>`;
  }
  if (dependencies.every(({ installation }) => installation.phase === "installed")) {
    return `<button class="secondary-button compact" disabled>${t("profile.dependenciesInstalled")}</button>`;
  }
  const activeDownload = dependencies.find(({ installation }) =>
    ["queued", "downloading", "verifying"].includes(installation.phase));
  if (activeDownload) {
    return `<button class="secondary-button compact" disabled>${t(`phase.${activeDownload.installation.phase}`)}</button>`;
  }
  return `<button class="install-button compact" data-action="installProfileModels" data-profile-id="${profile.id}">${t("profile.installDependencies")}</button>`;
}

function profileDependencies(profile) {
  return [profile.modelID, ...(profile.loras || []).map(({ modelID }) => modelID)]
    .filter((modelID, index, values) => modelID && values.indexOf(modelID) === index);
}

function isProfileAvailable(profile, models) {
  const modelIDs = profileDependencies(profile);
  return modelIDs.length > 0 && modelIDs.every((modelID) =>
    models.some(({ descriptor, installation }) =>
      descriptor.id === modelID && installation.phase === "installed"));
}

function profileLoRASummary(profile, models) {
  const loras = profile.loras || [];
  if (!loras.length) return t("lora.none");
  return loras.map((lora) => {
    const model = models.find(({ descriptor }) => descriptor.id === lora.modelID);
    const name = model?.descriptor.displayName || lora.modelID;
    const conditioning = lora.conditioning === "sourceImageCanny"
      ? ` · ${t("profile.cannyControl")}`
      : "";
    return `${escapeHTML(name)} · ${Math.round(Number(lora.scale || 0) * 100)}%${conditioning}`;
  }).join("<br>");
}

function renderProfileLoRARow(lora = {}) {
  const scale = Number.isFinite(Number(lora.scale)) ? Number(lora.scale) : 1;
  const conditioningScale = Number.isFinite(Number(lora.conditioningScale))
    ? Number(lora.conditioningScale)
    : 1;
  return `<div class="profile-lora-row" data-profile-lora-row>
    <label>${t("profile.loraModelID")}
      <input class="field" data-profile-lora-field="modelID" value="${escapeHTML(lora.modelID || "")}" />
    </label>
    <label>${t("lora.scale")}
      <input class="field" type="number" min="0" max="1" step="0.05" data-profile-lora-field="scale" value="${scale}" />
    </label>
    <label>${t("profile.loraConditioning")}
      <select class="field" data-profile-lora-field="conditioning">
        <option value="" ${lora.conditioning ? "" : "selected"}>${t("profile.noConditioning")}</option>
        <option value="sourceImageCanny" ${lora.conditioning === "sourceImageCanny" ? "selected" : ""}>${t("profile.cannyControl")}</option>
      </select>
    </label>
    <label>${t("profile.conditioningScale")}
      <input class="field" type="number" min="0" max="1" step="0.05" data-profile-lora-field="conditioningScale" value="${conditioningScale}" />
    </label>
    <button type="button" class="danger-button compact" data-action="removeProfileLoRA">${t("profile.removeLoRA")}</button>
  </div>`;
}

export function appendProfileLoRARow(button) {
  const list = button.closest("[data-profile-card]")?.querySelector("[data-profile-lora-list]");
  if (list) list.insertAdjacentHTML("beforeend", renderProfileLoRARow());
}

export function removeProfileLoRARow(button) {
  button.closest("[data-profile-lora-row]")?.remove();
}

function defaultsSummary(defaults) {
  if (defaults.durationSeconds) {
    return `${defaults.durationSeconds} sec · ${defaults.steps || "–"} steps`;
  }
  if (defaults.frameCount) {
    const resolution = defaults.width && defaults.height ? `${defaults.width}×${defaults.height} · ` : "";
    return `${resolution}${defaults.frameRate || "–"} FPS · ${defaults.frameCount} frames · ${defaults.steps || "–"} steps`;
  }
  if (defaults.width && defaults.height) return `${defaults.width}×${defaults.height} · ${defaults.steps || "–"} steps`;
  if (defaults.languageCode) return `${defaults.languageCode} · ${defaults.maxTokens || "–"} tokens`;
  if (defaults.upscaleScale) return `${defaults.upscaleScale}× · tile ${defaults.tileSize || "auto"}`;
  return t("profile.custom");
}
