import { languages, t } from "./i18n.js";
import { themes } from "./themes.js";
import { escapeHTML } from "./format.js";

export function renderSettings(state, ui) {
  const mcpService = state.mcpService || {
    isEnabled: false,
    isRunning: false,
    endpointURL: null,
    errorMessage: null,
  };
  const mcpStatus = mcpService.isRunning
    ? t("settings.mcpRunning")
    : mcpService.isEnabled
      ? t("settings.mcpStarting")
      : t("settings.mcpDisabled");
  return `
    <section class="page">
      <header class="page-header">
        <div class="page-header-copy">
          <h1>${t("settings.title")}</h1>
          <p>${t("settings.subtitle")}</p>
        </div>
      </header>
      <div class="page-scroll settings-page" data-scroll-id="settings">
        <section class="settings-card general-settings-card">
          <div>
            <h2>${t("settings.general")}</h2>
            <p>${t("settings.languageNote")}</p>
          </div>
          <label class="settings-control">
            <span>${t("settings.language")}</span>
            <select class="field" data-setting="language">
              ${languages.map((language) => `<option value="${language.id}" ${ui.language === language.id ? "selected" : ""}>${language.label}</option>`).join("")}
            </select>
          </label>
        </section>

        <section class="settings-card vertical appearance-settings-card">
          <div>
            <h2>${t("settings.appearance")}</h2>
            <p>${t("settings.appearanceNote")}</p>
          </div>
          <div class="theme-grid">
            ${themes.map((theme) => `
              <button class="theme-choice ${ui.theme === theme ? "active" : ""}" data-action="selectTheme" data-theme="${theme}">
                <span class="theme-swatch" data-swatch="${theme}"></span>
                <span>${t(`theme.${theme}`)}</span>
              </button>
            `).join("")}
          </div>
        </section>

        <section class="settings-card vertical output-settings-card">
          <div>
            <h2>${t("settings.output")}</h2>
            <p>${t("settings.outputNote")}</p>
          </div>
          <div class="settings-path-control">
            <input
              class="field"
              type="text"
              value="${escapeHTML(state.outputDirectoryPath)}"
              placeholder="${t("settings.outputPlaceholder")}"
              aria-label="${t("settings.outputPath")}"
              title="${t("settings.outputPath")}"
              data-output-directory
              data-preserve-focus="output-directory"
              autocomplete="off"
              spellcheck="false"
            />
            <button
              class="icon-button"
              data-action="chooseOutputDirectory"
              aria-label="${t("settings.chooseOutputDirectory")}"
              title="${t("settings.chooseOutputDirectory")}"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M3.5 6.5h6l2 2h9v9a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z" />
              </svg>
            </button>
          </div>
        </section>

        <section class="settings-card vertical civitai-settings-card">
          <div>
            <h2>${t("settings.civitai")}</h2>
            <p>${t("settings.civitaiNote")}</p>
            <p><a class="settings-external-link" href="https://civitai.com/" target="_blank" rel="noreferrer">${t("settings.civitaiWebsite")}</a></p>
          </div>
          <div class="settings-token-control">
            <label class="settings-token-label" for="civitai-api-token">${t("settings.civitaiToken")}</label>
            <input
              id="civitai-api-token"
              class="field"
              type="password"
              value=""
              placeholder="${t("settings.civitaiTokenPlaceholder")}"
              data-civitai-token
              autocomplete="new-password"
              spellcheck="false"
            />
            <div class="button-row settings-token-actions">
              <button class="primary-button compact" data-action="saveCivitaiToken">${t("settings.civitaiSave")}</button>
              <button class="danger-button compact" data-action="clearCivitaiToken" ${state.civitaiTokenConfigured ? "" : "disabled"}>${t("settings.civitaiClear")}</button>
            </div>
          </div>
          <p class="settings-token-status ${state.civitaiTokenConfigured ? "is-configured" : ""}">${state.civitaiTokenConfigured ? t("settings.civitaiConfigured") : t("settings.civitaiNotConfigured")}</p>
          <p>${t("settings.civitaiTokenNote")}</p>
        </section>

        <section class="settings-card vertical huggingface-settings-card">
          <div>
            <h2>${t("settings.huggingFace")}</h2>
            <p>${t("settings.huggingFaceNote")}</p>
            <p><a class="settings-external-link" href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">${t("settings.huggingFaceWebsite")}</a></p>
          </div>
          <div class="settings-token-control">
            <label class="settings-token-label" for="huggingface-api-token">${t("settings.huggingFaceToken")}</label>
            <input
              id="huggingface-api-token"
              class="field"
              type="password"
              value=""
              placeholder="${t("settings.huggingFaceTokenPlaceholder")}"
              data-huggingface-token
              autocomplete="new-password"
              spellcheck="false"
            />
            <div class="button-row settings-token-actions">
              <button class="primary-button compact" data-action="saveHuggingFaceToken">${t("settings.huggingFaceSave")}</button>
              <button class="danger-button compact" data-action="clearHuggingFaceToken" ${state.huggingFaceTokenConfigured ? "" : "disabled"}>${t("settings.huggingFaceClear")}</button>
            </div>
          </div>
          <p class="settings-token-status ${state.huggingFaceTokenConfigured ? "is-configured" : ""}">${state.huggingFaceTokenConfigured ? t("settings.huggingFaceConfigured") : t("settings.huggingFaceNotConfigured")}</p>
          <p>${t("settings.huggingFaceTokenNote")}</p>
        </section>

        <section class="settings-card vertical mcp-settings-card">
          <div class="mcp-setting-header">
            <div>
              <h2>${t("settings.mcp")}</h2>
              <p>${t("settings.mcpNote")}</p>
            </div>
            <label class="switch-control">
              <input
                type="checkbox"
                role="switch"
                data-setting="mcpEnabled"
                ${mcpService.isEnabled ? "checked" : ""}
              />
              <span class="switch-track" aria-hidden="true"><span></span></span>
              <strong>${mcpStatus}</strong>
            </label>
          </div>
          <div class="mcp-endpoint-block ${mcpService.isEnabled ? "" : "is-disabled"}" aria-disabled="${mcpService.isEnabled ? "false" : "true"}">
            <label>${t("settings.mcpEndpoint")}</label>
            <code class="command-box">${escapeHTML(mcpService.endpointURL || (mcpService.isEnabled ? t("settings.mcpStarting") : t("settings.mcpDisabled")))}</code>
            <p>${t("settings.mcpEndpointNote")}</p>
          </div>
          ${mcpService.errorMessage
            ? `<p class="mcp-service-error">${escapeHTML(t("settings.mcpError", { message: mcpService.errorMessage }))}</p>`
            : ""}
          <p>${t("settings.mcpTools")}</p>
        </section>
      </div>
    </section>
  `;
}
