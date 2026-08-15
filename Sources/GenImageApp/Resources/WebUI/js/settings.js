import { languages, t } from "./i18n.js";
import { themes } from "./themes.js";
import { escapeHTML } from "./format.js";

export function renderSettings(state, ui) {
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

        <section class="settings-card vertical mcp-settings-card">
          <div>
            <h2>${t("settings.mcp")}</h2>
            <p>${t("settings.mcpNote")}</p>
          </div>
          <label>${t("settings.mcpCommand")}</label>
          <code class="command-box">swift run GenImageMCP</code>
          <p>${t("settings.mcpTools")}</p>
        </section>
      </div>
    </section>
  `;
}
