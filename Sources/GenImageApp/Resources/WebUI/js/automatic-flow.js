import { t } from "./i18n.js";

export function renderAutomaticFlow() {
  return `
    <section class="page">
      <header class="page-header">
        <div class="page-header-copy">
          <h1>${t("automaticFlow.title")}</h1>
        </div>
      </header>
      <div class="page-scroll" data-scroll-id="automatic-flow">
        <div class="empty-state">
          <span class="empty-icon" aria-hidden="true">⚙</span>
          <strong>${t("automaticFlow.building")}</strong>
        </div>
      </div>
    </section>
  `;
}
