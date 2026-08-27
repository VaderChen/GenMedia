import { t } from "./i18n.js";

const architectureLabels = {
  mlxSwift: "MLX Swift",
  coreML: "Core ML",
};

export function escapeHTML(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function capabilityLabel(value) {
  return t(`cap.${value}`);
}

export function actionLabel(value) {
  if (value === "importImage") return t("sidebar.import");
  if (value === "importMedia") return t("workspace.importMedia");
  if (value === "describe") return t("cap.imageToText");
  if (value === "generate") return t("cap.textToImage");
  if (value === "generateVideo") return t("workspace.generateVideo");
  if (value === "generateMusic") return t("workspace.generateMusic");
  if (value === "generateSubtitles") return t("workspace.generateSubtitles");
  if (value === "createImageLoop") return t("workspace.createImageLoop");
  if (value === "mergeMedia") return t("workspace.mergeMedia");
  if (value === "imageToImage") return t("cap.imageToImage");
  if (value === "upscale") return t("cap.upscale");
  return value;
}

export function kindLabel(value) {
  if (value === "imported") return t("asset.original");
  if (value === "importedVideo") return t("asset.importedVideo");
  if (value === "importedAudio") return t("asset.importedAudio");
  if (value === "generated") return t("asset.generated");
  if (value === "generatedVideo") return t("asset.video");
  if (value === "generatedAudio") return t("asset.audio");
  if (value === "generatedSubtitle") return t("asset.subtitle");
  if (value === "edited") return t("cap.imageToImage");
  if (value === "upscaled") return t("asset.upscaled");
  return value;
}

export function architectureLabel(value) {
  if (value === "localService") return t("arch.localService");
  if (value === "externalCLI") return t("arch.externalCLI");
  return architectureLabels[value] || value;
}

export function phaseLabel(value) {
  return t(`phase.${value}`);
}

export function jobLabel(value) {
  return t(`job.${value}`);
}

export function gigabytes(value) {
  return `${Number(value || 0).toFixed(1)} GB`;
}

export function percent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}
