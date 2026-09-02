// Output-size arithmetic for the Web UI — the mirror of GenImageCore's OutputGeometry.swift.
//
// Every rule that turns a requested size into a size the Runtime can actually generate lives
// in one of these two files. When one changes, change the other: the alignment, the supported
// range, and the rounding direction all have to agree, or the number the user sees and the
// number the Runtime denoises drift apart.

// One latent token per 16 px — every canvas has to land on a multiple of this.
export const ALIGNMENT = 16;
export const MINIMUM_DIMENSION = 64;
export const MAXIMUM_DIMENSION = 4096;

// Below this output area, image-to-image still renders at the Runtime's native canvas and
// resamples down, so the composition holds — but the detail the file can carry does not.
export const RECOMMENDED_MINIMUM_SIDE = 512;
export const RECOMMENDED_MINIMUM_AREA = RECOMMENDED_MINIMUM_SIDE * RECOMMENDED_MINIMUM_SIDE;

export const H3_ALIGNMENT = 32;
export const H3_TECHNICAL_MINIMUM_DIMENSION = 64;
export const H3_QUALITY_WARNING_MINIMUM_SHORT_SIDE = 512;
export const H3_NATIVE_QUALITY_REFERENCE_SHORT_SIDE = 768;

/// Nearest multiple of ALIGNMENT, clamped into the supported range.
export function quantizeDimension(value) {
  if (!Number.isFinite(value) || value <= 0) return MINIMUM_DIMENSION;
  const rounded = Math.round(value / ALIGNMENT) * ALIGNMENT;
  return Math.min(MAXIMUM_DIMENSION, Math.max(MINIMUM_DIMENSION, rounded));
}

export function isSupportedDimension(value) {
  return Number.isFinite(value)
    && value >= MINIMUM_DIMENSION
    && value <= MAXIMUM_DIMENSION
    && value % ALIGNMENT === 0;
}

/// The slider range for one axis once the other axis is locked to `ratio`.
export function resolutionBounds(dimension, ratio) {
  const ratioValue = dimension === "width"
    ? ratio.width / ratio.height
    : ratio.height / ratio.width;
  return {
    min: Math.max(MINIMUM_DIMENSION, Math.ceil((MINIMUM_DIMENSION * ratioValue) / ALIGNMENT) * ALIGNMENT),
    max: Math.min(MAXIMUM_DIMENSION, Math.floor((MAXIMUM_DIMENSION * ratioValue) / ALIGNMENT) * ALIGNMENT),
  };
}

/// Both sides for `ratio`, anchored on whichever axis the user is dragging. The anchor gives
/// way when the derived side would leave the supported range.
export function dimensionsForAspect(anchor, value, ratioWidth, ratioHeight) {
  const safeWidth = Math.max(1, ratioWidth);
  const safeHeight = Math.max(1, ratioHeight);
  if (anchor === "height") {
    let height = quantizeDimension(value);
    const derivedWidth = height * safeWidth / safeHeight;
    let width;
    if (derivedWidth < MINIMUM_DIMENSION) {
      width = MINIMUM_DIMENSION;
      height = quantizeDimension(width * safeHeight / safeWidth);
    } else if (derivedWidth > MAXIMUM_DIMENSION) {
      width = MAXIMUM_DIMENSION;
      height = quantizeDimension(width * safeHeight / safeWidth);
    } else {
      width = quantizeDimension(derivedWidth);
    }
    return { width, height };
  }

  let width = quantizeDimension(value);
  const derivedHeight = width * safeHeight / safeWidth;
  let height;
  if (derivedHeight < MINIMUM_DIMENSION) {
    height = MINIMUM_DIMENSION;
    width = quantizeDimension(height * safeWidth / safeHeight);
  } else if (derivedHeight > MAXIMUM_DIMENSION) {
    height = MAXIMUM_DIMENSION;
    width = quantizeDimension(height * safeWidth / safeHeight);
  } else {
    height = quantizeDimension(derivedHeight);
  }
  return { width, height };
}

export function defaultVideoDimensionsForAspect(ratioWidth, ratioHeight) {
  const anchor = ratioWidth >= ratioHeight ? "width" : "height";
  return dimensionsForAspect(anchor, 1280, ratioWidth, ratioHeight);
}

/// A source image's own size, snapped to something the Runtime can generate. The longer side
/// leads so the aspect survives the snap.
export function sourceImageDimensions(asset) {
  const sourceWidth = Math.max(1, Number(asset.pixelWidth));
  const sourceHeight = Math.max(1, Number(asset.pixelHeight));
  if (sourceWidth >= sourceHeight) {
    const width = quantizeDimension(sourceWidth);
    return { width, height: quantizeDimension(width * sourceHeight / sourceWidth) };
  }
  const height = quantizeDimension(sourceHeight);
  return { width: quantizeDimension(height * sourceWidth / sourceHeight), height };
}

/// The configured image output when it is below the recommended area, otherwise null.
export function undersizedImageOutput(state) {
  const width = Number(state?.recipe?.width);
  const height = Number(state?.recipe?.height);
  if (![width, height].every((value) => Number.isFinite(value) && value > 0)) return null;
  if (width * height >= RECOMMENDED_MINIMUM_AREA) return null;
  return { width, height };
}

export function isMiniMaxH3ModelID(modelID) {
  return typeof modelID === "string" && modelID.toLowerCase().includes("minimax-h3");
}

export function normalizedH3VideoDimension(value) {
  if (!Number.isFinite(value) || value <= 0) return H3_TECHNICAL_MINIMUM_DIMENSION;
  const aligned = Math.floor(value / H3_ALIGNMENT) * H3_ALIGNMENT;
  return Math.min(MAXIMUM_DIMENSION, Math.max(H3_TECHNICAL_MINIMUM_DIMENSION, aligned));
}

export function normalizedH3VideoDimensions(width, height) {
  return {
    width: normalizedH3VideoDimension(width),
    height: normalizedH3VideoDimension(height),
  };
}

export function undersizedH3VideoOutput(state, profile) {
  if (!isMiniMaxH3ModelID(profile?.modelID)) return null;
  const width = Number(state?.videoOutputSettings?.width);
  const height = Number(state?.videoOutputSettings?.height);
  if (![width, height].every((value) => Number.isFinite(value) && value > 0)) return null;

  const normalized = normalizedH3VideoDimensions(width, height);
  if (Math.min(normalized.width, normalized.height) >= H3_QUALITY_WARNING_MINIMUM_SHORT_SIDE) {
    return null;
  }
  return {
    width,
    height,
    normalizedWidth: normalized.width,
    normalizedHeight: normalized.height,
  };
}
