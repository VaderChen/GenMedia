const pending = new Map();
const stateListeners = new Set();
const activityListeners = new Set();
const clipboardImageListeners = new Set();
let sequence = 0;

function nativeHandler() {
  return window.webkit?.messageHandlers?.genimage;
}

export function invoke(method, params = {}) {
  const id = `web-${Date.now()}-${sequence++}`;
  const handler = nativeHandler();

  if (!handler) {
    return Promise.reject(new Error("GenImage native bridge is unavailable."));
  }

  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    handler.postMessage({ id, method, params });
  });
}

export function onState(listener) {
  stateListeners.add(listener);
  return () => stateListeners.delete(listener);
}

export function onActivity(listener) {
  activityListeners.add(listener);
  return () => activityListeners.delete(listener);
}

export function onClipboardImage(listener) {
  clipboardImageListeners.add(listener);
  return () => clipboardImageListeners.delete(listener);
}

window.GenImageNative = {
  receive(message) {
    if (message?.kind !== "response" || !message.id) return;
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    if (message.ok) request.resolve(message.payload);
    else request.reject(new Error(message.error || "Native command failed."));
  },

  receiveState(state) {
    stateListeners.forEach((listener) => listener(state));
  },

  receiveActivity(activity) {
    activityListeners.forEach((listener) => listener(activity));
  },

  receiveClipboardImage(image) {
    clipboardImageListeners.forEach((listener) => listener(image));
  },
};
