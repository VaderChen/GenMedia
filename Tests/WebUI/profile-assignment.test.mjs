import assert from "node:assert/strict";
import { test } from "node:test";
import { profileAssignment, resolveProfileAssignment } from "../../Sources/GenImageApp/Resources/WebUI/js/profile-assignment.js";

const base = { id: "base", name: "Qwen", capability: "textToImage", modelID: "qwen21", modelRevision: "pinned", architecture: "mlxSwift" };
const turbo = { ...base, id: "turbo", name: "Qwen Turbo" };
const pe = { ...turbo, id: "pe", name: "Qwen Turbo＋提示詞增強" };
const profiles = [base, turbo, pe];

test("exact profile ID wins over an earlier base-model match", () => {
  assert.equal(resolveProfileAssignment(profiles, profileAssignment(pe)), pe);
});
test("saved variant names survive built-in UUID changes", () => {
  const relaunched = profiles.map((profile) => ({ ...profile, id: `new-${profile.id}` }));
  for (const profile of profiles) {
    assert.equal(resolveProfileAssignment(relaunched, profileAssignment(profile)).name, profile.name);
  }
});
test("legacy drafts resolve only an unambiguous model and capability", () => {
  const draft = { ...profileAssignment(pe), profileID: "gone" };
  delete draft.profileName;
  assert.equal(resolveProfileAssignment(profiles, draft), null);
  assert.equal(resolveProfileAssignment([base], draft), base);
  assert.equal(resolveProfileAssignment([{ ...base, capability: "imageToImage" }], draft), null);
});
