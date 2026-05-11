//! Self-describing capability metadata for the `wowsunpack` binary.
//!
//! The `capabilities` subcommand emits a machine-readable JSON document
//! describing the running build (package version, release tag, git commit,
//! dirty state, target triple, profile) plus the contracts, features, and
//! schema IDs it supports. Downstream pipeline tooling uses this to fail
//! closed when required features are missing.
//!
//! Spec: see `TOOLKIT_VERSION_CONTRACT_AND_PACKAGING_PLAN.md` in the
//! consumer repo. Schema id: `wowsunpack.capabilities/v1`.
//!
//! Determinism: every collection serialises in a fixed order so that two
//! consecutive invocations of `wowsunpack capabilities --json` produce
//! byte-identical output. `contracts` and `schemas` use `BTreeMap`
//! (key-sorted); `features` is a `Vec` in declaration order.

use std::collections::BTreeMap;

use serde::Serialize;

/// Schema identifier for the capabilities payload itself.
const SCHEMA_ID: &str = "wowsunpack.capabilities/v1";

/// Hardcoded fork repository URL. Independent of any local git remote.
const REPOSITORY: &str = "https://github.com/sqz269/wows-toolkit";

/// Release channel marker. Distinguishes the ship-pipeline fork from any
/// upstream `landaire/wows-toolkit` build (which will not emit this
/// subcommand at all).
const RELEASE_CHANNEL: &str = "ship-pipeline";

// Build-time metadata captured by build.rs. All values are always set to
// some string; "unknown" is used when neither a CI env override nor a
// git invocation produced a value.
const RELEASE_TAG: &str = env!("WOWS_TOOLKIT_RELEASE_TAG");
const GIT_COMMIT: &str = env!("WOWS_TOOLKIT_GIT_COMMIT");
const GIT_DIRTY: &str = env!("WOWS_TOOLKIT_GIT_DIRTY");
const TARGET: &str = env!("WOWS_TOOLKIT_TARGET");
const BUILD_PROFILE: &str = env!("WOWS_TOOLKIT_BUILD_PROFILE");

/// Contract version table. Each entry pairs a logical capability with the
/// integer major version of its output contract / behaviour. Bumping a
/// number signals a breaking change to consumers; additive changes use a
/// new feature flag instead.
const CONTRACTS: &[(&str, u32)] = &[
    ("toolkit_capabilities", 1),
    ("ship_export", 1),
    ("model_export", 1),
    ("placements_json", 1),
    ("skel_ext_candidates_json", 3),
    ("material_mappings_json", 1),
    ("armor_json", 1),
    ("ammo_json", 1),
    ("dump_bones_json", 1),
    ("raw_dds_export", 1),
    ("texture_swizzle", 1),
];

/// Feature flags. Each flag names a fine-grained CLI capability. New
/// optional features append a flag rather than bumping the parent
/// contract.
const FEATURES: &[&str] = &[
    "export_ship.all_render_sets",
    "export_ship.accessories_mode",
    "export_ship.placements_json",
    "export_ship.skel_ext_candidates_json",
    "export_ship.material_mappings_json",
    "export_ship.raw_dds_dir",
    "export_model.skel_ext_candidates_json",
    "export_model.material_mappings_json",
    "batch_export_model.raw_dds_dir",
    "armor.json",
    "ammo.json",
    "dump_bones.json",
    "texture.swizzle_dir",
];

/// Schema identifiers emitted in toolkit JSON outputs. Keyed by contract
/// name, value is the `<schema-id>/v<n>` string consumers check.
const SCHEMAS: &[(&str, &str)] = &[
    ("placements_json", "wowsunpack.placements/v1"),
    ("skel_ext_candidates_json", "wowsunpack.skel_ext_candidates/v3"),
    ("material_mappings_json", "wowsunpack.material_mappings/v1"),
    ("armor_json", "wowsunpack.armor/v1"),
    ("ammo_json", "wowsunpack.ammo/v1"),
    ("dump_bones_json", "wowsunpack.visual_bones/v1"),
];

#[derive(Debug, Serialize)]
pub struct ToolInfo {
    pub name: &'static str,
    pub package_version: &'static str,
    pub release_channel: &'static str,
    pub release_tag: &'static str,
    pub git_commit: &'static str,
    pub git_dirty: bool,
    pub repository: &'static str,
    pub target: &'static str,
    pub build_profile: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Capabilities {
    pub schema: &'static str,
    pub tool: ToolInfo,
    pub contracts: BTreeMap<&'static str, u32>,
    pub features: Vec<&'static str>,
    pub schemas: BTreeMap<&'static str, &'static str>,
}

/// Parse the build-time `WOWS_TOOLKIT_GIT_DIRTY` string into a bool.
/// "true" → true; anything else (including "false" and "unknown") → false.
fn parse_git_dirty(raw: &str) -> bool {
    raw.eq_ignore_ascii_case("true")
}

/// Build the capability descriptor for this binary.
pub fn build() -> Capabilities {
    let contracts: BTreeMap<&'static str, u32> =
        CONTRACTS.iter().copied().collect();
    let schemas: BTreeMap<&'static str, &'static str> =
        SCHEMAS.iter().copied().collect();
    let features: Vec<&'static str> = FEATURES.to_vec();

    Capabilities {
        schema: SCHEMA_ID,
        tool: ToolInfo {
            name: "wowsunpack",
            package_version: env!("CARGO_PKG_VERSION"),
            release_channel: RELEASE_CHANNEL,
            release_tag: RELEASE_TAG,
            git_commit: GIT_COMMIT,
            git_dirty: parse_git_dirty(GIT_DIRTY),
            repository: REPOSITORY,
            target: TARGET,
            build_profile: BUILD_PROFILE,
        },
        contracts,
        features,
        schemas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_id_is_pinned() {
        let caps = build();
        assert_eq!(caps.schema, "wowsunpack.capabilities/v1");
        assert_eq!(caps.tool.name, "wowsunpack");
        assert_eq!(caps.tool.release_channel, "ship-pipeline");
        assert_eq!(caps.tool.repository, "https://github.com/sqz269/wows-toolkit");
    }

    #[test]
    fn contract_count_matches_spec() {
        let caps = build();
        assert_eq!(caps.contracts.len(), 11);
        assert_eq!(caps.features.len(), 13);
        assert_eq!(caps.schemas.len(), 6);
    }

    #[test]
    fn schema_versions_are_pinned() {
        let caps = build();
        assert_eq!(caps.contracts.get("toolkit_capabilities").copied(), Some(1));
        assert_eq!(caps.contracts.get("skel_ext_candidates_json").copied(), Some(3));
        assert_eq!(
            caps.schemas.get("skel_ext_candidates_json").copied(),
            Some("wowsunpack.skel_ext_candidates/v3"),
        );
    }

    #[test]
    fn parse_git_dirty_only_true_is_true() {
        assert!(parse_git_dirty("true"));
        assert!(parse_git_dirty("TRUE"));
        assert!(!parse_git_dirty("false"));
        assert!(!parse_git_dirty("unknown"));
        assert!(!parse_git_dirty(""));
    }

    #[test]
    fn serialization_is_deterministic() {
        let a = serde_json::to_string_pretty(&build()).unwrap();
        let b = serde_json::to_string_pretty(&build()).unwrap();
        assert_eq!(a, b);
    }
}
