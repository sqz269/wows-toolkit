//! Detects available game data builds and emits cfg flags for conditional test compilation.
//!
//! Emits:
//! - `has_game_data` — at least one build is available
//! - `has_build_NNNNN` — specific build number is available
//!
//! Tests can use:
//! ```ignore
//! #[test]
//! #[cfg_attr(not(has_game_data), ignore)]
//! fn test_needs_game_data() { ... }
//! ```

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

#[derive(Deserialize, Default)]
struct Registry {
    latest_path: Option<PathBuf>,
    #[serde(default)]
    builds: BTreeMap<String, RegistryEntry>,
}

#[derive(Deserialize)]
struct RegistryEntry {
    #[allow(dead_code)]
    version: String,
}

fn find_workspace_root() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    let mut dir = manifest_dir.as_path();
    loop {
        if dir.join("game_versions.toml").exists() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
}

fn scan_bin_dir(path: &Path) -> Vec<u32> {
    let bin_dir = path.join("bin");
    let Ok(entries) = std::fs::read_dir(&bin_dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()))
        .collect()
}

fn discover_builds(workspace_root: &Path) -> Vec<u32> {
    let data_dir = match std::env::var("WOWS_GAME_DATA") {
        Ok(d) => PathBuf::from(d),
        Err(_) => workspace_root.join("game_data"),
    };

    let registry_path = data_dir.join("versions.toml");
    let registry: Registry =
        std::fs::read_to_string(&registry_path).ok().and_then(|s| toml::from_str(&s).ok()).unwrap_or_default();

    let mut builds: Vec<u32> = Vec::new();

    // Builds from registry
    for key in registry.builds.keys() {
        if let Ok(build) = key.parse::<u32>() {
            // For downloaded builds, verify the directory exists
            let build_dir = data_dir.join("builds").join(key);
            if build_dir.exists() {
                builds.push(build);
            }
        }
    }

    // Builds from latest_path
    if let Some(ref latest) = registry.latest_path {
        for build in scan_bin_dir(latest) {
            if !builds.contains(&build) {
                builds.push(build);
            }
        }
    }

    // Also scan game_data/builds/ for any unregistered builds
    let builds_dir = data_dir.join("builds");
    if builds_dir.exists()
        && let Ok(entries) = std::fs::read_dir(&builds_dir)
    {
        for entry in entries.filter_map(|e| e.ok()) {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
                && let Some(build) = entry.file_name().to_str().and_then(|s| s.parse::<u32>().ok())
                && !builds.contains(&build)
            {
                builds.push(build);
            }
        }
    }

    builds.sort();
    builds
}

/// Resolve a build-time metadata value from (1) the named CI env override,
/// or (2) a best-effort git command, or (3) a fallback string. Never panics
/// and never fails the build.
fn resolve_meta(env_var: &str, git_args: &[&str], git_cwd: &Path, fallback: &str) -> String {
    if let Ok(v) = std::env::var(env_var)
        && !v.trim().is_empty()
    {
        return v.trim().to_string();
    }
    match Command::new("git").current_dir(git_cwd).args(git_args).output() {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if s.is_empty() { fallback.to_string() } else { s }
        }
        _ => fallback.to_string(),
    }
}

/// Emit build-time env vars for the capabilities subcommand. All values are
/// best-effort: missing git / missing env vars resolve to "unknown" rather
/// than failing the build.
fn emit_capabilities_metadata(workspace_root: Option<&Path>) {
    // Re-run on any of the overrides changing.
    println!("cargo:rerun-if-env-changed=WOWS_TOOLKIT_RELEASE_TAG");
    println!("cargo:rerun-if-env-changed=WOWS_TOOLKIT_GIT_COMMIT");
    println!("cargo:rerun-if-env-changed=WOWS_TOOLKIT_GIT_DIRTY");

    // Use workspace root as git CWD if we found one; otherwise fall back to
    // the manifest dir. Either is inside the repo so `git -C <dir>` works.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let git_cwd: PathBuf =
        workspace_root.map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(&manifest_dir));

    // Release tag: env override or "unknown" (we don't try to derive from
    // git tags here — a fork may have many irrelevant tags upstream).
    let release_tag = std::env::var("WOWS_TOOLKIT_RELEASE_TAG")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let git_commit = resolve_meta(
        "WOWS_TOOLKIT_GIT_COMMIT",
        &["rev-parse", "HEAD"],
        &git_cwd,
        "unknown",
    );

    // Dirty: env override wins; else look at porcelain status. Empty
    // porcelain = clean; any output = dirty; failure = "unknown".
    let git_dirty = if let Ok(v) = std::env::var("WOWS_TOOLKIT_GIT_DIRTY")
        && !v.trim().is_empty()
    {
        v.trim().to_string()
    } else {
        match Command::new("git")
            .current_dir(&git_cwd)
            .args(["status", "--porcelain"])
            .output()
        {
            Ok(out) if out.status.success() => {
                if out.stdout.iter().all(|&b| b == b' ' || b == b'\n' || b == b'\r' || b == b'\t') {
                    "false".to_string()
                } else {
                    "true".to_string()
                }
            }
            _ => "unknown".to_string(),
        }
    };

    // TARGET / PROFILE are always provided by Cargo when running build scripts.
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    let build_profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());

    println!("cargo:rustc-env=WOWS_TOOLKIT_RELEASE_TAG={release_tag}");
    println!("cargo:rustc-env=WOWS_TOOLKIT_GIT_COMMIT={git_commit}");
    println!("cargo:rustc-env=WOWS_TOOLKIT_GIT_DIRTY={git_dirty}");
    println!("cargo:rustc-env=WOWS_TOOLKIT_TARGET={target}");
    println!("cargo:rustc-env=WOWS_TOOLKIT_BUILD_PROFILE={build_profile}");
}

/// Build numbers referenced by tests that may not be locally available.
/// Declared here so check-cfg doesn't warn about unknown cfgs.
const KNOWN_TEST_BUILDS: &[u32] = &[
    6965290,  // v12.3.1 (S-189 submarine replay)
    9531281,  // v14.1.0 (Hull DD replay)
    11965230, // v15.1.0 (Vermont, Marceau, Narai replays)
];

fn main() {
    // Declare all possible cfgs to satisfy check-cfg
    println!("cargo:rustc-check-cfg=cfg(has_game_data)");

    // Pre-declare check-cfg for all known test builds
    for &build in KNOWN_TEST_BUILDS {
        println!("cargo:rustc-check-cfg=cfg(has_build_{build})");
    }

    let workspace_root_opt = find_workspace_root();

    // Always emit capabilities metadata env vars — even if workspace
    // discovery failed (e.g. publish-style builds). Falls back to
    // "unknown" if git is unavailable.
    emit_capabilities_metadata(workspace_root_opt.as_deref());

    let Some(workspace_root) = workspace_root_opt else {
        return;
    };

    let builds = discover_builds(&workspace_root);

    for &build in &builds {
        // Declare check-cfg for any discovered build not in the known list
        if !KNOWN_TEST_BUILDS.contains(&build) {
            println!("cargo:rustc-check-cfg=cfg(has_build_{build})");
        }
        println!("cargo:rustc-cfg=has_build_{build}");
    }

    if !builds.is_empty() {
        println!("cargo:rustc-cfg=has_game_data");
    }

    // Re-run if registry changes
    let data_dir = match std::env::var("WOWS_GAME_DATA") {
        Ok(d) => PathBuf::from(d),
        Err(_) => workspace_root.join("game_data"),
    };
    println!("cargo:rerun-if-changed={}", data_dir.join("versions.toml").display());
    println!("cargo:rerun-if-env-changed=WOWS_GAME_DATA");
}
