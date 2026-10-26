//! The one ONNX Runtime in the process. Moonshine ships the dylib; Smart Turn
//! and the forced aligner load it through `ort`, which can be initialized
//! only once, so both go through here.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use color_eyre::eyre::{Result, bail, eyre};

const RUNTIME_LIBRARY: &str = "libonnxruntime.1.23.2.dylib";

/// Loads the ONNX Runtime library once per process. Moonshine links the
/// same file, so loading it again shares one copy.
/// A missing library is not remembered: setup may install it later.
pub fn init(project_root: &Path) -> Result<()> {
    static RUNTIME: OnceLock<Result<(), String>> = OnceLock::new();
    if let Some(loaded) = RUNTIME.get() {
        return loaded.clone().map_err(|error| eyre!(error));
    }
    let library = find(project_root)?;
    RUNTIME
        .get_or_init(|| {
            ort::init_from(&library)
                .map(|environment| {
                    let _ = environment.with_name("hex").commit();
                })
                .map_err(|error| format!("could not load {}: {error}", library.display()))
        })
        .clone()
        .map_err(|error| eyre!(error))
}

/// The bundled dylib in Hex.app's Frameworks, else the development copy
/// `./scripts/setup.sh` installs under `.venv`.
fn find(project_root: &Path) -> Result<PathBuf> {
    if let Ok(executable) = std::env::current_exe()
        && let Some(contents) = executable.parent().and_then(Path::parent)
    {
        let bundled = contents.join("Frameworks").join(RUNTIME_LIBRARY);
        if bundled.exists() {
            return Ok(bundled);
        }
    }
    let lib = project_root.join(".venv/lib");
    if let Ok(entries) = fs::read_dir(&lib) {
        for python in entries.flatten() {
            let candidate = python
                .path()
                .join("site-packages/moonshine_voice")
                .join(RUNTIME_LIBRARY);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }
    bail!("{RUNTIME_LIBRARY} not found; run ./scripts/setup.sh")
}
