#[cfg(feature = "logging")]
use tracing_subscriber::{EnvFilter, fmt};

#[cfg(feature = "logging")]
const DEFAULT_FILTER: &str = "rebuild1=trace,winit=info";

/// Installs the default tracing subscriber used by runnable binaries.
#[cfg(feature = "logging")]
pub fn init_default() {
    let filter = logging_filter();

    let _ = fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_names(true)
        .try_init();
}

/// No-op logger for minimum-size builds.
#[cfg(not(feature = "logging"))]
pub fn init_default() {}

/// Builds the logging filter, respecting an explicit `RUST_LOG` override.
#[cfg(feature = "logging")]
fn logging_filter() -> EnvFilter {
    let value = std::env::var("RUST_LOG")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_FILTER.to_owned());

    EnvFilter::try_new(value).unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER))
}
