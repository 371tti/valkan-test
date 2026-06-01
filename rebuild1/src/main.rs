/// Runs the selected rebuild implementation slice.
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    rebuild1::logging::init_default();

    match selected_run_mode()? {
        RunMode::Headless => {
            tracing::info!(mode = "headless", "starting rebuild1 run");
            rebuild1::app::run_headless_once().await?;
            tracing::info!(mode = "headless", "completed rebuild1 run");
        }
        RunMode::Windowed => {
            tracing::info!(mode = "windowed", "starting rebuild1 run");
            rebuild1::app::run_windowed()?;
            tracing::info!(mode = "windowed", "completed rebuild1 run");
        }
        RunMode::WindowSmoke => {
            tracing::info!(mode = "window-smoke", "starting rebuild1 run");
            rebuild1::app::run_windowed_smoke()?;
            tracing::info!(mode = "window-smoke", "completed rebuild1 run");
        }
    }

    Ok(())
}

enum RunMode {
    Headless,
    Windowed,
    WindowSmoke,
}

/// Parses the tiny command line used by this implementation slice.
fn selected_run_mode() -> Result<RunMode, String> {
    let mut args = std::env::args().skip(1);
    let Some(arg) = args.next() else {
        tracing::trace!(mode = "headless", "selected default run mode");
        return Ok(RunMode::Headless);
    };

    if args.next().is_some() {
        return Err(
            "expected at most one argument: --headless, --window, or --window-smoke".into(),
        );
    }

    match arg.as_str() {
        "--window" => {
            tracing::trace!(mode = "windowed", "selected run mode from cli");
            Ok(RunMode::Windowed)
        }
        "--window-smoke" => {
            tracing::trace!(mode = "window-smoke", "selected run mode from cli");
            Ok(RunMode::WindowSmoke)
        }
        "--headless" => {
            tracing::trace!(mode = "headless", "selected run mode from cli");
            Ok(RunMode::Headless)
        }
        _ => Err(format!(
            "unknown argument: {arg}; expected --headless, --window, or --window-smoke"
        )),
    }
}
