use std::{
    borrow::Cow,
    fs, io,
    path::PathBuf,
    time::{Duration, Instant, SystemTime},
};

use ash::vk;

use super::PipelineError;

#[derive(Debug, Clone)]
pub enum ShaderCode {
    WatchedSpv {
        path: PathBuf,
        fallback: &'static [u8],
    },
}

impl ShaderCode {
    pub fn watched_spv(path: impl Into<PathBuf>, fallback: &'static [u8]) -> Self {
        Self::WatchedSpv {
            path: path.into(),
            fallback,
        }
    }

    pub(super) fn load(&self) -> Result<Cow<'_, [u8]>, PipelineError> {
        match self {
            Self::WatchedSpv { path, fallback } => match fs::read(path) {
                Ok(bytes) => Ok(Cow::Owned(bytes)),
                Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Cow::Borrowed(fallback)),
                Err(source) => Err(PipelineError::Io {
                    path: path.clone(),
                    source,
                }),
            },
        }
    }

    fn modified(&self) -> Result<Option<SystemTime>, PipelineError> {
        match self {
            Self::WatchedSpv { path, .. } => match fs::metadata(path) {
                Ok(meta) => meta
                    .modified()
                    .map(Some)
                    .map_err(|source| PipelineError::Io {
                        path: path.clone(),
                        source,
                    }),
                Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
                Err(source) => Err(PipelineError::Io {
                    path: path.clone(),
                    source,
                }),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShaderStage {
    pub name: &'static str,
    pub stage: vk::ShaderStageFlags,
    pub entry: &'static str,
    pub code: ShaderCode,
}

impl ShaderStage {
    pub fn new(name: &'static str, stage: vk::ShaderStageFlags, code: ShaderCode) -> Self {
        Self {
            name,
            stage,
            entry: "main",
            code,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShaderSet {
    pub name: &'static str,
    pub stages: Vec<ShaderStage>,
}

impl ShaderSet {
    pub fn graphics(name: &'static str, vertex: ShaderCode, fragment: ShaderCode) -> Self {
        Self {
            name,
            stages: vec![
                ShaderStage::new("vertex", vk::ShaderStageFlags::VERTEX, vertex),
                ShaderStage::new("fragment", vk::ShaderStageFlags::FRAGMENT, fragment),
            ],
        }
    }

    pub fn watched_graphics(
        name: &'static str,
        vertex_path: impl Into<PathBuf>,
        vertex_fallback: &'static [u8],
        fragment_path: impl Into<PathBuf>,
        fragment_fallback: &'static [u8],
    ) -> Self {
        Self::graphics(
            name,
            ShaderCode::watched_spv(vertex_path, vertex_fallback),
            ShaderCode::watched_spv(fragment_path, fragment_fallback),
        )
    }

    pub fn watch_stamp(&self) -> Result<Option<SystemTime>, PipelineError> {
        let mut latest = None;

        for stamp in self.stages.iter().map(|stage| stage.code.modified()) {
            let Some(stamp) = stamp? else {
                continue;
            };

            match latest {
                Some(old) if old >= stamp => {}
                _ => latest = Some(stamp),
            }
        }

        Ok(latest)
    }
}

pub struct HotReload {
    enabled: bool,
    interval: Duration,
    last_check: Instant,
    stamp: Option<SystemTime>,
}

impl HotReload {
    pub fn new(shaders: &ShaderSet, interval: Duration) -> Self {
        let enabled = cfg!(debug_assertions);
        Self {
            enabled,
            interval,
            last_check: Instant::now(),
            stamp: enabled
                .then(|| shaders.watch_stamp().unwrap_or(None))
                .flatten(),
        }
    }

    pub fn changed(
        &mut self,
        shaders: &ShaderSet,
    ) -> Result<Option<Option<SystemTime>>, PipelineError> {
        if !self.enabled || self.last_check.elapsed() < self.interval {
            return Ok(None);
        }

        self.last_check = Instant::now();
        let stamp = shaders.watch_stamp()?;
        Ok((stamp != self.stamp).then_some(stamp))
    }

    pub fn accept(&mut self, stamp: Option<SystemTime>) {
        self.stamp = stamp;
    }
}
