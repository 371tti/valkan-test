use valkan_test::renderer::{ModelId, Renderer};

const MODEL_CANDIDATES: [&str; 3] = ["assets/model.glb", "assets/model.gltf", "assets/model.obj"];
const MODEL_PATH_ENV: &str = "MODEL_PATH";

pub fn load_scene_model(renderer: &mut Renderer) -> ModelId {
    load_first_model(renderer, &model_paths()).unwrap_or(ModelId::CUBE)
}

fn model_paths() -> Vec<String> {
    let mut paths = Vec::new();

    if let Ok(path) = std::env::var(MODEL_PATH_ENV)
        && !path.trim().is_empty()
    {
        paths.push(path);
    }

    paths.extend(MODEL_CANDIDATES.iter().map(|path| (*path).to_string()));
    paths
}

fn load_first_model(renderer: &mut Renderer, paths: &[String]) -> Option<ModelId> {
    for path in paths {
        match renderer.load_model(path.as_str()) {
            Ok(model) => {
                log::info!("loaded model: {path}");
                return Some(model);
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => log::warn!("failed to load model '{path}': {err}"),
        }
    }

    None
}
