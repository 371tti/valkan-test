use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Clone, Copy)]
struct Shader {
    const_name: &'static str,
    source: &'static str,
    output: &'static str,
    stage: &'static str,
}

const SHADERS: &[Shader] = &[
    Shader {
        const_name: "MESH_UNTEXTURED_VERT",
        source: "shaders/scene/mesh_untextured.vert.slang",
        output: "mesh_untextured.vert.spv",
        stage: "vertex",
    },
    Shader {
        const_name: "MESH_VERT",
        source: "shaders/scene/mesh.vert.slang",
        output: "mesh.vert.spv",
        stage: "vertex",
    },
    Shader {
        const_name: "MESH_SCENE_FRAG",
        source: "shaders/scene/mesh_scene.frag.slang",
        output: "mesh_scene.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "MESH_SCENE_OPAQUE_FRAG",
        source: "shaders/scene/mesh_scene_opaque.frag.slang",
        output: "mesh_scene_opaque.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "MESH_SCENE_FAST_FRAG",
        source: "shaders/scene/mesh_scene_fast.frag.slang",
        output: "mesh_scene_fast.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "MESH_SCENE_OPAQUE_FAST_FRAG",
        source: "shaders/scene/mesh_scene_opaque_fast.frag.slang",
        output: "mesh_scene_opaque_fast.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "MESH_SCENE_TEXTURED_FRAG",
        source: "shaders/scene/mesh_scene_textured.frag.slang",
        output: "mesh_scene_textured.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "MESH_SCENE_OPAQUE_TEXTURED_FRAG",
        source: "shaders/scene/mesh_scene_opaque_textured.frag.slang",
        output: "mesh_scene_opaque_textured.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "MESH_SCENE_TEXTURED_FAST_FRAG",
        source: "shaders/scene/mesh_scene_textured_fast.frag.slang",
        output: "mesh_scene_textured_fast.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "MESH_SCENE_OPAQUE_TEXTURED_FAST_FRAG",
        source: "shaders/scene/mesh_scene_opaque_textured_fast.frag.slang",
        output: "mesh_scene_opaque_textured_fast.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "SHADOW_DIRECTIONAL_VERT",
        source: "shaders/shadow/shadow_directional.vert.slang",
        output: "shadow_directional.vert.spv",
        stage: "vertex",
    },
    Shader {
        const_name: "SHADOW_LOCAL_VERT",
        source: "shaders/shadow/shadow_local.vert.slang",
        output: "shadow_local.vert.spv",
        stage: "vertex",
    },
    Shader {
        const_name: "SHADOW_OPAQUE_DIRECTIONAL_VERT",
        source: "shaders/shadow/shadow_opaque_directional.vert.slang",
        output: "shadow_opaque_directional.vert.spv",
        stage: "vertex",
    },
    Shader {
        const_name: "SHADOW_OPAQUE_LOCAL_VERT",
        source: "shaders/shadow/shadow_opaque_local.vert.slang",
        output: "shadow_opaque_local.vert.spv",
        stage: "vertex",
    },
    Shader {
        const_name: "SHADOW_DEPTH_FRAG",
        source: "shaders/shadow/shadow_depth.frag.slang",
        output: "shadow_depth.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "SHADOW_DEPTH_OPAQUE_FRAG",
        source: "shaders/shadow/shadow_depth_opaque.frag.slang",
        output: "shadow_depth_opaque.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "SHADOW_DEPTH_TEXTURED_FRAG",
        source: "shaders/shadow/shadow_depth_textured.frag.slang",
        output: "shadow_depth_textured.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "SHADOW_TRANSLUCENT_FRAG",
        source: "shaders/shadow/shadow_translucent.frag.slang",
        output: "shadow_translucent.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "SHADOW_TRANSLUCENT_TEXTURED_FRAG",
        source: "shaders/shadow/shadow_translucent_textured.frag.slang",
        output: "shadow_translucent_textured.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "POST_VERT",
        source: "shaders/post/post.vert.slang",
        output: "post.vert.spv",
        stage: "vertex",
    },
    Shader {
        const_name: "POST_TAA_RESOLVE_FRAG",
        source: "shaders/post/post_taa_resolve.frag.slang",
        output: "post_taa_resolve.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "POST_BLOOM_DOWNSAMPLE_FRAG",
        source: "shaders/post/bloom/post_bloom_downsample.frag.slang",
        output: "post_bloom_downsample.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "POST_BLOOM_UPSAMPLE_FRAG",
        source: "shaders/post/bloom/post_bloom_upsample.frag.slang",
        output: "post_bloom_upsample.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "POST_GOD_RAY_MASK_FRAG",
        source: "shaders/post/god_rays/post_god_ray_mask.frag.slang",
        output: "post_god_ray_mask.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "POST_GOD_RAY_PREFILTER_FRAG",
        source: "shaders/post/god_rays/post_god_ray_prefilter.frag.slang",
        output: "post_god_ray_prefilter.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "POST_GOD_RAY_RADIAL_FRAG",
        source: "shaders/post/god_rays/post_god_ray_radial.frag.slang",
        output: "post_god_ray_radial.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "POST_GOD_RAY_TEMPORAL_FRAG",
        source: "shaders/post/god_rays/post_god_ray_temporal.frag.slang",
        output: "post_god_ray_temporal.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "POST_FRAG",
        source: "shaders/post/post.frag.slang",
        output: "post.frag.spv",
        stage: "fragment",
    },
    Shader {
        const_name: "POST_FAST_FRAG",
        source: "shaders/post/post_fast.frag.slang",
        output: "post_fast.frag.spv",
        stage: "fragment",
    },
];

fn main() {
    println!("cargo:rerun-if-env-changed=SLANGC");
    println!("cargo:rerun-if-env-changed=SPIRV_VAL");
    for source in shader_sources(Path::new("shaders")) {
        println!("cargo:rerun-if-changed={}", source.display());
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    write_shader_assets(&out_dir);

    let slangc = env::var_os("SLANGC").unwrap_or_else(|| "slangc".into());
    let spirv_val = env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
    for shader in SHADERS {
        compile_shader(&slangc, shader, &out_dir);
        validate_shader(&spirv_val, shader, &out_dir);
    }
}

fn shader_sources(root: &Path) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    collect_slang_sources(root, &mut sources);
    sources.sort();
    sources
}

fn collect_slang_sources(path: &Path, sources: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(path).unwrap_or_else(|error| {
        panic!(
            "failed to read shader directory {}: {error}",
            path.display()
        )
    });
    for entry in entries {
        let entry = entry.expect("failed to read shader directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect_slang_sources(&path, sources);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "slang")
        {
            sources.push(path);
        }
    }
}

fn write_shader_assets(out_dir: &Path) {
    let mut generated = String::from("// Generated by build.rs. Do not edit.\n\n");
    for shader in SHADERS {
        generated.push_str(&format!(
            "pub(crate) const {}: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{}\"));\n",
            shader.const_name, shader.output
        ));
    }
    fs::write(out_dir.join("shader_assets.rs"), generated)
        .expect("failed to write generated shader asset registry");
}

fn compile_shader(slangc: &std::ffi::OsStr, shader: &Shader, out_dir: &Path) {
    let output = out_dir.join(shader.output);
    let reflection = out_dir.join(format!("{}.json", shader.output));
    let depfile = out_dir.join(format!("{}.d", shader.output));
    let mut command = Command::new(slangc);
    command.arg(shader.source);
    for include_dir in [
        "shaders/shared",
        "shaders/scene",
        "shaders/shadow",
        "shaders/post",
        "shaders/post/bloom",
        "shaders/post/god_rays",
    ] {
        command.arg("-I").arg(include_dir);
    }
    let status = command
        .arg("-target")
        .arg("spirv")
        .arg("-profile")
        .arg("glsl_450")
        .arg("-capability")
        .arg("spirv_1_3")
        .arg("-matrix-layout-column-major")
        .arg("-O3")
        .arg("-fp-mode")
        .arg("fast")
        .arg("-entry")
        .arg("main")
        .arg("-stage")
        .arg(shader.stage)
        .arg("-reflection-json")
        .arg(&reflection)
        .arg("-depfile")
        .arg(&depfile)
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap_or_else(|error| {
            panic!("failed to run slangc; set SLANGC or put slangc on PATH: {error}")
        });

    if !status.success() {
        panic!("slangc failed for {}", shader.source);
    }
}

fn validate_shader(spirv_val: &std::ffi::OsStr, shader: &Shader, out_dir: &Path) {
    let output = out_dir.join(shader.output);
    let status = Command::new(spirv_val)
        .arg("--target-env")
        .arg("vulkan1.1")
        .arg(&output)
        .status()
        .unwrap_or_else(|error| {
            panic!("failed to run spirv-val; set SPIRV_VAL or install SPIR-V Tools: {error}")
        });

    if !status.success() {
        panic!("spirv-val failed for {}", shader.output);
    }
}
