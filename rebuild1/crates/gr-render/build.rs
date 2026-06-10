use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=shaders/shadow_sampling.glsl");
    println!("cargo:rerun-if-changed=shaders/shadow_alpha.glsl");
    println!("cargo:rerun-if-changed=shaders/pbr_lighting.glsl");

    for (source, output) in [
        ("shaders/mesh.vert", "mesh.vert.spv"),
        ("shaders/mesh_scene.frag", "mesh_scene.frag.spv"),
        (
            "shaders/mesh_scene_textured.frag",
            "mesh_scene_textured.frag.spv",
        ),
        ("shaders/shadow.vert", "shadow.vert.spv"),
        ("shaders/shadow.frag", "shadow.frag.spv"),
        ("shaders/shadow_textured.frag", "shadow_textured.frag.spv"),
        (
            "shaders/shadow_translucent.frag",
            "shadow_translucent.frag.spv",
        ),
        (
            "shaders/shadow_translucent_textured.frag",
            "shadow_translucent_textured.frag.spv",
        ),
        ("shaders/post.vert", "post.vert.spv"),
        ("shaders/post.frag", "post.frag.spv"),
    ] {
        compile_shader(source, output);
    }
}

/// Compiles one GLSL shader into SPIR-V under Cargo's output directory.
fn compile_shader(source: &str, output_name: &str) {
    println!("cargo:rerun-if-changed={source}");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    let output = out_dir.join(output_name);
    let status = Command::new("glslc")
        .arg(source)
        .arg("-I")
        .arg("shaders")
        .arg("-O")
        .arg("-o")
        .arg(&output)
        .status()
        .expect("failed to run glslc; install the Vulkan SDK and put glslc on PATH");

    if !status.success() {
        panic!("glslc failed for {source}");
    }
}
