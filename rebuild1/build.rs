use std::{env, path::PathBuf, process::Command};

fn main() {
    compile_shader("shaders/debug_triangle.vert", "debug_triangle.vert.spv");
    compile_shader("shaders/debug_triangle.frag", "debug_triangle.frag.spv");
    compile_shader("shaders/mesh.vert", "mesh.vert.spv");
    compile_shader("shaders/mesh_scene.frag", "mesh_scene.frag.spv");
    compile_shader(
        "shaders/mesh_scene_textured.frag",
        "mesh_scene_textured.frag.spv",
    );
    compile_shader("shaders/shadow.vert", "shadow.vert.spv");
    compile_shader("shaders/shadow.frag", "shadow.frag.spv");
    compile_shader("shaders/shadow_textured.frag", "shadow_textured.frag.spv");
    compile_shader("shaders/post.vert", "post.vert.spv");
    compile_shader("shaders/post.frag", "post.frag.spv");
}

/// Compiles one GLSL shader into SPIR-V under Cargo's output directory.
fn compile_shader(source: &str, output_name: &str) {
    println!("cargo:rerun-if-changed={source}");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    let output = out_dir.join(output_name);
    let status = Command::new("glslc")
        .arg(source)
        .arg("-o")
        .arg(&output)
        .status()
        .expect("failed to run glslc; install the Vulkan SDK and put glslc on PATH");

    if !status.success() {
        panic!("glslc failed for {source}");
    }
}
