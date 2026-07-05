use std::{env, path::PathBuf, process::Command};

fn main() {
    for include in [
        "shaders/shadow_sampling.glsl",
        "shaders/shadow_alpha.glsl",
        "shaders/pbr_lighting.glsl",
        "shaders/post_common.glsl",
        "shaders/post_fxaa.glsl",
        "shaders/post_ssao.glsl",
        "shaders/post_ssr.glsl",
        "shaders/post_bloom_common.glsl",
        "shaders/post_bloom.glsl",
        "shaders/post_god_ray_common.glsl",
        "shaders/post_god_rays.glsl",
    ] {
        println!("cargo:rerun-if-changed={include}");
    }

    for (source, output) in [
        ("shaders/mesh_untextured.vert", "mesh_untextured.vert.spv"),
        ("shaders/mesh.vert", "mesh.vert.spv"),
        ("shaders/mesh_scene.frag", "mesh_scene.frag.spv"),
        ("shaders/mesh_scene_fast.frag", "mesh_scene_fast.frag.spv"),
        (
            "shaders/mesh_scene_textured.frag",
            "mesh_scene_textured.frag.spv",
        ),
        (
            "shaders/mesh_scene_textured_fast.frag",
            "mesh_scene_textured_fast.frag.spv",
        ),
        ("shaders/shadow.vert", "shadow.vert.spv"),
        ("shaders/shadow.frag", "shadow.frag.spv"),
        ("shaders/shadow_textured.frag", "shadow_textured.frag.spv"),
        ("shaders/shadow_depth.frag", "shadow_depth.frag.spv"),
        (
            "shaders/shadow_depth_textured.frag",
            "shadow_depth_textured.frag.spv",
        ),
        (
            "shaders/shadow_translucent.frag",
            "shadow_translucent.frag.spv",
        ),
        (
            "shaders/shadow_translucent_textured.frag",
            "shadow_translucent_textured.frag.spv",
        ),
        (
            "shaders/shadow_moment_blur.frag",
            "shadow_moment_blur.frag.spv",
        ),
        ("shaders/post.vert", "post.vert.spv"),
        (
            "shaders/post_bloom_downsample.frag",
            "post_bloom_downsample.frag.spv",
        ),
        (
            "shaders/post_bloom_upsample.frag",
            "post_bloom_upsample.frag.spv",
        ),
        (
            "shaders/post_god_ray_mask.frag",
            "post_god_ray_mask.frag.spv",
        ),
        (
            "shaders/post_god_ray_prefilter.frag",
            "post_god_ray_prefilter.frag.spv",
        ),
        (
            "shaders/post_god_ray_radial.frag",
            "post_god_ray_radial.frag.spv",
        ),
        (
            "shaders/post_god_ray_temporal.frag",
            "post_god_ray_temporal.frag.spv",
        ),
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
