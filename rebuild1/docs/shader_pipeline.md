# Shader and pipeline

## 前作の問題

Rust 側の descriptor binding、push constants、GLSL 側の layout が暗黙に一致していました。片側だけ直してもビルドで気づきにくく、shader が増えるほど認知負荷が上がります。

`rebuild1` では shader interface を先に名前付きで定義します。

shader と pipeline の操作も protocol 境界を意識します。user code は pipeline object を直接触らず、debug option や material handle を message に入れるだけです。

## Descriptor sets

最初の設計は次の 3 set に固定します。

| Set | Name | Owner | Contents |
| --- | --- | --- | --- |
| 0 | frame | frame/renderer | `FrameSnapshot` から作る camera, light, camera effects, time |
| 1 | material | assets | material parameters, textures |
| 2 | pass | graph/pass | shadow map, reflection, post input など pass 固有 |

binding 番号は `shader_interface` に集約します。pipeline 作成側、descriptor 作成側、shader 側が別々の数字を直接持たないようにします。

## Push constants

push constants は小さく保ちます。

許可:

- object transform index
- material index
- draw flags

避ける:

- 大きな matrix を毎 draw で直接 push する
- pass ごとに別 layout を乱立する
- shader ごとに意味の違う同じ offset を使う

## Pipeline library

`PipelineLibrary` は pipeline を名前付きで持ちます。

```text
shadow_opaque
shadow_cutout
scene_opaque
scene_cutout
scene_transparent
post_tonemap
```

pipeline 名は pass 名とは分けます。1 pass が複数 pipeline を使うことはあります。

`PipelineHandle` を protocol に出す場合でも、それは renderer 内部の Vulkan pipeline ではありません。user code が直接 pipeline を bind する設計にはしません。

## Shader reload

hot reload は便利ですが、設計を濁らせやすい部分です。

方針:

- reload watcher は shader source の変化だけを見る
- compile error は最後の正常 pipeline を維持する
- error は log に必ず出す
- pipeline layout が変わる reload は明示的に扱う
- shader binary を source of truth にしない

shader reload の結果は `ShaderReloaded` / `ShaderReloadFailed` event で返します。reload 失敗を log だけに埋めないようにします。

## Interface validation

理想は shader reflection か codegen です。最初はそこまで作り込まず、少なくとも次を守ります。

- binding 番号を document と Rust constants に集約する
- GLSL 側は同じ名前を include で共有できるようにする
- descriptor set layout の作成箇所を module ごとに分けない
- pipeline layout の差分を README ではなく設計ファイルで追う
- protocol handle と GPU internal handle を同じ型にしない

Stage 8 では `renderer::pipeline::shader_interface` に mesh shader binding contract を置き、`VulkanMeshStore::create` の前に set order と binding 衝突を検証します。これは reflection/codegen ではありませんが、Rust layout と GLSL layout の数字が散らばる状態には戻さないための gate です。

## Temporary debug triangle

first draw 用の debug triangle は real material / scene pipeline ではありません。

- set order は `frame/material/pass` を使う
- set 0 binding 0 に小さな `FrameData` uniform を置く
- set 1 と set 2 は空 layout で先に順序だけ固定する
- shader source は `shaders/` に置き、`build.rs` で SPIR-V にする
- app/headless からは `DrawPacket::DebugTriangle` として submit する

Stage 7 の mesh pipeline は set 0 binding 0 に `CameraSnapshot` 由来の view-projection uniform を持ちます。debug triangle は診断用として小さな tint uniform のまま分離します。

Stage 4 では first draw を優先し、temporary uniform のまま完了扱いにしました。Stage 6 で debug triangle は direct app call ではなく `DrawPacket` 経由になりました。full descriptor contents と hot reload error handling は shader interface validation / reload 実装時に扱います。

## Material variants

alpha mode は material policy と pipeline variant の両方に関わります。

```text
opaque      -> depth write on, blend off
cutout      -> depth write on, alpha test
transparent -> depth write off, blend on, sorted draw
```

透明描画は最初から完璧にしません。ただし opaque/cutout/transparent の意味は asset import 時点で失わないようにします。

Stage 6 では `MaterialAlphaMode` と named texture slot を protocol に持たせ、binding 番号は `renderer::pipeline::shader_interface` に集約しました。

Stage 8 では material descriptor set を mesh pipeline に接続し、base-color texture を持つ material だけ sampled texture shader variant に入ります。暗黙 white texture は作りません。alpha cutout は scene と shadow の両方で discard されます。
