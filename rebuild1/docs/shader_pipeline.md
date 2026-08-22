# Shader and pipeline

## 前作の問題

Rust 側の descriptor binding、push constants、shader 側の layout が暗黙に一致していました。片側だけ直してもビルドで気づきにくく、shader が増えるほど認知負荷が上がります。

`rebuild1` では shader interface を先に名前付きで定義します。

shader と pipeline の操作も protocol 境界を意識します。user code は pipeline object を直接触らず、debug option や material handle を message に入れるだけです。

## Descriptor sets

最初の設計は次の 3 set に固定します。

| Set | Name | Owner | Contents |
| --- | --- | --- | --- |
| 0 | frame | frame/renderer | `FrameSnapshot` から作る camera, light, camera effects, time |
| 1 | material | assets | material parameters, textures |
| 2 | pass | graph/pass | shadow cascades, translucent shadow maps, post input など pass 固有 |

binding 番号は `shader_interface` に集約します。pipeline 作成側、descriptor 作成側、shader 側が別々の数字を直接持たないようにします。

## Slang build pipeline

shader source は `crates/gr-render/shaders/` 以下の Slang を source of truth にします。用途ごとの構成は次の通りです。

```text
shaders/
  shared/             型に依存しない数学・Slang compatibility
  scene/              mesh entrypoint・material sampling・PBR lighting
  shadow/             Stable CSM・PCSS・local/translucent shadow
  post/               TAA・compose・SSAO・SSR
    bloom/            bloom downsample/upsample
    god_rays/         legacy mask/prefilter/radial/temporal + CSM/Fog volumetric quality branch
                       blue-noise ray marching + temporally clamped accumulation
```

entrypoint と include 専用 module をファイル名だけで区別せず、責務をディレクトリで固定します。`build.rs` の `SHADERS` manifest が entrypoint、stage、SPIR-V output、Rust asset 名を一元管理し、各ディレクトリを明示的な include search path として渡します。

build 時の処理:

- `slangc` で `main` entrypoint を SPIR-V へ compile する
- `-matrix-layout-column-major` を固定し、既存 uniform buffer の matrix ABI を保つ
- `-reflection-json` と depfile を `OUT_DIR` に出す
- `spirv-val --target-env vulkan1.1` で生成 SPIR-V を検証する
- `OUT_DIR/shader_assets.rs` を生成し、`renderer::vulkan::shader::assets::*` から読む

Vulkan 側は `renderer::vulkan::shader` だけが `include_bytes!` と shader module 作成を持ちます。個別 pipeline module は `assets::POST_VERT` のような生成済み asset を選ぶだけにします。

追加手順:

1. `.slang` source を対応する機能ディレクトリに追加する。
2. `build.rs` の `SHADERS` manifest に `const_name` / `source` / `output` / `stage` を追加する。
3. 必要な Vulkan pipeline module で `shader::assets::<CONST_NAME>` を選ぶ。
4. descriptor set や push constant layout を変える場合は `shader_interface` と docs を同じ変更として更新する。

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
shadow_translucent
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
- Slang 側は同じ名前を include で共有できるようにする
- descriptor set layout の作成箇所を module ごとに分けない
- pipeline layout の差分を README ではなく設計ファイルで追う
- protocol handle と GPU internal handle を同じ型にしない

Stage 8 では `renderer::pipeline::shader_interface` に mesh shader binding contract を置き、`VulkanMeshStore::create` の前に set order と binding 衝突を検証します。現状は reflection JSON を生成しますが、descriptor layout の自動生成にはまだ使いません。Rust layout と Slang layout の数字が散らばる状態には戻さないための gate です。

## Mesh and post pipeline

temporary debug triangle pipeline は削除済みです。first draw 後は `FrameSnapshot.render_items` が唯一の scene draw 入力になり、asset 未ロード時は scene clear frame を present します。

Stage 7 の mesh pipeline は set 0 binding 0 に `CameraSnapshot` 由来の view/projection uniform を持ちます。scene pass は scene HDR、depth、normal/material のMRTを出力し、Stable CSMのPCSS visibilityをその場で反映します。PCSSの最終filterはset 2 binding 9の比較サンプラ（LINEAR + `LESS_OR_EQUAL`）でhardware 2x2 PCFを使い、binding 10のraw depthはblocker探索だけに使います。blocker/filter各タップにはreceiver-planeのlight-space深度勾配を加え、斜面で同じ比較深度を反復する横縞を抑えます。receiver biasはD16量子化の2段余裕、texel/depth-span、法線角度に応じたslope、receiver-planeの1 texel深度勾配を合成し、balanced/highでは各スケールを引き上げます。タップ回転はラスタピクセルとcascadeの整数hashから生成し、連続ワールド座標sinハッシュ由来の短周期パターンを避けます。その後のHDR TAAは16位相のzero-centroid jitter、stable-grid reprojection、depth/normal confidence、YCoCg variance clampで履歴を蓄積します。post pass はTAA済みHDRとscene metadataを読み、SSR / SSAO / camera effects / tone mappingを適用します。FXAAはTAAとの二重適用を避けるためproduction postでは無効です。SSR / SSAO / AA / lighting wrap / renderer-wide contrast は `SetQualitySettings` で更新される renderer 状態からpush constantとframe uniformへpackします。

full descriptor contents と hot reload error handling は shader interface validation / reload 実装時に扱います。

## Material variants

alpha mode は material policy と pipeline variant の両方に関わります。

```text
opaque      -> depth write on, blend off
cutout      -> depth write on, alpha test
transparent -> depth write off, blend on, sorted draw
```

透明描画は最初から完璧にしません。ただし opaque/cutout/transparent の意味は asset import 時点で失わないようにします。

Stage 6 では `MaterialAlphaMode` と named texture slot を protocol に持たせ、binding 番号は `renderer::pipeline::shader_interface` に集約しました。

Stage 8 では material descriptor set を mesh pipeline に接続し、base-color texture を持つ material だけ sampled texture shader variant に入ります。暗黙 white texture は作りません。alpha cutout は scene と opaque shadow の両方で discard されます。

Stage 9 shadow slice では pass descriptor set が `shadow_cascade_0..3` と `translucent_shadow_0..3` を名前付き binding として持ちます。opaque/cutout material は depth cascade に入り、transparent material は transmittance cascade に入ります。
