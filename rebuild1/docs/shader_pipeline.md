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
  post/               compose・SMAA・SSAO・SSR
    bloom/            bloom downsample/upsample
    god_rays/         camera-ray density・CSM/translucent lighting・Beer-Lambert integration・temporal resolve
    god_rays/         camera-ray mask/temporal volume; low-quality prefilter/radial compatibility branch
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

Temporal PCSS はラスタピクセル/cascade hash に共有64フレームの低差異位相を加えてタップを回し、visibility と receiver depth だけを再投影履歴へ積分します。カメラ回転/移動はPCSS専用の連続反応度として履歴保持を短くし、通常の方向光移動はその半分の反応度で履歴を残します。履歴参照は深度一致した4近傍だけを再構成します。production Sun Shaft は GodRay の camera-ray strata を同じ temporal phase/light-motion で jitter し、専用 GodRay volume の積分結果をその履歴へ渡します。

temporary debug triangle pipeline は削除済みです。first draw 後は `FrameSnapshot.render_items` が唯一の scene draw 入力になり、asset 未ロード時は scene clear frame を present します。

Stage 7 の mesh pipeline は set 0 binding 0 に `CameraSnapshot` 由来の view/projection uniform を持ちます。scene pass は scene HDR、depth、normal/material、PCSS visibility/receiver-depth のMRTを出力し、Stable CSMのPCSS visibilityをその場で反映します。PCSSの最終filterはset 2 binding 9の比較サンプラ（LINEAR + `LESS_OR_EQUAL`）でhardware 2x2 PCFを使い、binding 10のraw depthはblocker探索だけに使います。visibility history は receiver を前フレーム VP へ再投影し、可視率と受け手深度だけを ping-pong で時間積分します。履歴は深度一致した4近傍だけを手動bilinear再構成し、CSM遷移の投影無効値を補間しません。BRDF、transmittance、直接光全体は現フレーム値のままです。blocker/filter各タップにはreceiver-planeのlight-space深度勾配を加え、斜面で同じ比較深度を反復する横縞を抑えます。receiver biasはD16量子化の2段余裕、texel/depth-span、法線角度に応じたslope、receiver-planeの1 texel深度勾配を合成し、balanced/highでは各スケールを引き上げます。タップ回転はラスタピクセルとcascadeの整数hashから生成し、連続ワールド座標sinハッシュ由来の短周期パターンを避けます。high-quality の本番 GodRay shader は analytic density → camera-ray 48-strata → CSM/PCSS・deep translucent log-transmittance → Henyey-Greenstein scattering → Beer-Lambert integration → half-resolution bilateral upscale を実行し、専用 history は前フレームの共有 temporal state で再投影します。Scene/GodRay HDR と scene metadata を post composition に渡します。post composition はGTAO型SSAO / SSR / bloom / GodRay / camera effects / tone mappingを `PostColor` へ書く。SSAOはview-space normalをslice planeへ射影し、両側の最大horizonを距離フォールオフ付きで探索してcosine-weighted arcを解析積分する。quality sample countは2/3/4 slicesと2/3/4 stepsへ展開し、near-field falloffとthin-occluder compensationを適用する。SMAA edge pass が luma edge を `edgesTex` に、weight pass が SearchTex/AreaTex の pattern weight を `blendTex` に、neighbourhood pass がその同じ PostColor と blendTex に対して公式のSMAA 1xを実行します。交差サンプルは水平検索で R、垂直検索で G を読む。F12 の `smaa_edges` / `smaa_weights` は中間値を画面へ出す診断経路です。TAA shader と history layout は optional な実装として残しますが、production default は temporal color history と frame jitter を使わず、TAA resources も確保しません。SSR / SSAO / SMAA / lighting wrap / renderer-wide contrast の連続値は `SetQualitySettings`、各機能のON/OFFは `SetQualityFeatures(RenderFeatureToggles)` で更新される renderer 状態からpush constantとframe uniformへpackします。

full descriptor contents と hot reload error handling は shader interface validation / reload 実装時に扱います。

SMAA の実装メモ: 最終 `PostColor` に対して公式 1x の local-contrast edge detection、SearchTex line search、AreaTex pattern coverage、directional neighborhood resolve を3つのfullscreen passで評価する。edge passの色入力はpoint texel、edgesTexの検索/weightとblendTexはlinear/clamp、resolveの色入力もlinearとし、隣接色の事前平均によるエッジ消失を避ける。resolve の neighbor/channel 対応は公式の `(+x,+y)`（right alpha / bottom green / current blue+red）を維持し、輪郭の反対方向へぼかさない。

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
