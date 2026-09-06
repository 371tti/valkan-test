# rebuild1 作業メモ

このファイルを短い索引にし、実装・検証・次の判断をここへ集約する。

## 構造

`src/app` → `protocol` → `renderer/graph` → `renderer/vulkan` → `shaders`

- `src/app`: 入力、品質プリセット、asset load、smoke 終了条件
- `protocol`: app と renderer の所有データ契約
- `renderer/graph`: pass/resource/state/barrier の宣言と依存順
- `renderer/vulkan`: device、swapchain、target、pipeline、command recording、寿命管理
- `shaders`: build.rs で SPIR-V 化する shader と descriptor 契約

## 現行の描画経路

```text
Stable CSM / translucent shadow
  -> scene + PCSS visibility history
  -> dedicated camera-ray GodRay (48 strata + temporal history)
  -> post composition (PostColor)
  -> SMAA edges -> weights -> neighbourhood
  -> readback -> present
```

- 旧3Dボリューム経路、専用target、compute shader、診断モード、device fallbackは削除済み
- GodRayの低品質screen-space radial経路は volumetric feature がOFFのプロファイルだけで使用し、専用camera-ray経路とは同一frameで合成しない
- TAA は production default では dormant。PCSS と GodRay の履歴は別の ping-pong resource
- SSAO は GTAO 型の horizon search + cosine arc integration。SMAA は論文どおり3 pass

## 品質契約

- `RenderFeatureToggles`: 機能の ON/OFF
- `RenderQualitySettings`: sample/step/resolution/radius/intensity の連続値
- `SetQualityFeatures` は連続値を変更しない。キー `1..4` は各機能集合と連続プロファイルを選ぶ
- PCSS の解像度・tap・bias・blur 半径は連続値のまま維持する

## 検証

- `cargo fmt --all -- --check`
- `cargo check --workspace --release --target-dir .codex-target`
- `cargo test --workspace --target-dir .codex-target`
- `scripts/verify-renderer.ps1 -Mode smoke -Quality high -SmokeFrames 6`
- `scripts/verify-renderer.ps1 -Mode smoke -Quality balanced -QualitySequence '1,4,1,4' -SmokeFrames 5`

自動検証は API、数値、graph、shader compile、validation、asset loaded frame を担当する。AA、Sun Shaft の層、PCSS のちらつきなど、画面を見ないと判定できないものは手動確認に置く。

## 現在の重点

- 旧経路削除後の専用 GodRay graph が mask → temporal のみになることを確認済み
- 画面で side-on の層状化、箱状 clip、履歴 ghosting、透明影の過減衰を確認する
- PCSS は CSM 境界を原因候補から除外済み。近距離・斜面・カメラ角度・連続光源移動で raw/history/final を比較する
