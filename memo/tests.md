# テスト方針

## 自動で残すもの

- protocol の clamp/serialization、snapshot、asset importer、visibility、LOD
- graph の pass/resource/state/barrier、専用 GodRay の route 分岐
- Vulkan API 契約（GodRay/SMAA render pass、descriptor、format、cleanup）
- shader compile / SPIR-V validation
- `RenderFeatureToggles` のビット往復、1〜4 の ON/OFF 集合、連続品質値の不変条件
- 最終画像の受け渡し契約: `post -> smaa_edges -> smaa_weights -> smaa -> readback -> present`

## 自動 gate から外すもの

実画像を比較しないまま banding、ghosting、halo、flicker、AA の見た目を合否にするテスト。これらは手動確認へ移す。smoke は asset loaded frame、present 数、validation error、skipped frame、timeout を保証する。

## 実行記録

- `cargo fmt --all -- --check`: pass
- `cargo check --workspace --release --target-dir .codex-target`: pass
- `cargo test --workspace --target-dir .codex-target`: pass
- high smoke: 専用 camera-ray GodRay の target、descriptor、履歴、validation を確認
- `1,4,1,4` smoke: CSM/品質切替後の view 更新、resource lifetime、validation を確認
- high smoke: 影中・低散乱時の GodRay 合成修正後も validation と asset-loaded frame が通ることを確認
- 旧3Dボリュームの target/pipeline/shader/diagnostic が生成されないこと: source 全探索で確認

## 手動確認

- `4`（high）で GodRay: 光源と視線が直交に近い角度、カメラ回転、前後移動。平面層、箱状 clip、履歴 ghosting がないこと
- `4`（high）で影へ移動: 明るい物体が画面外/不在でも、GodRay の透過率が画面全体を急激に暗くしないこと
- 光源を連続移動: phase が方向・強度・色・移動量に追従し、層が平均されること
- PCSS: 同一 cascade 内で遠近・斜面・カメラ角度を比較。`pcss_visibility_raw`、`pcss_history`、最終画像を順に見る
- 透明影: foliage/glass の重なりで direct light が過減衰しないこと
- SMAA/SSAO: `smaa_edges`、`smaa_weights`、`ssao_gtao` の中間値と最終画像を比較

CSM 境界は今回の層状化の原因候補から除外済み。画面に異常が残る場合は専用 camera-ray のサンプリング、shadow 評価、temporal 再投影を分離して調べる。
