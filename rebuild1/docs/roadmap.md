# Roadmap

## 実装前の完了条件

実装を始める前に次を決めます。

- module boundary
- agent implementation notes
- short-term code generation policy
- message protocol
- async runtime policy
- task/thread topology
- ECS integration boundary
- code quality rules
- user app trait
- renderer backend trait
- descriptor set layout
- 最小 render graph の resource/pass model
- asset handle policy
- fallback policy
- rendering test の最初の手順

## Stage 0: design freeze

完了条件:

- `docs/architecture.md` の責務境界に納得している
- `docs/design_summary.md` の最終方針と読む順番に納得している
- `docs/compact_design.md` で現在地と次の gate を把握できる
- `docs/agent_implementation_notes.md` の renderer 境界、Vulkan 隠蔽、実装順に納得している
- `docs/code_generation_policy.md` の短期実装ポリシーと stop signs に納得している
- `docs/messaging.md` の command/event/snapshot 境界に納得している
- `docs/async_runtime.md` の async task / renderer thread 方針に納得している
- `docs/ecs_integration.md` の ECS world / render extraction / FrameSnapshot 境界に納得している
- `docs/code_quality.md` の関数説明、制約付き型、guard 方針に納得している
- `docs/aaa_traits.md` の trait 方針に沿って、抽象化する場所としない場所が分かれている
- `docs/render_graph.md` の初期 pass 構成で最初の実装が切れる
- `docs/assets.md` の importer と asset store の境界が曖昧でない
- `docs/shader_pipeline.md` の descriptor set 方針が決まっている
- `docs/testing.md` の test 方針に沿って、不要 test を増やさない判断ができる

## Stage 1: platform and device

作るもの:

- [x] window/event loop
- [x] async runtime
- [x] `UserApp` async trait skeleton
- [x] `RendererCommand` / `RendererEvent`
- [x] async bounded `RendererTransport`
- [x] dedicated renderer thread
- [x] Vulkan instance/device/queue
- [x] debug validation layer
- [x] surface/swapchain
- [x] frames-in-flight skeleton

完了条件:

- [x] resize しても落ちない
- [x] validation error がない
- [x] resource owner が docs と一致している
- [x] app/user code から renderer へ直接 Vulkan 状態を触る経路がない
- [x] renderer task が Vulkan object を単独所有している
- [x] 追加した関数すべてに具体的な説明がある
- [x] boundary validation と内部関数の制約が分かれている

軌道修正:

- `vulkan.rs` が巨大化し始めたため、debug utils と swapchain resources を module 分割した。
- resize は command queue に全件積まず、latest state へ coalesce する。
- 次に command pool / command buffer / sync を追加する前に frame module を作る。

## Stage 2: protocol-driven clear

作るもの:

- `ConfigureSurface` command
- `ResizeSurface` command
- `SubmitFrame` command
- `FramePresented` event
- minimal `FrameSnapshotBuilder`
- [x] swapchain image view / render pass / framebuffer
- [x] clear-only renderer service
- [x] app/window 側の redraw-driven `SubmitFrame` loop
- [x] renderer thread 上の clear/present loop

完了条件:

- direct method call ではなく async command channel で clear/present できる
- command/event log だけで frame の流れを追える
- `FrameSnapshot` は参照や Vulkan handle を含まない
- `FrameSnapshot` は ECS Entity や component reference を含まない
- main thread は winit callback で `await` しない
- `FrameSnapshotBuilder` は validation 済みの constrained type を返す

## Stage 3: graph skeleton

作るもの:

- [x] graph resource 宣言
- [x] single clear pass
- [x] swapchain image transition

完了条件:

- [x] graph が clear -> present だけを管理する
- [x] barrier が pass callback の外にある
- [x] resize 後に graph を再構築できる

## Stage 4: first draw

作るもの:

- [x] hardcoded triangle vertex buffer
- [x] frame uniform
- [x] basic pipeline

完了条件:

- [x] triangle が出る
- [x] debug pipeline の descriptor set order が `frame/material/pass` と一致する
- [x] shader source が build 時に compile され、compile error が見える
- [x] validation layer で first draw の重大 error が出ない

Stage 4 から外したもの:

- full frame/material/pass descriptor contents は shader interface validation で扱う
- shader hot reload error logging は shader reload 実装時に扱う

## Stage 5: assets without fallback

前提:

- [x] `SurfaceId` / `SurfaceGeneration` が protocol に入っている
- [x] `FrameSnapshot` が target surface generation を持っている
- [x] stale frame を `FrameDropped` / `DropReason::StaleSurfaceGeneration` として扱える
- [x] GPU resource の deferred destroy 方針が入っている
- [x] temporary debug triangle path を削除し、正式な `render_items` 経路へ集約した

作るもの:

- [x] importer skeleton
- [x] `LoadAsset` / `AssetLoaded` / `AssetLoadFailed`
- [x] intermediate scene
- [x] GPU asset store
- [x] explicit user model loading の protocol path

完了条件:

- [x] 読み込み失敗時に renderer が何も勝手に出さない
- [x] user code が placeholder を出す場合は user code 側に明示されている
- [x] asset handle の stale policy が決まっている

実装メモ:

- Stage 5 の importer は `.r1scene` manifest だけを読む最小 skeleton。
- ファイル読み取りは worker task で行い、renderer thread では imported scene の handle 登録だけを行う。
- `LoadAsset` 失敗は `AssetLoadFailed` として返し、cube や placeholder は作らない。
- `GpuAssetStore` は protocol handle を発行し、`UnloadAsset` で stale 化して deferred destroy queue に積む。
- asset 未ロード時は debug triangle ではなく scene clear frame を submit する。

## Stage 6: materials and texture

作るもの:

- [x] named material slots
- [x] texture payload registration
- [x] material descriptor
- [x] alpha mode
- [x] shader binding constants for material slots

完了条件:

- [x] `.r1scene` の textured plane manifest が importer/store path を通る
- [x] alpha cutout material が import/store path を通る
- [x] material slot と shader binding の番号が `shader_interface` から追える

実装メモ:

- Stage 6 は `protocol/material.rs`、`renderer/assets/material.rs`、`renderer/assets/texture.rs` に分割した。
- `.r1scene` は `texture solid r g b a` と `material cutout base_color=0 alpha_cutoff=0.5` を扱う。
- material は imported texture index を protocol `TextureHandle` へ解決する。
- texture は validated `TextureDescriptor` として store に入り、暗黙 white texture は作らない。
- Vulkan mesh pipeline での実描画は Stage 7 で扱う。sampled image / sampler / material descriptor set は texture rendering gate で扱う。

## Stage 7: real passes

作るもの:

- [x] imported mesh -> renderer-owned vertex/index geometry
- [x] Vulkan vertex/index buffer upload
- [x] GLB geometry import for the app-level model check
- [x] mesh draw packet execution
- [x] camera snapshot in `FrameSnapshot`
- [x] old-style mouse/keyboard free camera in the window app
- [x] all loaded mesh/material pairs extracted as draw packets
- [x] depth target attached to the swapchain scene pass
- [x] scene pass with color/depth rendering
- [x] pass schedule for shadow / scene / post / present cadence

Stage 7 から外したもの:

- Stage 7.5 以降で進めるもの:
  - sampled material texture を mesh shader に反映する

完了条件:

- [x] pass dependency が graph に見える
- [x] pass update cadence が graph の外にある
- [x] layout transition を pass 内で手書きしない
- [x] `old/assets/model.glb` を window で load し、全 mesh の indexed draw が validation error なしで走る
- [x] camera は app/user 側 state であり、renderer へは owned `CameraSnapshot` だけが渡る

実装メモ:

- `renderer/assets/mesh.rs` を追加し、`.r1scene` の `mesh plane` は 4 vertices / 6 indices の renderer-owned geometry に変換する。
- `GpuAssetStore` は mesh handle を単なる active set ではなく geometry record として保持する。
- `vulkan/buffer.rs` を追加し、host-visible buffer 作成と typed upload を mesh/material/post 系で共有する。
- `vulkan/mesh.rs` を追加し、`LoadAsset` で device がある場合は mesh handle ごとに vertex/index buffer を作る。
- `old/assets/model.glb` を `rebuild1/assets/model.glb` にコピーし、window app が存在するときだけ app policy として `LoadAsset` を送る。
- GLB は worker importer で triangle primitive を `ImportedMesh::Indexed` に変換する。renderer core は `assets/model.glb` の探索を知らない。
- mesh pipeline は `vulkan/mesh.rs` に閉じ込め、scene pass 内の `render_items` は indexed draw を記録する。
- `ImportedScene` は bounds を持ち、`AssetLoaded` で app 側 camera が model を frame する。
- GLB の importer 側 clip-space 正規化は削除し、world-space position を `CameraSnapshot` の view-projection で描画する。
- window app は left click で cursor capture、Escape で release、WASD/arrow、Space/E、Shift/Q、Ctrl、mouse wheel を old-style free camera として処理する。
- scene pass は scene color + depth attachment を持ち、mesh pipeline は depth test/write を有効化する。
- post composition pass は scene color を sampled image として読み、`PostColor` へ fullscreen triangle で書く。専用 SMAA edge pass が `edgesTex`、weight pass が `blendTex` を生成し、neighbourhood pass が `PostColor` と `blendTex` を読み swapchain image へ書く。
- `pass_schedule.rs` は frame snapshot から shadow / scene / post / present の cadence を作り、Vulkan recording の外で trace できるようにする。

## Stage 7.5: graph compiler foundation and real targets

作るもの:

- [x] `GraphPass` / `GraphResourceDecl` / read-write usage
- [x] pass の read/write から依存 edge を自動生成
- [x] topological sort と cycle detection
- [x] resource lifetime と barrier plan
- [x] unused pass culling
- [x] barrier merge / transient alias / render pass merge の候補情報
- [x] scene color + depth -> post composition -> SMAA edge/weight/neighbourhood 3-pass -> swapchain present の graph
- [x] actual post process shader
- [x] `--window-smoke` で GLB load 後の graph-driven mesh frame を検証
- [x] Vulkan imported texture image upload
- [x] material descriptor set upload
- [x] fixed cascade shadow targets と opaque shadow passes
- [x] translucent shadow transmittance targets と transparent shadow passes

完了条件:

- [x] pass order を手書き固定しない
- [x] graph が read/write から必要な barrier を生成する
- [x] scene/post/swapchain の target lifetime が graph から読める
- [x] shadow cascade / translucent shadow の target lifetime が graph から読める
- [x] cadence は `pass_schedule.rs` に残し、dependency は graph が管理する
- [x] Vulkan object は renderer backend から外へ出ない

実装メモ:

- 最初から巨大な AAA graph compiler にはしないが、toy 実装にはしない。今の compiler は deterministic な plan、barrier、lifetime、optimization hint を返す。
- Vulkan executor は fixed swapchain pass ではなく、compiled graph の pass/barrier を実際に記録する。
- fake shadow graph は置かない。shadow cascade、translucent shadow、scene color/depth、swapchain image はすべて graph resource と Vulkan resource が対応している。
- shadow pass は `render_items` の opaque/cutout caster を depth-only pipeline で記録する。
- translucent shadow pass は opaque D16 arrayをsampleし、cascade間で共有して毎pass clearするD32 scratchのdepth test/writeで最近傍casterのtransmittanceだけを記録する。
- scene pass は shadow cascade と translucent transmittance を graph read として宣言する。
- imported texture は renderer asset store の owned `TextureDescriptor` clone から Vulkan sampled image へ upload する。
- material は named slot と alpha policy を descriptor set へ upload する。暗黙 white texture は作らない。
- mesh shader の sampled material 表示と shader interface validation は Stage 8 で完了済み。残りは high-quality dedicated GodRay smoke と必要な visual acceptance の運用です。

## Stage 8: verification

作るもの:

- [x] fixed test scenes
- [x] screenshot/manual capture 手順
- [x] validation layer checklist
- [x] sampled material texture shader path
- [x] shader interface validation
- [x] renderer-side mesh visibility optimization

完了条件:

- [x] 少なくとも triangle、texture、alpha cutout、shadow、post camera effects を確認できる
- [x] unit test と rendering test の役割が混ざっていない
- [x] mesh draw は bounds から画面外 culling と screen-size LOD を選べる

実装メモ:

- GLB importer は vertex color、inverse-transpose normal、負スケール時の winding 補正、base-color / normal / metallic-roughness / occlusion / emissive slot、PBR scalar、emissive strength、double-sided policy を CPU intermediate data として持ち、renderer thread では file format を読まない。
- `.r1scene` は `texture checker ...` を明示 directive として持ち、`assets/stage8_textured_cutout.r1scene` で texture / alpha cutout / shadow cutout を smoke できる。
- mesh pipeline は `scene` / `opaque shadow` / `translucent shadow` ごとに untextured/textured variant と cull/double-sided variant を持つ。asset fallback は使わず、glTF の省略 slot は明示的な material default map で descriptor を満たす。
- scene mesh shader は material descriptor set と graph pass descriptor set を読み、opaque shadow cascade と translucent transmittance を scene lighting に反映する。
- shadow resources は swapchain extent に依存しない fixed resources として device にぶら下げる。near/mid/far cascade は camera frustum に合わせ、scene 全体の bounds で解像度を薄めない。
- post pass は scene color を tone map し、`FrameSnapshot::camera_effects` の露出/white balance を適用する。
- auto exposure / white balance は app/user extraction 側で計算し、renderer には owned な camera effect scalar だけを渡す。ほぼ黒の metering では露出を強く上げず、光のない場所を暗く残す。
- camera metering は renderer event 経由の final framebuffer readback を使う。app 側は owned な luminance/color summary だけを受け取り、Vulkan object は renderer 境界の外へ出さない。
- `FrameSnapshot::optimization` は frustum culling と screen-size LOD のポリシーだけを持つ。renderer は mesh upload 時に bounds と coarse index LOD を backend-local に作り、scene pass の draw recording 直前で不要な mesh を捨てる。
- mesh pipeline は back-face culling を有効化する。double-sided material が必要になったら pipeline variant と material policy として明示的に追加する。
- `REBUILD1_WINDOW_ASSET=assets/stage8_textured_cutout.r1scene cargo run -- --window-smoke` で fixed verification scene を選べる。
- `REBUILD1_WINDOW_ASSET=assets/stage9_translucent_shadow.r1scene cargo run -- --window-smoke` で transparent shadow verification scene を選べる。
- `scripts/verify-renderer.ps1 -Mode smoke -Quality high -SmokeFrames 1` で dedicated GodRay の API/validation 経路を確認できる。
- `assets/model.glb` smoke は old model 相当の geometry/material/texture import と camera 操作確認の入口として残す。

## Open questions

- TODO: Stage 7.5 の graph compiler を、async compute scheduling まで扱える executor に育てる。
- ECS crate は自作するか、既存 crate を使うか
- `ExternalObjectId` を debug/picking 用に最初から入れるか
- shader interface は codegen するか、まず constants と include で始めるか
- allocator は自前 wrapper にするか、既存 crate を使うか
- asset unload を最初から入れるか、append-only で始めるか
- screenshot comparison を最初から CI に入れるか
- message log をどの粒度で保存するか
- async runtime は tokio で確定するか、runtime 抽象を残すか
