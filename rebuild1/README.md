# rebuild1 design

`rebuild1/` は次の renderer を設計しながら実装していく作り直し用 crate です。前回のコードは `old/` に退避済みで、ここでは設計ドキュメントと実装を同時に保守します。

作業中の短い索引と修正状態は、リポジトリ直下の [`memo/README.md`](../memo/README.md) を先に読みます。問題一覧は [`memo/issues.md`](../memo/issues.md)、テストの線引きは [`memo/tests.md`](../memo/tests.md) にあります。

## 設計の芯

`rebuild1` は、app / user code / renderer を async message protocol で分離し、renderer 専用 thread に Vulkan を閉じ込める設計です。

将来 ECS を上に載せても、renderer は ECS world を読みません。user/ECS 側で render extraction を行い、owned な `FrameSnapshot` を renderer に送ります。

実装時は読みやすさを最適化対象にします。短い関数、明示された制約、boundary validation、constrained type を優先し、すべての関数に「具体的に何をする関数か」を書きます。

## 前作から持ち越さないもの

- 読み込み失敗時の暗黙 cube fallback
- 何でも持つ巨大な `Renderer`
- `assets/gpu.rs` に render target まで入れる構造
- renderer 内にファイル形式 importer を抱え込む構造
- pass 順序と image layout 遷移の手管理
- Rust 側 descriptor binding と shader layout の暗黙同期
- 見た目を保証しない unit test の増殖

## 設計ドキュメント

通常はこの 3 つだけ読めば現在地が分かるようにします。

- [compact_design.md](docs/compact_design.md): まず読む圧縮版。現在地、境界、次の gate
- [roadmap.md](docs/roadmap.md): 実装順、完了条件、未決事項
- [agent_implementation_notes.md](docs/agent_implementation_notes.md): 実装 agent に渡す境界、禁止事項、実装順の固定メモ

以下は迷ったときに読む reference です。

- [design_summary.md](docs/design_summary.md): 全体設計の短いまとめ、最終決定、読む順番
- [code_generation_policy.md](docs/code_generation_policy.md): 直近の実装で守る短期ポリシー、module 境界、完了チェック
- [architecture.md](docs/architecture.md): 全体構成、責務境界、初期化と frame flow
- [messaging.md](docs/messaging.md): app / user code / renderer をつなぐ command、event、snapshot protocol
- [async_runtime.md](docs/async_runtime.md): 最初から thread 分割する async runtime、task、channel 方針
- [aaa_traits.md](docs/aaa_traits.md): AAA 指向へ拡張するための境界、data flow、trait 方針
- [ecs_integration.md](docs/ecs_integration.md): 将来 ECS を載せるための render extraction 境界
- [code_quality.md](docs/code_quality.md): 短く、安全で、読みやすいコードを書くための制約
- [render_graph.md](docs/render_graph.md): pass、resource、barrier、resize の設計
- [assets.md](docs/assets.md): importer、GPU asset store、handle、fallback 方針
- [shader_pipeline.md](docs/shader_pipeline.md): shader interface、pipeline、hot reload 方針
- [testing.md](docs/testing.md): unit test と rendering test の線引き

## 現在の実装メモ

- app は headless path と winit window path を持つ。
- protocol は command/event/envelope/id/snapshot/surface/transport に分かれている。
- renderer の連続値（samples/steps/resolution/radius/intensity/look）は `SetQualitySettings(RenderQualitySettings)` で送る。機能のON/OFFだけを切り替える `SetQualityFeatures(RenderFeatureToggles)` は別コマンドとし、window のキー1〜4は対応する連続品質プロファイルとON/OFF集合を順に送る。キー1=全OFF＋performance、2=AA+SSAO＋interactive、3=そこへSSR+Bloom+GodRay＋balanced、4=さらにVolumetric Fog＋high quality。ECS owned な `FrameSnapshot` には混ぜない。
- renderer は dedicated thread 上で `RendererBackend` を動かす。
- Vulkan backend は instance / validation debug callback / device / surface / swapchain / 4-layer Stable CSM depth array / scene・post・SMAA (edge/weight/blend) render pass / framebuffer / frame resources / mesh / material / post pipeline を持ち、TAA は有効時だけ遅延確保する。
- `SubmitFrame` は target surface に対して acquire、graph compile、barrier record、必要な Stable CSM / translucent shadow pass、scene pass、専用 GodRay の camera-ray mask・temporal volume（低品質時だけ prefilter・radial・temporal）、bloom/post composition、SMAA edge detection・weight calculation・neighborhood blending、readback、submit、present を行う。
- present 待ちの semaphore は swapchain image ごとに持ち、frame slot ごとに再利用しない。
- window path は surface configure 後に redraw-driven の最小 `SubmitFrame` loop を走らせる。
- renderer graph は `stable_csm_depth[cascade 0..3] -> optional translucent_shadow -> scene(HDR+metadata) -> dedicated GodRay volume -> optional TAA -> bloom/post composition(PostColor) -> smaa_edges -> smaa_weights -> smaa(swapchain) -> optional readback -> present` を実行計画として持ち、resource state から explicit barrier plan を生成する。SMAA の edge/weight は専用中間 target、color neighborhood は composition 済み `PostColor` だけを読み、SSR/SSAO/bloom/GodRay/camera grading 後の画像へ適用する。既定は空間 SMAA、TAA は現在 dormant。CSMは4層を1つの安定したlight-space gridで更新し、PCSSの最終filterは比較サンプラのhardware PCFを使う。
- shader source は機能別に分けた Slang (`crates/gr-render/shaders/{shared,scene,shadow,post}/`) を `build.rs` で SPIR-V にする。`slangc` と `spirv-val` を使い、生成 asset registry は `renderer::vulkan::shader` に集約する。temporary debug triangle pipeline は削除済みで、asset 未ロード時は scene clear frame を present する。
- SMAA は最終 `PostColor` を入力に、公式 1x の local-contrast edge、SearchTex line search、AreaTex pattern coverage、directional neighborhood resolve を評価する。edge detector の `PostColor` tap は point read、`edgesTex`/`blendTex` と LUT の検索は linear+clamp、最終色の resolve も linear read とし、事前に色を平均してエッジを消さない。
- SSAO は固定近傍比較ではなく、GTAO型のview-space horizon searchとcosine-weighted analytic arc integrationを使う。品質値は2/3/4 slices × 2/3/4 steps（8/18/32 taps）へ展開し、平方距離分布、near-field falloff、receiver bias、thin-occluder compensationを適用する。TAAがdormantのため、ノイズ回転はフレーム間で固定し、まず単フレームの接触・平面・細線挙動を安定させる。
- 通常の feature-disabled path は local light がない frame で専用 `SCENE_FAST_PATH` shader を選ぶ。local-light code と normal/material metadata target への書き込みを省き、opaque variant では alpha discard も compile しない。さらに機能ビットと連続値の両方が無効なら専用 post fast pipeline を選び、scene color 1 sample と共有 camera effects / tone mapping だけを実行する。
- god ray の経路は feature bit で選択し、連続な Fog 密度・距離・サンプル予算は `SetQualitySettings` で調整する。`high_quality()` の本番 route は専用 half-resolution GodRay camera-ray volume（48 個別 jitter strata）で analytic height medium を積分し、view-depth を ray cosine で実距離化する。directional CSM/PCSS と deep translucent log-transmittance、Henyey-Greenstein位相、ambient in-scatteringを同じ ray 上で評価し、Beer-Lambert透過を適用する。短い raster far clip でも medium range は設定距離まで維持し、半解像度 bilateral upscale 後に PostColor へ合成して SMAA へ渡す。各 frame の sample phase/light motion/feedback は共有 temporal state から供給し、mask/history の再投影と履歴 clamp を行う。PCSS は scene MRT の visibility と receiver depth だけを別の ping-pong history へ保存し、BRDF/直接光は現フレーム値を使う。カメラ・light・surface・品質変更では両履歴を破棄する。
- Vulkan timestamp 対応環境では、`RUST_LOG=gr_render::renderer::vulkan::gpu_timing=trace` を指定すると通常描画command bufferへpass checkpointを追加する。fence完了後にcommand buffer全体の `gpu_frame_ms`、passごとの `gpu_pass_ms`、最終記録pass後の `gpu_command_buffer_tail_ms` をtraceし、present待ちを含むCPU時間はGPU描画時間として扱わない。
- Stage 4 first draw は window capture で triangle 表示確認済み。
- winit resize は command channel を詰まらせないよう、in-flight 1 件と pending 最新 1 件に coalesce する。stale な resize は live surface capabilities で解決した extent を現在値と比較し、同一なら swapchain と pipeline を再生成しない。
- `FrameSnapshot` は `SurfaceId` / `SurfaceGeneration` を持ち、古い generation の frame は `FrameDropped` で落とす。
- Stage 5 asset path は `.r1scene` importer skeleton、`LoadAsset` / `AssetLoaded` / `AssetLoadFailed`、`GpuAssetStore`、deferred destroy queue まで実装済み。CPU asset import は FIFO・同時実行最大 1 件で非同期化し、読込中も renderer thread が frame、resize、shutdown に応答できる。
- `FrameSnapshot` は正式な mesh draw 入力として `render_items` を持つ。temporary `DrawPacket` / debug triangle 経路は削除済み。
- Stage 6 material/texture path は named slot、alpha mode、validated `TextureDescriptor`、`MaterialDescriptor`、shader binding constants まで実装済み。
- renderer assets は `store.rs` / `mesh.rs` / `material.rs` / `texture.rs` / `garbage.rs` に分割済み。
- Stage 7 の mesh rendering slice は完了済みで、`.r1scene` / GLB geometry は Vulkan backend の device-local vertex/index buffer へ upload する。asset store は mesh の vertex/index count だけを保持し、upload 後の CPU geometry を重複保持しない。
- glTF の encoded PNG/JPEG image decode、RGB -> RGBA conversion、mesh packing / LOD generation は、入力順を維持した最大 8 worker の上限付き並列処理で行う。image decode は小さい入力を serial path に残し、thread 起動コストを避ける。Vulkan resource の作成と upload は renderer thread に残す。
- `old/assets/model.glb` は app-level sample として `rebuild1/assets/model.glb` にコピー済み。window path はこのファイルが存在するときだけ `LoadAsset` を送り、全 mesh/material pair を `render_items` として submit する。
- `render_items` は `vulkan/mesh.rs` の mesh pipeline で indexed draw を記録する。`FrameSnapshot` は owned `CameraSnapshot` を持ち、mesh shader は app-side camera の view-projection で world-space GLB を描画する。
- window path は old-style free camera controls を持つ。left click で cursor capture、Escape で release、WASD/arrow、Space/E、Shift/Q、Ctrl、mouse wheel で移動する。
- directional shadow は device-owned の単一 D16 array（4 cascade）へ保存する。CSMの段数とsplitは固定し、共有解像度・tap数・biasなどの連続値は `SetQualitySettings` で個別に変更する。キー1〜4はこの連続プロファイルも performance / interactive / balanced / high へ切り替える。各cascadeはcamera frustumを包む固定sphereから正方形projectionを作り、隣接cascadeの受け面をオーバーラップさせ、light-space中心を1 shadow texel単位でsnapするため、カメラ移動でshadowが泳がない。directional map はカメラ/光源/キャスター集合と resource 状態を署名し、変化がないフレームでは再描画せず永続配列を再利用する。local cubemapは独立したdepth resourceを維持する。
- blocker探索だけがraw depth samplerを使い、raw depthはLINEARでcoverage遷移を滑らかにする。最終PCSS filterは `compareEnable = true`、`LESS_OR_EQUAL`、LINEAR filter の `SampleCmpLevelZero` を使う。各タップがVulkanの2x2 hardware PCFを利用し、オーバーラップ区間では隣接層をブレンドする。Receiver Plane Depth Biasでタップ位置ごとの受け面深度勾配も補正し、規則的な横縞を抑える。biasはD16の2段量子化余裕、texel/depth-span、receiverのslope、receiver-planeの1 texel勾配を合成し、品質設定のreceiver/slope/normal/plane scaleで調整する。PCSSの回転seedは連続ワールド座標のsinハッシュを使わず、ラスタピクセルとcascadeを整数混合した基底へ共有64フレーム低差異位相を加えるため、サブユニット周期のパターンを作らない。blocker平均は量子化幅内をsoft weightし、比較用biasとpenumbra計算用receiver深度を分離する。品質設定はblocker/filter tap数、太陽角半径、receiver/slope/normal/plane biasを制御する。
- glTF import は node transform を vertex へ bake した後、同一 material の近接した opaque/cutout primitive だけを上限付き spatial batch にまとめる。camera/CSM culling 境界を保つため遠い geometry は混ぜず、描画順が必要な transparent primitive は結合しない。
- glTF の `MASK` triangle は、bilinear filter と `REPEAT` を含む完全な texture footprint を summed-area alpha classifier で保守的に判定する。全 footprint が確実に opaque の triangle だけを opaque material へ分離し、確実に transparent の triangle は除去する。判定不能または mixed の triangle は cutout のまま残す fail-closed specialization とする。
- mesh LOD は三角形を単純に間引かない。meshoptimizer の `Medium = LockBorder`、`Low = None`、`VeryLow = Regularize` で index LOD を作り、normal / material UV を attribute error に渡す。vertex-cache reorder と opaque/cutout の overdraw reorder は維持し、transparent の triangle order は変えない。
- directional shadow geometry のLOD選択は共有shadow-mapの実解像度と投影texel budgetに従う。CSMの解像度をカスケードごとに変えたり、近距離専用の段を追加したりせず、全層同一の解像度で遠距離のsub-texel triangleだけを送らない。
- GPU vertex は 12-byte position stream と 20-byte surface stream の SoA に分ける。opaque shadow path は position stream だけを bind し、alpha-cutout shadow path だけが material UV を含む surface stream も読む。
- translucent directional shadowは基準方向（direction 0）だけを描画し、opaque D16 arrayをsampleして最前面casterのRGBA16F transmittance/depthを記録する。4 cascadeのfixed-function D32 depthはpassごとにclearする単一scratchを共有する。scene shaderは色付き透過を基準方向の結果へ一度だけ適用し、方向サンプル数に比例した重複合成を避ける。
- swapchain dependent resource は scene color/depth、normal/material、PCSS visibility のフル解像度 ping-pong history、post composition target/framebuffer、SMAA の RG `edgesTex` / RGBA `blendTex` target と、機能を有効にしたときだけ作る TAA history を持つ。TAA は stable-grid reprojection、2x2 depth候補、同一surfaceのnormal/count、YCoCg variance clampを実装するが、現在の production path では無効。post composition pipeline は scene HDR と metadata を読み、material reflectance aware SSR、SSAO、bloom、GodRay、camera effects、tone mappingを `PostColor` へ書く。SMAA の3本のpipelineが `PostColor -> edgesTex -> blendTex` を通し、swapchain へ書く。旧screen-space shadow blurは持たない。
- `vulkan/material.rs` は imported texture payload を sampled image へ upload し、material parameter buffer と descriptor set を作る。暗黙 fallback texture は作らない。
- Stage 8 で GLB base-color texture、vertex normal、material texture sampling、alpha cutout shadow、scene shadow sampling、post camera effects を通した。
- Stage 9 shadow slice で fixed cascade shadow、translucent shadow pass、cascade descriptor indexing、sampled opaque-depth rejection、transparent shadow smoke scene を通した。
- camera effects は `FrameSnapshot::camera_effects` として app/user 側で抽出し、renderer は露出/ホワイトバランスの owned 値だけを post pass に適用する。contrast/saturation の renderer-wide 補正は quality command 側で扱う。
- lighting quality は low ambient、directional wrap、indirect strength を分ける。完全に光がない場所は暗く残しつつ、画面外や背面側の key light で normal terminator が急に黒く落ちる問題を抑える。
- mesh pipeline は scene / opaque shadow / translucent shadow ごとに untextured/textured variant を持ち、texture descriptor がある material だけ texture sampling shader を使う。
- `REBUILD1_WINDOW_ASSET=assets/stage8_textured_cutout.r1scene` で fixed texture/cutout verification scene を window smoke に流せる。
- `REBUILD1_WINDOW_ASSET=assets/stage9_translucent_shadow.r1scene` で transparent shadow verification scene を window smoke に流せる。
- `--window-smoke` は既存の asset path を必須とし、asset load 後に指定数の mesh frame を present して自動終了する。`REBUILD1_WINDOW_QUALITY=high` は feature preset 4（Volumetric/専用 GodRay volume をON）を指定し、連続値予算は renderer の現在設定を維持する。`verify-renderer.ps1 -Mode smoke -QualitySequence '1,4,1,4'` では機能ビットの切替を反復検証できる（PowerShellでは列を文字列として渡す）。
- asset load 失敗時に renderer が cube や placeholder を作る経路はない。
- まだ独立した user task、reflection JSON を使った descriptor codegen、screenshot golden image は未実装。

## 最終方針

- app / user code / renderer は直接結合しない。
- renderer は `RendererCommand` を受け取り、`RendererEvent` を返す。
- Vulkan object は renderer thread だけが所有する。
- ECS world から renderer へ渡すのは render extraction 済みの `FrameSnapshot` だけ。
- renderer core は暗黙 fallback をしない。
- asset import、GPU upload、render target、render graph、pipeline を分ける。
- pass dependency と image layout transition は render graph に集める。
- shader binding は名前付き interface に集約する。
- safety は boundary validation と constrained type で作る。
- すべての関数に具体的な説明を書く。
- trace log を主に使い、info は重要な lifecycle event に絞る。

## 目標構成

```text
app/
  window         # winit event loop and window owner

protocol/
  messages/      # command, event, request id, handle
  material       # material slots, alpha mode, texture descriptors
  transport/     # async bounded channel, record/replay bridge

user/
  app trait      # async user code entry point
  ecs            # 将来の ECS world owner
  extract        # ECS/user state -> FrameSnapshot

renderer/
  mod.rs         # renderer thread, backend trait, null backend
  assets/        # GPU asset handles, mesh/material/texture records, deferred destroy queue
  surface.rs     # null/backend-neutral surface registry
  vulkan.rs      # Vulkan backend orchestration
  pass_schedule.rs # frame pass cadence outside backend recording
  vulkan/
    buffer.rs    # host-visible typed buffer upload helper
    debug.rs     # validation layer and debug messenger
    swapchain.rs # swapchain image views, fixed shadows, scene/post resources
    frame.rs     # frames in flight, command pools, sync, frame execution
    material.rs  # sampled images, sampler, material parameter buffers, descriptor sets
    mesh.rs      # backend-local mesh vertex/index buffers and mesh pipeline
    post.rs      # post composition and three-pass SMAA pipelines
    taa.rs       # optional temporal history route (dormant by default)
  graph/         # pass declarations, resources, barriers
  targets/       # depth, shadow, scene color
  pipeline/      # shader interface, shader modules, layouts, pipeline cache
  scene/         # renderable scene data

import.rs        # .r1scene skeleton and GLB geometry importer
```
