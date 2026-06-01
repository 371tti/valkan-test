# Vulkan/Ash learning project

Rust + `winit` + `ash` で、Vulkan の初期化からモデル描画までを追うための学習用プロジェクトです。

## 実装ステップ

1. `winit` でウィンドウとイベントループを作る
2. `ash::Entry` / `Instance` を作る
3. Debug validation layer を有効化する
4. Window から Vulkan surface を作る
5. Physical device と queue family を選ぶ
6. Logical device / queue を作る
7. Swapchain / depth target / command buffer を作る
8. Shader module と graphics pipeline を作る
9. CPU 側の mesh / texture / material を GPU resource へアップロードする
10. Shadow / reflection / scene / post pass を command buffer に記録して present する

## Tree

```text
src/
  app/
    mod.rs              # winit ApplicationHandler、window lifecycle、frame timing
    input.rs            # winit input -> scene key
  demo/
    scene.rs            # SceneController のサンプル
    camera.rs           # free camera
    model_loading.rs    # MODEL_PATH / assets/model.* の探索
  renderer/
    mod.rs              # Renderer state と公開 API
    lifecycle/
      init.rs           # Vulkan instance/device/swapchain/pipeline 作成
      frame.rs          # draw frame
      swapchain.rs      # swapchain rebuild
      drop.rs           # Vulkan resource teardown
    passes/
      shadows.rs
      reflections.rs
      draw.rs
      rendering.rs
    assets/
      cpu.rs            # CPU asset data と loader entry
      cpu/gltf.rs       # glTF import
      cpu/obj.rs        # OBJ/MTL import
      gpu.rs            # GPU asset upload、descriptor、render targets
      image.rs          # Vulkan image helpers
    gpu/
      image_layout.rs
    pipeline/
      mod.rs            # graphics pipeline description/build
      shader.rs         # shader code loading / hot reload watch
      reload.rs         # pipeline rebuild
    scene/
      world.rs          # RenderScene, ids, light, reflection settings
      camera.rs
      controller.rs
      material.rs
      transform.rs
      math.rs
      uniforms.rs
    camera/
      metering.rs
    pass_schedule.rs    # pass update cadence
```

## 描画パイプライン

1. `app` が window/input/timing を管理し、毎フレーム `SceneController::scene` を呼ぶ
2. `demo` がカメラ、ライト、読み込んだモデルを含む `RenderScene` を返す
3. `assets::cpu` が glTF/OBJ/MTL/image を CPU データへ変換する
4. `assets::gpu` が CPU データを Vulkan buffer/image/descriptor へアップロードする
5. `passes::shadows` / `passes::reflections` が補助 target を更新する
6. `passes::draw` が scene target と swapchain への描画 command を記録する
7. `Renderer::draw` が acquire -> submit -> present を行う

## モデル読み込み

起動時は次の順でモデルを探します。

1. `MODEL_PATH` 環境変数
2. `assets/model.glb`
3. `assets/model.gltf`
4. `assets/model.obj`

モデルが見つからない場合、組み込み cube などの代替アセットは描画しません。scene はモデル無しで動きます。

## 設計の反省点

このプロジェクトは Vulkan を学びながら機能を継ぎ足したため、動くものはできた一方で、設計としてはかなり苦しい部分があります。

### Renderer が大きすぎる

`Renderer` が instance/device/swapchain/sync/assets/pipelines/render targets/pass schedule/debug utils を全部持っています。Vulkan の lifetime を追いやすい反面、どの変更も `Renderer` 全体へ波及しやすいです。

次回は `DeviceContext`、`SwapchainContext`、`FrameResources`、`RenderTargets`、`AssetStore`、`PassSystem` のように、所有権と lifetime が近い単位で分けるべきです。

### 初期化が一枚岩

`lifecycle/init.rs` に Vulkan instance 作成、device 選択、swapchain 作成、target 作成、descriptor 作成、pipeline 作成が集まっています。読む順番は分かるものの、失敗時の切り分けや差し替えが難しいです。

次回は builder 風に分けるよりも、成果物ごとの小さな factory に分けます。例えば `create_device_context`、`create_swapchain_resources`、`create_scene_bindings`、`create_pipelines` のように、返す構造体を明確にする方がよいです。

### GPU asset と render target が混ざっている

`assets/gpu.rs` は mesh/material/texture upload、descriptor set、buffer helper、shadow map、reflection probe、scene target まで抱えています。これは「GPU resource」という括りが広すぎた結果です。

次回は asset upload と render target を分離します。`GpuAssets` はモデル・マテリアル・テクスチャに限定し、`targets/` や `descriptors/` を別 module にします。

### Asset import を renderer に入れすぎた

glTF/OBJ/MTL/image の読み込みが `renderer::assets::cpu` にあります。レンダラが扱いやすい CPU 表現へ変換する意図はありますが、ファイル形式ごとの複雑さが renderer の中に入り込みました。

次回は import crate/module と renderer asset module を分けます。import 側は `ImportedScene` のような中間表現を返し、renderer 側はそれを upload するだけにします。

### Render graph がない

shadow、reflection、scene、post の順序や image layout 遷移を手で管理しています。pass が増えるほど、どの image がいつ writable/readable なのかが見えにくくなります。

次回は最初から小さな render graph / frame graph を用意します。最低でも pass ごとの inputs/outputs/layout を宣言して、barrier と実行順を一箇所で決めるべきです。

### Pipeline と shader interface が手作業

Rust 側の descriptor binding、push constants、GLSL 側の layout が暗黙に一致している前提です。変更時に片側だけ直してもコンパイルでは気づきにくいです。

次回は shader interface を宣言的にまとめます。理想は reflection か codegen、最低でも Rust 側に `SceneSetLayout` / `MaterialSetLayout` のような名前付き binding 定義を置き、GLSL の binding と対応を明示します。

### ID がただの index

`MeshId` / `MaterialId` / `TextureId` / `ModelId` は `usize` wrapper です。削除や再利用を始めると stale handle を検出できません。

次回は generation 付き handle か、削除しない append-only store として明示します。asset unload をするなら `slotmap` 的な設計に寄せます。

### フォールバックが設計を曇らせた

以前はモデル読み込み失敗時に組み込み cube を描画していました。起動確認には便利でしたが、ロード失敗と描画成功が混ざり、実際に何が起きているか分かりにくくなりました。

次回は fallback asset を暗黙に使いません。失敗は失敗としてログに出し、必要なら demo 側が明示的に placeholder を選ぶ形にします。

### テストがレンダリング品質を保証していなかった

OBJ/GLTF のフィールド値を見るテストは、実際にレンダリングしたときの見え方を保証しません。マテリアル、法線、alpha、texture slot は最終的に shader、descriptor、pipeline state と組み合わさらないと正しさを判断しにくいです。

次回は unit test を純粋関数に限定します。レンダラの品質は、固定 scene の screenshot 比較、GPU validation layer、RenderDoc capture、または小さな offscreen render test で確認します。

## 次回作るなら

最初から次の境界で始めるのがよさそうです。

```text
renderer/
  device/        # instance, device, queues, allocator
  frame/         # frames in flight, command pools, sync
  swapchain/     # surface, swapchain, resize
  graph/         # pass declarations, resources, barriers
  targets/       # depth, shadow, reflection, scene color
  assets/        # GPU asset store only
  import/        # glTF/OBJ/image import to intermediate scene
  pipeline/      # shader modules, layouts, pipeline cache
  scene/         # renderable scene data
```

方針:

- Vulkan object の owner を一箇所に決める
- `unsafe` と破棄順序を resource wrapper に閉じ込める
- pass は `record_*` 関数の集合ではなく、inputs/outputs を持つ小さな型にする
- shader binding は名前付き定数か codegen で同期する
- asset import と GPU upload を分ける
- fallback は demo policy として扱い、renderer core に入れない
- unit test は math、scheduling、layout description など純粋ロジックに限定する
- 見た目は screenshot/offscreen render test で見る

## 用語

[words-def.md](./words-def.md) よめー
