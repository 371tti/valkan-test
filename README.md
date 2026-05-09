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
10. Reflection pass / main pass を command buffer に記録して present する

## Tree

```text
src/
  app/
    mod.rs          # winit の ApplicationHandler、ウィンドウのライフサイクル、フレーム時間管理
    input.rs        # winit のキー入力を scene のキー割り当てへ変換
  demo/
    scene.rs        # SceneController のサンプル実装
    camera.rs       # フリーカメラの入力と移動
    model_loading.rs
    math.rs
  renderer/
    mod.rs          # Renderer の状態と公開 API
    lifecycle/
      init.rs       # Vulkan の instance/device/swapchain/pipeline 作成
      drop.rs       # Vulkan リソースの破棄順序
    passes/
      reflections.rs
      draw.rs       # reflection pass + main pass のコマンド記録
    assets/
      cpu.rs        # glTF/OBJ/MTL/image を CPU 側データとして読み込み
      gpu.rs        # バッファ、テクスチャ、ディスクリプタ、レンダーターゲット
      mod.rs
    pipeline/
      mod.rs        # シェーダ読み込み、ホットリロード、グラフィックスパイプライン構築
    scene/
      mod.rs        # render-scene のデータモデルと数学型
    uniforms.rs     # scene の uniform buffer とオブジェクトの push constants
    math.rs         # renderer 内部向けの 3D 補助関数
```

## 描画パイプラインの流れ

1. `app` が window/input/timing を管理し、毎フレーム `SceneController::scene` を呼ぶ
2. `demo` がカメラと読み込むモデルを決めて `RenderScene` を返す
3. `renderer::assets::cpu` が glTF/OBJ/MTL/image を CPU データへ変換する
4. `renderer::assets::gpu` が CPU データを Vulkan buffer/image/descriptor へアップロードする
5. `renderer::passes::reflections` が反射用の camera/uniform/target を準備する
6. `renderer::passes::draw` が reflection pass と main pass の command buffer を記録する
7. `Renderer::draw` が acquire -> submit -> present の順に GPU へ渡す

## draw の中身

1. CPU が fence を待ち、今の frame slot を再利用できることを確認する
2. `acquire_next_image` で描画対象の swapchain image を取得する
3. reflection target と scene uniform を更新する
4. command buffer に reflection pass と main pass を記録する
5. `queue_submit` で描画を送る
6. `queue_present` で描画結果を window system に渡す

## モデルの読み込み

起動時は次の順でモデルを探します。

1. `MODEL_PATH` 環境変数
2. `assets/model.glb`
3. `assets/model.gltf`
4. `assets/model.obj`

モデルが見つからない場合は組み込みの cube を描画します。

## 用語

[words-def.md](./words-def.md) よめー 

# めも

src/renderer/
  lifecycle/   # init / drop
  passes/      # draw / reflections
  assets/      # cpu.rs: 読み込み, gpu.rs: アップロード/descriptor
  pipeline/    # graphics pipeline
  scene/       # RenderScene / Camera / Material
  uniforms.rs
  math.rs

上にまとめる. ok./