Vulkan および Ash クレートを用いてグラフィックスプログラミングをする際の用語定義

## 0. 全体構造の用語

| 用語       | 定義                                                                 |
| -------- | ------------------------------------------------------------------ |
| Vulkan   | 低レベルグラフィックス / GPU API。GPUに対して、バッファ作成、メモリ管理、描画命令、同期、表示などを明示的に指示する。  |
| ash      | Rust用のVulkanバインディング。Vulkan C APIにかなり近い薄いラッパー。                      |
| GPU      | 描画や並列計算を実行するプロセッサ。Vulkanでは `Device` 側。                             |
| CPU      | アプリケーション本体を実行し、GPU用のリソースやコマンドを作る側。Vulkanでは `Host` と呼ばれることも多い。      |
| Host     | CPU側のこと。Vulkan APIを呼ぶ側。                                            |
| Device   | GPU側のこと。ただし Vulkan の `Device` は「論理デバイス」を指す場合が多い。                   |
| VRAM     | GPU側のメモリ。頂点データ、テクスチャ、描画先画像などを置く。                                   |
| Resource | GPUが使うデータ。Buffer、Image、Samplerなど。                                  |
| Object   | Vulkanで作成する管理対象。Instance、Device、Swapchain、Buffer、Image、Pipelineなど。 |

## 1. Vulkan初期化まわり

| 用語               | 定義                                                        |
| ---------------- | --------------------------------------------------------- |
| Entry            | Vulkanライブラリへの入口。`ash::Entry`。ここから `Instance` を作る。         |
| Instance         | Vulkan API全体の実行環境。アプリケーションとVulkanランタイムの接続点。               |
| ApplicationInfo  | アプリ名、エンジン名、Vulkan APIバージョンなどをInstance作成時に渡す情報。            |
| Extension        | Vulkan本体に追加する機能。Surface作成、Swapchain、Debugなどは拡張機能として扱われる。  |
| Layer            | Vulkan API呼び出しに割り込む追加層。代表例がValidation Layer。              |
| Validation Layer | Vulkanの使い方が間違っていないか検査してくれるデバッグ用Layer。Vulkan開発ではほぼ必須。      |
| Debug Messenger  | Validation Layerからの警告・エラーを受け取るための仕組み。                     |
| Loader           | VulkanドライバやLayerを探してAPI呼び出しを中継する仕組み。OS上にあるVulkanランタイムの一部。 |

## 2. デバイスまわり

| 用語               | 定義                                                              |
| ---------------- | --------------------------------------------------------------- |
| PhysicalDevice   | 実際のGPU。NVIDIA GPU、AMD GPU、Intel iGPUなど。                         |
| LogicalDevice    | アプリが使うために作成するGPU操作用ハンドル。`ash::Device`。                          |
| Device Extension | LogicalDeviceで有効化する拡張機能。Swapchainを使うなら `VK_KHR_swapchain` が必要。  |
| Feature          | GPU機能の有効/無効。例: geometry shader、sampler anisotropy、wide linesなど。 |
| Limit            | GPUの制限値。最大テクスチャサイズ、最大Uniform Bufferサイズなど。                       |
| Property         | GPUの性質。名前、ベンダーID、デバイス種別、対応APIバージョンなど。                           |
| Queue Family     | GPU上のキューの種類。Graphics、Compute、Transfer、Presentなどの能力を持つ。          |
| Queue            | GPUにCommand Bufferを投入する実行列。                                     |
| Graphics Queue   | 描画コマンドを実行できるQueue。                                              |
| Present Queue    | Swapchain Imageを画面表示へ渡せるQueue。Graphics Queueと同じ場合もある。           |
| Compute Queue    | Compute Shaderを実行できるQueue。                                      |
| Transfer Queue   | Buffer/Image間のコピーなど転送処理に特化したQueue。                              |

## 3. ウィンドウ・表示まわり

| 用語              | 定義                                                                       |
| --------------- | ------------------------------------------------------------------------ |
| Window          | OS上のウィンドウ。Rustでは今回 `winit` で作る。                                          |
| Surface         | VulkanとOSのウィンドウシステムを接続するオブジェクト。描画結果をどこに表示するかを表す。                         |
| WSI             | Window System Integration。VulkanがWindows/Linux/macOS等のウィンドウシステムと連携する仕組み。 |
| Swapchain       | 表示用Imageを複数枚管理する仕組み。画面に出すためのバックバッファ群。                                    |
| Swapchain Image | Swapchainが持つ描画先Image。毎フレーム1枚取得して描画し、Presentする。                           |
| Acquire         | 次に描画してよいSwapchain Imageを取得する操作。`acquire_next_image`。                     |
| Present         | 描画済みSwapchain Imageを画面表示側に渡す操作。                                          |
| Present Mode    | 表示タイミングの方式。VSyncあり/なしなどに関係する。                                            |
| FIFO            | 基本的なVSyncありPresent Mode。ほぼ必ず対応している。                                      |
| MAILBOX         | 低遅延なVSync系Present Mode。古いフレームを捨てて最新フレームを表示しやすい。                          |
| IMMEDIATE       | VSyncなしで即時表示。ティアリングが出る可能性がある。                                            |
| Surface Format  | Swapchain Imageのピクセル形式と色空間。例: `B8G8R8A8_SRGB`。                           |
| Extent          | 描画領域のサイズ。だいたいウィンドウのピクセル幅・高さ。                                             |
| Compositor      | OSの画面合成システム。複数ウィンドウの内容を最終画面に合成する。                                        |

## 4. Image / Buffer / Memory

| 用語             | 定義                                                                     |
| -------------- | ---------------------------------------------------------------------- |
| Buffer         | 線形メモリ領域。頂点、インデックス、Uniform、Storage、コピー元/先などに使う。                         |
| Image          | 2D/3D画像リソース。テクスチャ、Depth Buffer、Swapchain Image、Render Targetなど。        |
| ImageView      | Imageの見え方を定義するオブジェクト。ShaderやFramebufferからImageを使うには多くの場合ImageViewが必要。  |
| Texture        | ShaderからサンプリングするImage。Vulkanでは通常 `Image + ImageView + Sampler` の組み合わせ。 |
| Sampler        | テクスチャをどう読むかを定義する。補間、繰り返し、ミップマップ、異方性フィルタなど。                             |
| DeviceMemory   | Vulkanで確保するGPUメモリ。BufferやImageにbindして使う。                               |
| Memory Type    | GPUが提供するメモリ種別。CPUから見えるか、GPU専用か、キャッシュされるか等が違う。                          |
| HOST_VISIBLE   | CPUからmapできるメモリ。CPUで書き込める。                                              |
| DEVICE_LOCAL   | GPUに近い高速メモリ。VRAM相当。CPUから直接触れない場合がある。                                   |
| HOST_COHERENT  | CPU書き込み後のflushが不要になりやすいメモリ。                                            |
| HOST_CACHED    | CPU読み書きがキャッシュされるメモリ。                                                   |
| Map            | GPUメモリをCPUアドレス空間に対応付けること。                                              |
| Flush          | CPUが書いた内容をGPUから見えるように反映する操作。非coherentメモリで必要。                           |
| Invalidate     | GPUが書いた内容をCPU側キャッシュに反映する操作。                                            |
| Alignment      | メモリ配置の境界制約。Uniform Bufferなどで重要。                                        |
| Staging Buffer | CPUから書き込みやすい一時Buffer。そこからGPU専用Buffer/Imageへコピーする。                      |
| Render Target  | 描画結果を書き込むImage。Color Attachmentなど。                                     |
| Depth Buffer   | 奥行き情報を保存するImage。Depth Testに使う。                                         |
| MSAA Image     | マルチサンプルアンチエイリアス用のImage。最終的にSwapchain Imageへresolveする。                  |

## 5. Commandまわり

| 用語                       | 定義                                                           |
| ------------------------ | ------------------------------------------------------------ |
| Command                  | GPUに実行させる命令。描画、コピー、バリア、クリアなど。                                |
| Command Buffer           | GPUに送るCommandを記録する入れ物。CPUが記録し、Queueへsubmitする。                |
| Command Pool             | Command Bufferを確保するためのプール。Queue Familyごとに作る。                 |
| Record                   | Command Bufferに命令を書き込むこと。                                    |
| Submit                   | Command BufferをQueueに投入すること。                                 |
| Primary Command Buffer   | 直接QueueにsubmitできるCommand Buffer。                             |
| Secondary Command Buffer | Primaryから呼び出す補助Command Buffer。大規模描画で使うことがある。                 |
| Reset                    | Command BufferやCommand Poolを再利用可能状態に戻すこと。                    |
| One-time Submit          | 一度だけ使うCommand Buffer。BufferコピーやImage Layout Transitionでよく使う。 |

## 6. 同期まわり

| 用語                   | 定義                                                                     |
| -------------------- | ---------------------------------------------------------------------- |
| Synchronization      | CPU/GPU間、GPU処理間の実行順序やメモリ可視性を制御する仕組み。Vulkanでかなり重要。                      |
| Fence                | CPUがGPU処理完了を待つための同期オブジェクト。                                             |
| Semaphore            | GPU Queue間、またはAcquire/Present間の同期に使うオブジェクト。                            |
| Binary Semaphore     | signaled / unsignaled の2状態を持つSemaphore。基本形。                            |
| Timeline Semaphore   | 数値カウンタで進行を管理するSemaphore。複雑な同期に向く。                                      |
| Pipeline Barrier     | GPU内部の処理順序とメモリ可視性を制御する命令。                                              |
| Memory Barrier       | メモリアクセスの順序と可視性を制御するBarrier。                                            |
| Image Memory Barrier | ImageのLayout Transitionやアクセス制御に使うBarrier。                              |
| Layout Transition    | Imageの用途に応じてレイアウトを変更すること。例:描画先→Present用。                               |
| Access Mask          | どの種類のメモリアクセスを同期対象にするか。例: color attachment write。                       |
| Pipeline Stage       | GPUパイプライン上の段階。Vertex Shader、Fragment Shader、Color Attachment Outputなど。 |
| Frames in Flight     | CPUが複数フレーム分の処理を先行して用意する仕組み。2〜3が一般的。                                    |

## 7. RenderPass / Attachmentまわり

| 用語                 | 定義                                                                  |
| ------------------ | ------------------------------------------------------------------- |
| RenderPass         | 描画処理の構造を定義するオブジェクト。どのAttachmentにどう描くかを表す。                           |
| Dynamic Rendering  | RenderPass/Framebufferを使わず、より直接的に描画先を指定する新しい描画方式。最近のVulkanではこちらも有力。 |
| Attachment         | RenderPassで使う描画先。Color、Depth、Stencilなど。                             |
| Color Attachment   | 色を書き込む描画先。最終的にはSwapchain Imageになることが多い。                             |
| Depth Attachment   | 奥行きを書き込む描画先。Depth Buffer。                                           |
| Stencil Attachment | ステンシル値を書き込む描画先。特殊なマスク処理などに使う。                                       |
| Framebuffer        | RenderPassで使うAttachment ImageView群をまとめるオブジェクト。一般語の「画面バッファ」とは少し違う。   |
| LoadOp             | 描画開始時にAttachmentの内容をどう扱うか。Load、Clear、DontCareなど。                    |
| StoreOp            | 描画終了時にAttachmentの内容を保存するかどうか。Store、DontCareなど。                      |
| Subpass            | RenderPass内の描画段階。複数SubpassでタイルGPU向け最適化などができる。                       |

## 8. Pipelineまわり

| 用語                      | 定義                                                                   |
| ----------------------- | -------------------------------------------------------------------- |
| Pipeline                | GPUの描画処理設定をまとめたオブジェクト。Shader、頂点入力、ラスタライズ、Depth、Blendなどを含む。           |
| Graphics Pipeline       | 通常の3D/2D描画用Pipeline。                                                 |
| Compute Pipeline        | Compute Shader実行用Pipeline。                                           |
| Pipeline Layout         | Descriptor Set LayoutとPush Constant範囲をまとめたもの。Shaderがどんな外部データを使うかの構造。 |
| Pipeline Cache          | Pipeline作成結果をキャッシュし、次回作成を高速化する仕組み。                                   |
| Shader                  | GPU上で動くプログラム。                                                        |
| SPIR-V                  | Vulkanが読み込むShaderの中間バイナリ形式。GLSLやHLSLからコンパイルする。                       |
| Shader Module           | SPIR-VをVulkanオブジェクト化したもの。                                            |
| Vertex Shader           | 頂点ごとに実行されるShader。座標変換などを行う。                                          |
| Fragment Shader         | ピクセル候補ごとに実行されるShader。色やマテリアル計算を行う。                                   |
| Geometry Shader         | プリミティブ単位で形状を増減できるShader。使わないことも多い。                                   |
| Tessellation Shader     | 曲面や細分化を扱うShader。高度な用途向け。                                             |
| Entry Point             | Shader内で実行開始される関数名。例: `main`。                                        |
| Specialization Constant | Pipeline作成時にShader内定数を差し替える仕組み。                                      |

## 9. Descriptorまわり

| 用語                     | 定義                                                    |
| ---------------------- | ----------------------------------------------------- |
| Descriptor             | ShaderがBufferやImageを参照するための情報。                        |
| Descriptor Set         | 複数のDescriptorをまとめたセット。draw時にbindする。                   |
| Descriptor Set Layout  | Descriptor Setの構造定義。binding番号、型、Shader stageなどを定義する。  |
| Descriptor Pool        | Descriptor Setを確保するためのプール。                            |
| Binding                | Descriptor Set内の番号。Shader側の `binding = 0` などに対応する。    |
| Uniform Buffer         | Shaderに小〜中規模の読み取り専用データを渡すBuffer。MVP行列など。              |
| Storage Buffer         | Shaderから大きなデータを読み書きできるBuffer。                         |
| Combined Image Sampler | ImageViewとSamplerをまとめてShaderに渡すDescriptor。テクスチャでよく使う。 |
| Input Attachment       | Subpass内で前段のAttachmentを読むためのDescriptor。               |
| Push Constant          | 小さいデータを高速にShaderへ渡す仕組み。行列やIDなどに使うことがある。               |
| Dynamic Uniform Buffer | offsetを変えて同じUniform Bufferを複数オブジェクトで使う仕組み。            |

## 10. 頂点・メッシュまわり

| 用語                    | 定義                                                   |
| --------------------- | ---------------------------------------------------- |
| Vertex                | 頂点。位置、法線、UV、色などを持つ。                                  |
| Vertex Buffer         | 頂点配列を入れるBuffer。                                      |
| Index Buffer          | 頂点の参照番号を入れるBuffer。頂点の重複を減らす。                         |
| Vertex Input          | Vertex Bufferのデータ構造をPipelineに教える設定。                  |
| Binding Description   | Vertex Bufferの1頂点あたりのサイズや入力レートを定義する。                 |
| Attribute Description | 位置、色、UVなどがVertex内のどこにあるかを定義する。                       |
| Primitive             | GPUが描く基本形状。点、線、三角形など。                                |
| Triangle List         | 3頂点ごとに独立した三角形を描く方式。                                  |
| Triangle Strip        | 頂点列から連続した三角形を作る方式。                                   |
| Mesh                  | 頂点、インデックス、マテリアルなどをまとめた形状データ。                         |
| Model                 | 複数Mesh、Transform、Material、Textureなどを含む3Dオブジェクト。      |
| glTF                  | 3Dモデル形式。PBR、Mesh、Material、Texture、Animationなどを扱いやすい。 |

## 11. 座標変換まわり

| 用語                      | 定義                                                         |
| ----------------------- | ---------------------------------------------------------- |
| Local Space             | モデル自身の座標空間。                                                |
| World Space             | シーン全体の座標空間。                                                |
| View Space              | カメラから見た座標空間。                                               |
| Clip Space              | GPUのクリッピング用座標空間。Vertex Shaderの出力先。                         |
| NDC                     | Normalized Device Coordinates。正規化デバイス座標。画面変換前の -1〜1 などの空間。 |
| Model Matrix            | Local SpaceからWorld Spaceへ変換する行列。                           |
| View Matrix             | World SpaceからView Spaceへ変換する行列。                            |
| Projection Matrix       | View SpaceからClip Spaceへ変換する行列。                             |
| MVP Matrix              | Model × View × Projection をまとめた行列。                         |
| Transform               | 位置、回転、拡大縮小。                                                |
| Camera                  | View MatrixとProjection Matrixを作るための視点。                     |
| Perspective Projection  | 遠くのものが小さく見える投影。3Dゲームで一般的。                                  |
| Orthographic Projection | 遠近感のない投影。UIやCAD的表示で使う。                                     |

## 12. ラスタライズまわり

| 用語                | 定義                                                     |
| ----------------- | ------------------------------------------------------ |
| Rasterization     | 三角形などの形状をピクセル候補に変換する処理。                                |
| Fragment          | ピクセル候補。必ず最終ピクセルになるとは限らない。                              |
| Pixel             | 画面上の最終的な画素。                                            |
| Viewport          | NDCを画面座標に変換する領域。                                       |
| Scissor           | 描画を許可する矩形領域。                                           |
| Culling           | 見えない面を描かない処理。                                          |
| Back-face Culling | 裏面ポリゴンを捨てる処理。                                          |
| Front Face        | 表面と判定する頂点の巻き順。時計回り/反時計回り。                              |
| Winding Order     | 三角形頂点の並び順。表裏判定に使う。                                     |
| Depth Test        | 奥行きに基づいて手前のFragmentだけを残す処理。                            |
| Depth Write       | Depth Bufferへ奥行きを書き込む処理。                               |
| Stencil Test      | ステンシル値に基づいて描画可否を決める処理。                                 |
| Blending          | 既存の色と新しい色を合成する処理。透明描画で使う。                              |
| Alpha             | 透明度成分。                                                 |
| MSAA              | Multi-Sample Anti-Aliasing。ピクセル内に複数サンプルを持ってエッジを滑らかにする。 |
| Resolve           | MSAA Imageを通常の1サンプルImageに変換する処理。                       |

## 13. フレーム処理の用語

| 用語                   | 定義                                          |
| -------------------- | ------------------------------------------- |
| Frame                | 1回分の描画単位。                                   |
| Framebuffer Resize   | ウィンドウサイズ変更によりSwapchainを作り直す必要がある状態。         |
| Swapchain Recreation | ウィンドウサイズ変更やSurface状態変更に応じてSwapchainを作り直す処理。 |
| In-flight Frame      | GPUがまだ処理中のフレーム。                             |
| CPU-GPU Parallelism  | CPUが次フレームの準備をしつつ、GPUが前フレームを描画する並列性。         |
| VSync                | ディスプレイ更新タイミングに同期して表示する仕組み。                  |
| Tearing              | 画面の途中で別フレームが混ざって裂けたように見える現象。                |
| Latency              | 入力から画面反映までの遅延。                              |
| Frame Pacing         | フレーム表示間隔を安定させる制御。                           |

## 14. `ash`特有の用語・感覚

| 用語            | 定義                                                        |
| ------------- | --------------------------------------------------------- |
| `ash::vk`     | Vulkanの型や定数が入っているモジュール。例: `vk::InstanceCreateInfo`。       |
| Builder風API   | `vk::XXX::default().field(...)` のように構造体を組み立てる書き方。         |
| Handle        | Vulkanオブジェクトを指す軽量な値。`vk::Buffer`、`vk::Image`など。           |
| Loader Struct | 拡張機能を呼ぶためのash側構造体。例: `ash::khr::surface::Instance`。       |
| unsafe        | Vulkan API呼び出しの多くはRust側で安全性を保証できないため `unsafe` になる。        |
| Drop          | Vulkanオブジェクト破棄処理。作成の逆順で `destroy_*` する必要がある。              |
| Null Handle   | 無効なVulkanハンドル。`vk::Buffer::null()` など。                    |
| Result        | Vulkan APIの成功/失敗結果。`vk::Result` またはRustの `Result` に変換される。 |

## 15. `winit` / window handleまわり

| 用語                 | 定義                                                 |
| ------------------ | -------------------------------------------------- |
| winit              | Rustのクロスプラットフォームウィンドウ作成・イベント処理ライブラリ。               |
| EventLoop          | ウィンドウイベントを処理し続けるループ。                               |
| ApplicationHandler | winit 0.30系のアプリイベント処理用trait。                       |
| Window             | OSウィンドウを表すwinitの型。                                 |
| WindowEvent        | リサイズ、閉じる、再描画要求などのイベント。                             |
| RedrawRequested    | 再描画要求イベント。ここで `renderer.draw()` する。                |
| raw-window-handle  | OSネイティブのウィンドウハンドルを抽象化するcrate。                      |
| ash-window         | `raw-window-handle` からVulkan Surfaceを作るための補助crate。 |

## 16. よく混同する用語

| 用語                               | 違い                                                                |
| -------------------------------- | ----------------------------------------------------------------- |
| PhysicalDevice / LogicalDevice   | PhysicalDeviceは実GPU。LogicalDeviceはアプリが使うために作った操作用オブジェクト。          |
| Image / ImageView                | Imageは実データ。ImageViewはそのImageをどう見るか。                               |
| Buffer / DeviceMemory            | Bufferは用途と範囲を表すオブジェクト。DeviceMemoryは実際のメモリ。bindして使う。               |
| Framebuffer / Swapchain Image    | FramebufferはRenderPass用のAttachment集合。Swapchain Imageは実際の表示用Image。 |
| Semaphore / Fence                | SemaphoreはGPU-GPU同期向け。FenceはCPUがGPU完了を待つ向け。                       |
| Pipeline / Shader                | ShaderはGPUプログラム。PipelineはShaderを含む描画状態全体。                         |
| Descriptor Set / Pipeline Layout | Descriptor Setは実際にbindするリソース群。Pipeline LayoutはShaderが要求するリソース構造。  |
| Command Buffer / Queue           | Command Bufferは命令列。QueueはそれをGPUに実行させる投入先。                         |
| Attachment / Texture             | Attachmentは描画先。TextureはShaderから読む画像。Imageとしては同じ仕組みを使うことがある。       |
