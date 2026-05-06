# Graphicsの学習用

Step 1: winitでウィンドウを出す
Step 2: ash::Entry / Instance を作る
Step 3: Validation Layer を有効化
Step 4: Surface を作る
Step 5: PhysicalDevice を選ぶ
Step 6: LogicalDevice / Queue を作る
Step 7: Swapchain を作る
Step 8: CommandBuffer で clear color
Step 9: 三角形
Step 10: Buffer / Uniform / Texture / Depth

## 基本構造
一般的なグラフィックスプログラムの構造だからここらへんは理解すべきなんだろなと Valkanに限らない

1. CPUが描画に必要なデータ(ポリゴン, テクスチャ, GPUパイプラインの構成など)を構築
2. GPUにコマンド(描画命令,データ転送命令など)を送る
3. GPUがコマンドを実行して指定のパイプラインで描画する
4. GPUが描画結果をフレームバッファ(VRAM上の描画領域)に出力する
5. OS(のウィンドウシステム)がフレームバッファの内容をウィンドウに表示する

## 用語定義
[用語定義](./words-def.md)をよめー

## 手順
まずwindowを出す
ここはwinit(クロスプラットフォームなウィンドウライブラリ)を使ってまずイベントループ(キー入力とかOSのWMからのイベントを処理するためのループ)を作ってからウィンドウを作成する

## draw の中身
1. CPUが fence を待つ
   → この frame slot を再利用してよいか確認する

2. acquire_next_image
   → 今回描いてよい Swapchain Image を取得する
   → そのImageが使えるようになったら image_available_semaphore がsignalされる

3. queue_submit
   → image_available_semaphore をwaitする
   → つまり「Imageが描画可能になるまで描画開始しない」
   → 描画が終わったら render_finished_semaphore をsignalする

4. queue_present
   → render_finished_semaphore をwaitする
   → つまり「描画完了後に画面表示へ渡す」