//! windowの作成とAppのイベント処理、また最上位構造体の定義をしてるよ
//! 
//! んぺ＾＾

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes},
};

use crate::renderer::Renderer;

const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;
const DEFAULT_TITLE: &str = "TEST";

#[derive(Debug, Clone)]
pub struct WindowConfig {
    title: String,
    size_px: PhysicalSize<u32>,
    max_size_px: Option<PhysicalSize<u32>>,
    min_size_px: Option<PhysicalSize<u32>>,
    resizable: bool,
    transparent: bool,
}

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,

    // 作成前の設定
    config: WindowConfig,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            config: WindowConfig::default(),
        }
    }
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: DEFAULT_TITLE.into(),
            size_px: PhysicalSize::new(DEFAULT_WIDTH, DEFAULT_HEIGHT),
            max_size_px: None,
            min_size_px: None,
            resizable: true,
            transparent: false,
        }
    }
}

impl App {
    pub fn window(&self) -> Option<Arc<Window>> {
        self.window.as_ref().map(Arc::clone)
    }

    pub fn window_title<S: Into<String>>(mut self, title: S) -> Self {
        self.config.title = title.into();
        self
    }

    pub fn window_size(mut self, width: u32, height: u32) -> Self {
        self.config.size_px = PhysicalSize::new(width, height);
        self
    }

    pub fn window_max_size(mut self, width: u32, height: u32) -> Self {
        self.config.max_size_px = Some(PhysicalSize::new(width, height));
        self
    }

    pub fn window_min_size(mut self, width: u32, height: u32) -> Self {
        self.config.min_size_px = Some(PhysicalSize::new(width, height));
        self
    }

    pub fn window_resizable(mut self, resizable: bool) -> Self {
        self.config.resizable = resizable;
        self
    }

    pub fn window_transparent(mut self, transparent: bool) -> Self {
        self.config.transparent = transparent;
        self
    }

    pub fn set_window_title<S: Into<String>>(&self, title: S) {
        if let Some(window) = &self.window {
            window.set_title(&title.into());
        }
    }

    // 成功したら新しいサイズを返す、失敗したら現在のサイズを返す
    //
    // request_inner_size は「要求」なので、実際に変更されたサイズは WindowEvent::Resized で見る。
    pub fn req_window_size(&self, width: u32, height: u32) -> PhysicalSize<u32> {
        if let Some(window) = &self.window {
            if let Some(size) = window.request_inner_size(PhysicalSize::new(width, height)) {
                return size;
            }

            return window.inner_size();
        }

        self.config.size_px
    }

    // 固定サイズにしたからといってResizeイベントが来ないわけではないぞー
    pub fn set_window_resizable(&self, resizable: bool) {
        if let Some(window) = &self.window {
            window.set_resizable(resizable);
        }
    }

    pub fn set_window_max_size(&self, width: u32, height: u32) {
        if let Some(window) = &self.window {
            window.set_max_inner_size(Some(PhysicalSize::new(width, height)));
        }
    }

    pub fn clear_window_max_size(&self) {
        if let Some(window) = &self.window {
            window.set_max_inner_size(None::<PhysicalSize<u32>>);
        }
    }

    pub fn set_window_min_size(&self, width: u32, height: u32) {
        if let Some(window) = &self.window {
            window.set_min_inner_size(Some(PhysicalSize::new(width, height)));
        }
    }

    pub fn clear_window_min_size(&self) {
        if let Some(window) = &self.window {
            window.set_min_inner_size(None::<PhysicalSize<u32>>);
        }
    }
}

impl ApplicationHandler for App {
    /// windowが作成可能になったときに呼び出される
    /// Valkan surface の作成はwindow作成後に行う必要があるためこれが終わってからSurfaceを作成するんだろなと
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let mut attrs = WindowAttributes::default()
            .with_inner_size(self.config.size_px)
            .with_title(self.config.title.clone())
            .with_resizable(self.config.resizable)
            .with_transparent(self.config.transparent);

        if let Some(max_size) = self.config.max_size_px {
            attrs = attrs.with_max_inner_size(max_size);
        }

        if let Some(min_size) = self.config.min_size_px {
            attrs = attrs.with_min_inner_size(min_size);
        }

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let renderer = Renderer::new(Arc::clone(&window));
        
        self.renderer = Some(renderer);
        self.window = Some(window);
    }

    /// 文字通りイベントを受け取るやつ
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            // フレームの描画が必要なときに呼び出される
            // 描画処理を実際に呼ばれてからするとは限らないと思っておいた方が良いかもしれない(くどい)
            WindowEvent::RedrawRequested => {
                if let Some(ref mut renderer) = self.renderer {
                    renderer.draw();
                }
            }

            // ウィンドウのサイズが変更されたときに呼び出される
            // ValkanではSwapchain Imageのサイズ変更が必要なためここでいじいじしないと 
            // Window resized
            //   ↓
            // old swapchain is no longer suitable
            //   ↓
            // wait device idle or synchronize
            //   ↓
            // destroy old image views / swapchain
            //   ↓
            // create new swapchain
            //   ↓
            // continue rendering
            WindowEvent::Resized(size) => {
                // ここで renderer.resize(size.width, size.height) を呼ぶ予定
                log::debug!("window resized: {}x{}", size.width, size.height);
                if let Some(ref mut renderer) = self.renderer {
                    renderer.resize(size.width, size.height);
                    renderer.draw();
                }
            }

            _ => {}
        }
    }

    // 待ち ブロッキング
    // ここで新しいeventがくるまで待機できる
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}