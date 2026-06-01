//! Window/event-loop integration.
//!
//! `App` owns the OS window, forwards input to the active scene, and asks the
//! renderer to draw each frame.

mod input;

use std::{sync::Arc, time::Instant};

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{CursorGrabMode, Window, WindowAttributes},
};

use crate::renderer::{Renderer, SceneContext, SceneController, SceneMessage};

const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;
const DEFAULT_TITLE: &str = "TEST";

#[derive(Debug, Clone)]
pub struct WindowConfig {
    title: String,
    size_px: PhysicalSize<u32>,
    min_size_px: Option<PhysicalSize<u32>>,
    transparent: bool,
}

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    scene: Box<dyn SceneController>,
    started_at: Instant,
    last_frame_at: Instant,
    frame: u64,
    mouse_captured: bool,

    config: WindowConfig,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: DEFAULT_TITLE.into(),
            size_px: PhysicalSize::new(DEFAULT_WIDTH, DEFAULT_HEIGHT),
            min_size_px: None,
            transparent: false,
        }
    }
}

impl App {
    pub fn new(scene: impl SceneController + 'static) -> Self {
        let now = Instant::now();

        Self {
            window: None,
            renderer: None,
            scene: Box::new(scene),
            started_at: now,
            last_frame_at: now,
            frame: 0,
            mouse_captured: false,
            config: WindowConfig::default(),
        }
    }

    pub fn window_title<S: Into<String>>(mut self, title: S) -> Self {
        self.config.title = title.into();
        self
    }

    pub fn window_size(mut self, width: u32, height: u32) -> Self {
        self.config.size_px = PhysicalSize::new(width, height);
        self
    }

    pub fn window_min_size(mut self, width: u32, height: u32) -> Self {
        self.config.min_size_px = Some(PhysicalSize::new(width, height));
        self
    }

    pub fn window_transparent(mut self, transparent: bool) -> Self {
        self.config.transparent = transparent;
        self
    }

    fn is_drawable_size(size: PhysicalSize<u32>) -> bool {
        size.width > 0 && size.height > 0
    }

    fn render(&mut self) {
        if self.renderer.is_none() {
            return;
        }

        let now = Instant::now();
        let size = self
            .window
            .as_ref()
            .map(|window| window.inner_size())
            .unwrap_or(self.config.size_px);

        if !Self::is_drawable_size(size) {
            self.last_frame_at = now;
            return;
        }

        let delta_time = now
            .duration_since(self.last_frame_at)
            .as_secs_f32()
            .min(1.0 / 15.0);
        let metering = self
            .renderer
            .as_ref()
            .map(Renderer::camera_metering)
            .unwrap_or_default();
        let context = SceneContext {
            elapsed: now.duration_since(self.started_at).as_secs_f32(),
            delta_time,
            frame: self.frame,
            window_size: [size.width, size.height],
            metering,
        };

        self.last_frame_at = now;
        self.frame += 1;
        self.scene.on_message(SceneMessage::RedrawRequested);

        let scene = self.scene.scene(context);
        if let Some(renderer) = &mut self.renderer {
            renderer.draw(&scene);
        }
    }

    fn set_mouse_captured(&mut self, captured: bool) {
        let Some(window) = &self.window else {
            self.mouse_captured = captured;
            return;
        };

        if captured {
            let grab_result = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            if let Err(err) = grab_result {
                log::warn!("failed to grab cursor: {err}");
                self.mouse_captured = false;
                window.set_cursor_visible(true);
                return;
            }
        } else if let Err(err) = window.set_cursor_grab(CursorGrabMode::None) {
            log::warn!("failed to release cursor: {err}");
        }

        self.mouse_captured = captured;
        window.set_cursor_visible(!captured);
    }
}

impl ApplicationHandler for App {
    /// Called once the event loop can create a native window.
    ///
    /// Vulkan surface creation depends on the native window handles, so renderer
    /// initialization starts here instead of in `App::new`.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let mut attrs = WindowAttributes::default()
            .with_inner_size(self.config.size_px)
            .with_title(self.config.title.clone())
            .with_transparent(self.config.transparent);

        if let Some(min_size) = self.config.min_size_px {
            attrs = attrs.with_min_inner_size(min_size);
        }

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let mut renderer = Renderer::new(Arc::clone(&window));
        let size = window.inner_size();
        self.scene.on_renderer_ready(&mut renderer);

        self.renderer = Some(renderer);
        self.window = Some(window);
        self.set_mouse_captured(true);
        self.scene.on_message(SceneMessage::Started {
            window_size: [size.width, size.height],
        });
    }

    /// Routes window events into scene messages and renderer lifecycle calls.
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.scene.on_message(SceneMessage::CloseRequested);
                event_loop.exit();
            }

            WindowEvent::RedrawRequested => {
                self.render();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if input::is_escape(event.physical_key) && event.state == ElementState::Pressed {
                    self.set_mouse_captured(false);
                }

                self.scene.on_message(SceneMessage::Keyboard {
                    key: input::scene_key(event.physical_key),
                    pressed: event.state == ElementState::Pressed,
                });
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => self.set_mouse_captured(true),

            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(position) => {
                        (position.y as f32 / 32.0).clamp(-4.0, 4.0)
                    }
                };
                self.scene.on_message(SceneMessage::MouseWheel { delta });
            }

            WindowEvent::Resized(size) => {
                log::debug!("window resized: {}x{}", size.width, size.height);
                self.scene.on_message(SceneMessage::Resized {
                    width: size.width,
                    height: size.height,
                });
                if let Some(ref mut renderer) = self.renderer {
                    renderer.resize(size.width, size.height);
                }
                if Self::is_drawable_size(size) {
                    self.render();
                } else {
                    self.last_frame_at = Instant::now();
                }
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if !self.mouse_captured {
            return;
        }

        if let DeviceEvent::MouseMotion { delta } = event {
            self.scene.on_message(SceneMessage::MouseMotion {
                delta: [delta.0 as f32, delta.1 as f32],
            });
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window
            && Self::is_drawable_size(window.inner_size())
        {
            window.request_redraw();
        }
    }
}
