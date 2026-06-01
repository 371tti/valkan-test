mod demo;

use demo::MainScene;
use valkan_test::app::App;
use winit::event_loop::EventLoop;

fn main() {
    #[cfg(debug_assertions)]
    {
        env_logger::init_from_env(env_logger::Env::default().filter_or("LOG_LEVEL", "trace"));
    }

    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(MainScene::default())
        .window_title("Ash Vulkan")
        .window_size(2400, 1600)
        .window_min_size(240, 160)
        .window_transparent(false);

    event_loop.run_app(&mut app).unwrap();
}
