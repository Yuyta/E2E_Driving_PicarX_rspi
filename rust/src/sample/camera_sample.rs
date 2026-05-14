use ndarray::Array3;
use show_image::{create_window, WindowOptions, ImageInfo, PixelFormat, ImageView};
use crate::rpi_camera::RpiCamera;

#[show_image::main]
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting real-time camera using OS native command...");

    let width = 640;
    let height = 480;
    let mut rgb_buffer = Array3::<u8>::zeros((height, width, 3));

    // open camera device
    let mut camera = RpiCamera::new(width, height, 30)?;
    println!("★ Camera stream started! Reading raw frames...");

    // create window
    let window = create_window(
        "PiCar-X Rust Camera Live",
        WindowOptions { preserve_aspect_ratio: true, ..Default::default() },
    )?;
    let event_channel = window.event_channel()?;

    'main_loop: loop {
        // fetch latest-rgb-frame
        if !camera.fetch_latest_frame_to_rgb(&mut rgb_buffer) {
            break;
        }

        // create image_view
        let rgb_slice = rgb_buffer.as_slice().unwrap();
        let image_view = ImageView::new(
            ImageInfo::new(PixelFormat::Rgb8, width as u32, height as u32),
            rgb_slice
        );

        // set image_view to window
        if window.set_image("live_frame", image_view).is_err() {
            break;
        }

        // key-input event (Q or ESC)
        while let Ok(event) = event_channel.try_recv() {
            if let show_image::event::WindowEvent::KeyboardInput(key_event) = event {
                if key_event.input.state == show_image::event::ElementState::Pressed {
                    match key_event.input.key_code {
                        Some(show_image::event::VirtualKeyCode::Escape) | Some(show_image::event::VirtualKeyCode::Q) => {
                            println!("Exit signal received.");
                            break 'main_loop;
                        }
                        _ => {}
                    }
                }
            }
        }

        // thread::sleep(Duration::from_millis(2));
    }

    println!("Stopping camera...");
    Ok(())
}

