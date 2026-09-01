pub mod audio_input;
mod llm_agent;
mod picarx;
mod rpi_camera;
mod sample;
mod sensor_monitor;
pub mod tts;

fn main() {
    if let Err(e) = sample::picarx_sample::main() {
        eprintln!("Fatal error: {}", e);
        std::process::exit(1);
    }
}
