mod rpi_camera;
mod picarx;
mod sample;
mod llm_agent;
mod sensor_monitor;

fn main() {
    if let Err(e) = sample::picarx_sample::main() {
        eprintln!("Fatal error: {}", e);
        std::process::exit(1);
    }
}

