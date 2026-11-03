pub mod api;
pub mod audio_input;
mod llm_agent;
pub mod memory;
mod picarx;
mod rpi_camera;
pub mod safety;
mod sample;
mod sensor_monitor;
pub mod tts;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let args: Vec<String> = std::env::args().collect();
    let is_gui_mode = args.iter().any(|arg| arg == "--gui");

    if is_gui_mode {
        println!("PiCar-X: 従来のGUIモードを起動します (--gui)");
        if let Err(e) = sample::picarx_sample::main() {
            eprintln!("Fatal error in GUI mode: {}", e);
            std::process::exit(1);
        }
    } else {
        println!("=================================================");
        println!(" Starting PiCar-X Agentic AI Server (Port: 8080)");
        println!("=================================================");
        println!("※ 従来のGUIを起動したい場合は `--gui` 引数を指定してください");

        let port: u16 = std::env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8080);

        if let Err(e) = api::run_server(port).await {
            eprintln!("Fatal server error: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
