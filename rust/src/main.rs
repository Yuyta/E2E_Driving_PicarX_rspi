pub mod api;
pub mod autopilot;
mod model_cli;
mod picarx;
mod rpi_camera;
pub mod safety;
mod sensor_monitor;
pub mod tts;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    // `picar-x-rust model ...` は走行モデルの版の管理だけをして終わる（サーバーは起動しない）
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("model") {
        std::process::exit(model_cli::run(&args[1..]));
    }

    println!("=================================================");
    println!(" Starting PiCar-X Agentic AI Server (Port: 8080)");
    println!("=================================================");

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    if let Err(e) = api::run_server(port).await {
        eprintln!("Fatal server error: {}", e);
        std::process::exit(1);
    }

    Ok(())
}
