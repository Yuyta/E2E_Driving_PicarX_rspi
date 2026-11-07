pub mod api;
mod picarx;
mod rpi_camera;
pub mod safety;
mod sensor_monitor;
pub mod tts;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

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
