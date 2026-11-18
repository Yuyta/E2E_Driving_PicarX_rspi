//! カメラの JPEG を、走行モデルの入力（1x3xHxW、正規化した f32）へ変換する。
//!
//! PC の `autonomous_driver.preprocess`（OpenCV）と同じ手順にする: 縮小 → RGB → 0〜1 → 平均と
//! 標準偏差で正規化 → NCHW。カメラは 320x240 で、入力は 160x120 なので、縮小はちょうど半分。
//! OpenCV の INTER_LINEAR はちょうど半分の縮小では 2x2 の平均になるので、同じ計算をする。

use jpeg_decoder::{Decoder, PixelFormat};

/// 縮小した RGB（HWC、u8）
#[derive(Debug, Clone)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

pub fn decode_jpeg(jpeg: &[u8]) -> Result<RgbImage, String> {
    let mut decoder = Decoder::new(jpeg);
    let pixels = decoder
        .decode()
        .map_err(|e| format!("JPEG を読めません: {e}"))?;
    let info = decoder.info().ok_or("JPEG の情報がありません")?;
    let (width, height) = (usize::from(info.width), usize::from(info.height));
    let data = match info.pixel_format {
        PixelFormat::RGB24 => pixels,
        PixelFormat::L8 => pixels.iter().flat_map(|&v| [v, v, v]).collect(),
        other => return Err(format!("対応していない画素の形式です: {other:?}")),
    };
    if data.len() != width * height * 3 {
        return Err("JPEG の大きさが合いません".to_string());
    }
    Ok(RgbImage {
        width,
        height,
        data,
    })
}

/// `width` x `height` へ縮小する。ちょうど半分なら 2x2 の平均、ほかは双線形（画素の中心で合わせる）
pub fn resize(src: &RgbImage, width: usize, height: usize) -> RgbImage {
    if src.width == width && src.height == height {
        return src.clone();
    }
    let mut data = vec![0u8; width * height * 3];
    if src.width == width * 2 && src.height == height * 2 {
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    let at =
                        |yy: usize, xx: usize| u32::from(src.data[(yy * src.width + xx) * 3 + c]);
                    let sum = at(2 * y, 2 * x)
                        + at(2 * y, 2 * x + 1)
                        + at(2 * y + 1, 2 * x)
                        + at(2 * y + 1, 2 * x + 1);
                    data[(y * width + x) * 3 + c] = ((sum + 2) / 4) as u8;
                }
            }
        }
    } else {
        let sx = src.width as f32 / width as f32;
        let sy = src.height as f32 / height as f32;
        for y in 0..height {
            let fy = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
            let y0 = (fy as usize).min(src.height - 1);
            let y1 = (y0 + 1).min(src.height - 1);
            let wy = fy - y0 as f32;
            for x in 0..width {
                let fx = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
                let x0 = (fx as usize).min(src.width - 1);
                let x1 = (x0 + 1).min(src.width - 1);
                let wx = fx - x0 as f32;
                for c in 0..3 {
                    let p =
                        |yy: usize, xx: usize| f32::from(src.data[(yy * src.width + xx) * 3 + c]);
                    let top = p(y0, x0) * (1.0 - wx) + p(y0, x1) * wx;
                    let bottom = p(y1, x0) * (1.0 - wx) + p(y1, x1) * wx;
                    data[(y * width + x) * 3 + c] =
                        (top * (1.0 - wy) + bottom * wy).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    RgbImage {
        width,
        height,
        data,
    }
}

/// 正規化の値（モデルのマニフェストから）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normalization {
    pub scale: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// RGB（HWC）→ 正規化した NCHW の f32（長さ 3*H*W）
pub fn to_tensor(image: &RgbImage, norm: &Normalization) -> Vec<f32> {
    let plane = image.width * image.height;
    let mut out = vec![0f32; plane * 3];
    for (i, px) in image.data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * plane + i] = (f32::from(px[c]) / norm.scale - norm.mean[c]) / norm.std[c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const JPEG: &[u8] = include_bytes!("../../tests/fixtures/pattern_320x240.jpg");
    const EXPECTED: &str = include_str!("../../tests/fixtures/pattern_expected.json");

    fn imagenet() -> Normalization {
        Normalization {
            scale: 255.0,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    #[test]
    fn decodes_and_halves_like_opencv() {
        let image = decode_jpeg(JPEG).unwrap();
        assert_eq!((image.width, image.height), (320, 240));
        let small = resize(&image, 160, 120);
        let expected: serde_json::Value = serde_json::from_str(EXPECTED).unwrap();
        for (key, rgb) in expected["rgb_160x120_samples"].as_object().unwrap() {
            let (y, x) = key.split_once(',').unwrap();
            let (y, x): (usize, usize) = (y.parse().unwrap(), x.parse().unwrap());
            for c in 0..3 {
                let got = i32::from(small.data[(y * 160 + x) * 3 + c]);
                let want = rgb[c].as_i64().unwrap() as i32;
                // JPEG の復号の丸めは OpenCV（libjpeg-turbo）と1〜2違うことがある
                assert!(
                    (got - want).abs() <= 3,
                    "pixel {key} channel {c}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn tensor_matches_the_pc_preprocessing() {
        let small = resize(&decode_jpeg(JPEG).unwrap(), 160, 120);
        let tensor = to_tensor(&small, &imagenet());
        assert_eq!(tensor.len(), 3 * 120 * 160);
        let expected: serde_json::Value = serde_json::from_str(EXPECTED).unwrap();
        let mean = tensor.iter().map(|&v| f64::from(v)).sum::<f64>() / tensor.len() as f64;
        assert!((mean - expected["tensor_mean"].as_f64().unwrap()).abs() < 0.01);
        for (index, want) in expected["tensor_samples"].as_object().unwrap() {
            let got = f64::from(tensor[index.parse::<usize>().unwrap()]);
            assert!(
                (got - want.as_f64().unwrap()).abs() < 0.06,
                "tensor[{index}] {got} vs {want}"
            );
        }
    }

    #[test]
    fn bilinear_resize_keeps_flat_images_flat() {
        let src = RgbImage {
            width: 30,
            height: 20,
            data: vec![77; 30 * 20 * 3],
        };
        let out = resize(&src, 16, 12);
        assert!(out.data.iter().all(|&v| v == 77));
        assert!(decode_jpeg(b"not a jpeg").is_err());
    }
}
