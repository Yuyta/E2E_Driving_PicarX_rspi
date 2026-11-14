//! 前へ進む指令なのに、画像が変わらない（引っかかった、押し付けられた）ことの検出。
//!
//! 画像を 40x30 の明るさに縮め、0.5 秒前の画像との差の平均（0〜255）を見る。差が小さい状態が
//! 続いたら引っかかったとみなす。布や暗い物に押し付けられると、反射が戻らず超音波では
//! 「遠い」と読めるので、Pi の反射ループでは止まらない。
//!
//! しきい値は収集したデータで決めた（2026-11-14、8セッション）。0.5 秒の差は、普通に前進して
//! いる時は中央値 20、引っかかった時は中央値 1.9。差 2.5 未満が約 0.3 秒続いたら引っかかったと
//! すると、PC の動きの確認で印の付いた引っかかり 32 件のうち 21 件を見つけ、ほかに見つけた
//! 14 件も、画像を見ると暗い物に押し付けられていた（特徴点がなく PC の確認では印が付かなかった）。

use std::collections::VecDeque;

pub const THUMB_W: usize = 40;
pub const THUMB_H: usize = 30;

#[derive(Debug, Clone, Copy)]
pub struct StuckConfig {
    /// この時間前の画像と比べる
    pub span_s: f64,
    /// 差の平均（0〜255）がこれ未満なら変わっていない
    pub still_diff: f64,
    /// 変わっていない状態がこの時間続いたら引っかかった
    pub sustain_s: f64,
}

impl Default for StuckConfig {
    fn default() -> Self {
        Self {
            span_s: 0.5,
            still_diff: 2.5,
            sustain_s: 0.3,
        }
    }
}

/// RGB（HWC、u8）の画像を、40x30 の明るさ（RGB の平均）へ縮める
pub fn thumbnail(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    let mut out = vec![0f32; THUMB_W * THUMB_H];
    let bw = width / THUMB_W;
    let bh = height / THUMB_H;
    if bw == 0 || bh == 0 {
        return out;
    }
    let scale = 1.0 / (3.0 * (bw * bh) as f32);
    for ty in 0..THUMB_H {
        for tx in 0..THUMB_W {
            let mut sum = 0u32;
            for y in ty * bh..(ty + 1) * bh {
                let row = y * width;
                for x in tx * bw..(tx + 1) * bw {
                    let i = (row + x) * 3;
                    sum += u32::from(rgb[i]) + u32::from(rgb[i + 1]) + u32::from(rgb[i + 2]);
                }
            }
            out[ty * THUMB_W + tx] = sum as f32 * scale;
        }
    }
    out
}

pub fn mean_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return f64::INFINITY;
    }
    let total: f64 = a.iter().zip(b).map(|(x, y)| f64::from((x - y).abs())).sum();
    total / a.len() as f64
}

#[derive(Debug, Clone)]
pub struct StuckDetector {
    pub config: StuckConfig,
    frames: VecDeque<(f64, Vec<f32>)>,
    still_since: Option<f64>,
    pub last_diff: Option<f64>,
}

impl StuckDetector {
    pub fn new(config: StuckConfig) -> Self {
        Self {
            config,
            frames: VecDeque::new(),
            still_since: None,
            last_diff: None,
        }
    }

    /// 前進していない時、指令を変えた時に呼ぶ。比べる画像を捨てる
    pub fn reset(&mut self) {
        self.frames.clear();
        self.still_since = None;
        self.last_diff = None;
    }

    /// 前へ進む指令を続けている間の1フレーム。引っかかったと判断したら true
    pub fn update(&mut self, now: f64, thumb: Vec<f32>) -> bool {
        let c = self.config;
        // span_s より前の画像のうち、いちばん新しいものと比べる
        let mut reference = None;
        while let Some((t, _)) = self.frames.front() {
            if now - t < c.span_s {
                break;
            }
            reference = self.frames.pop_front();
        }
        if let Some(r) = reference {
            // 次の比較にも使えるよう戻す（span_s を少し超えた画像）
            self.frames.push_front(r);
        }
        let diff = self
            .frames
            .front()
            .filter(|(t, _)| now - t >= c.span_s)
            .map(|(_, old)| mean_abs_diff(old, &thumb));
        self.frames.push_back((now, thumb));
        self.last_diff = diff;

        match diff {
            Some(d) if d < c.still_diff => {
                let since = *self.still_since.get_or_insert(now);
                now - since >= c.sustain_s
            }
            _ => {
                self.still_since = None;
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(value: f32) -> Vec<f32> {
        vec![value; THUMB_W * THUMB_H]
    }

    #[test]
    fn thumbnail_averages_blocks_of_rgb() {
        let (w, h) = (80, 60);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                rgb[i] = if x < 40 { 30 } else { 90 };
                rgb[i + 1] = rgb[i];
                rgb[i + 2] = rgb[i];
            }
        }
        let t = thumbnail(&rgb, w, h);
        assert_eq!(t.len(), THUMB_W * THUMB_H);
        assert!((t[0] - 30.0).abs() < 1e-4);
        assert!((t[THUMB_W - 1] - 90.0).abs() < 1e-4);
    }

    #[test]
    fn unchanged_images_for_long_enough_mean_stuck() {
        let mut d = StuckDetector::new(StuckConfig::default());
        let mut fired = None;
        for i in 0..20 {
            let now = f64::from(i) / 15.0;
            if d.update(now, flat(100.0)) && fired.is_none() {
                fired = Some(now);
            }
        }
        // 0.5 秒たって比べ始め、さらに 0.3 秒続いた
        let at = fired.expect("should detect");
        assert!((0.75..0.9).contains(&at), "fired at {at}");
    }

    #[test]
    fn changing_images_are_not_stuck_and_reset_clears_history() {
        let mut d = StuckDetector::new(StuckConfig::default());
        for i in 0..30 {
            let now = f64::from(i) / 15.0;
            assert!(!d.update(now, flat(f64::from(i * 5) as f32)));
        }
        assert!(d.last_diff.unwrap() > 2.5);
        d.reset();
        assert!(!d.update(3.0, flat(0.0)));
        assert_eq!(d.last_diff, None);
    }
}
