// ぷよぷよ連鎖形総当たり - メインエントリポイント

use anyhow::Result;

// ライブラリから必要な型をインポート
use bruteform::app::App;

fn main() -> Result<()> {
    // eframe の起動オプション
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(egui::vec2(980.0, 760.0)),
        ..Default::default()
    };

    // GUI起動
    eframe::run_native(
        "ぷよぷよ 連鎖形 総当たり（6×14）— Rust GUI",
        options,
        Box::new(|cc| {
            // 日本語フォントのインストール
            install_japanese_fonts(&cc.egui_ctx);
            Box::new(App::default())
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI起動に失敗: {e}"))
}

/// 日本語フォントのインストール
fn install_japanese_fonts(ctx: &egui::Context) {
    use egui::{FontData, FontDefinitions, FontFamily};

    let mut fonts = FontDefinitions::default();

    // Windows フォント候補
    let windir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".to_string());
    let fontdir = std::path::Path::new(&windir).join("Fonts");
    let candidates = [
        "meiryo.ttc",
        "meiryob.ttc",
        "YuGothR.ttc",
        "YuGothM.ttc",
        "YuGothB.ttc",
        "YuGothUI.ttc",
        "YuGothU.ttc",
        "msgothic.ttc",
        "msmincho.ttc",
    ];

    let mut loaded = false;
    for name in candidates.iter() {
        let path = fontdir.join(name);
        if let Ok(bytes) = std::fs::read(&path) {
            let key = format!("jp-{}", name.to_lowercase());
            fonts.font_data.insert(key.clone(), FontData::from_owned(bytes));
            fonts
                .families
                .get_mut(&FontFamily::Proportional)
                .unwrap()
                .insert(0, key.clone());
            fonts
                .families
                .get_mut(&FontFamily::Monospace)
                .unwrap()
                .insert(0, key.clone());
            loaded = true;
            break;
        }
    }

    if loaded {
        ctx.set_fonts(fonts);
    } else {
        eprintln!("日本語フォントを見つけられませんでした。C:\\Windows\\Fonts を確認してください。");
    }
}
