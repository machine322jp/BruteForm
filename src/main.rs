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
            bruteform::app::ui::helpers::install_japanese_fonts(&cc.egui_ctx);
            Box::new(App::default())
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI起動に失敗: {e}"))
}
