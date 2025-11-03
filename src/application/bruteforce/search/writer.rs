// ファイル書き出しスレッド

use anyhow::{Context, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::application::bruteforce::event::SearchEvent;
use crate::profiling::TimeDelta;

/// 書き込みチャネルとスレッドハンドルを返す
pub fn spawn_writer_thread(
    outfile: PathBuf,
    event_tx: Sender<SearchEvent>,
) -> (Sender<Vec<String>>, JoinHandle<Result<()>>) {
    let (wtx, wrx) = unbounded::<Vec<String>>();
    
    let handle = thread::spawn(move || {
        writer_thread_main(wrx, outfile, event_tx)
    });
    
    (wtx, handle)
}

/// ライタースレッドのメイン処理
fn writer_thread_main(
    wrx: Receiver<Vec<String>>,
    outfile: PathBuf,
    tx: Sender<SearchEvent>,
) -> Result<()> {
    let mut io_time = Duration::ZERO;
    let file = File::create(&outfile)
        .with_context(|| format!("出力を作成できません: {}", outfile.display()))?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
    
    while let Ok(batch) = wrx.recv() {
        let t0 = Instant::now();
        for line in batch {
            writer.write_all(line.as_bytes())?;
            writer.write_all(b"\n")?;
        }
        io_time += t0.elapsed();
    }
    
    writer.flush()?;
    
    if io_time != Duration::ZERO {
        let mut td = TimeDelta::default();
        td.io_write_total = io_time;
        let _ = tx.send(SearchEvent::ProfileDelta(td));
    }
    
    Ok(())
}
