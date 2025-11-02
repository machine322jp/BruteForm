// 結果の書き込み

use anyhow::Result;
use std::path::Path;
use std::fs::File;
use std::io::{BufWriter, Write};
use serde::Serialize;

use crate::domain::search::SearchResult;

/// 検索結果を書き込むためのtrait
pub trait ResultWriter: Send + Sync {
    /// 単一の結果を書き込む
    fn write_result(&mut self, result: &SearchResult) -> Result<()>;
    
    /// 複数の結果をバッチで書き込む
    fn write_batch(&mut self, results: &[SearchResult]) -> Result<()> {
        for result in results {
            self.write_result(result)?;
        }
        Ok(())
    }
    
    /// 書き込みを完了（フラッシュ）
    fn flush(&mut self) -> Result<()>;
    
    /// 書き込んだ結果数を取得
    fn count(&self) -> u64;
}

/// ファイルへの結果書き込み実装
pub struct FileResultWriter {
    writer: BufWriter<File>,
    count: u64,
    format: OutputFormat,
}

/// 出力フォーマット
#[derive(Clone, Copy, Debug)]
pub enum OutputFormat {
    /// JSON Lines形式（1行1結果）
    JsonLines,
    /// JSON配列形式
    JsonArray,
}

impl FileResultWriter {
    /// 新しいファイルライターを作成
    pub fn new(path: &Path, format: OutputFormat) -> Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        
        Ok(Self {
            writer,
            count: 0,
            format,
        })
    }
    
    /// JSON Lines形式で作成
    pub fn json_lines(path: &Path) -> Result<Self> {
        Self::new(path, OutputFormat::JsonLines)
    }
    
    /// JSON配列形式で作成
    pub fn json_array(path: &Path) -> Result<Self> {
        let mut writer = Self::new(path, OutputFormat::JsonArray)?;
        writer.writer.write_all(b"[\n")?;
        Ok(writer)
    }
}

impl ResultWriter for FileResultWriter {
    fn write_result(&mut self, result: &SearchResult) -> Result<()> {
        match self.format {
            OutputFormat::JsonLines => {
                let json = serde_json::to_string(result)?;
                writeln!(self.writer, "{}", json)?;
            }
            OutputFormat::JsonArray => {
                if self.count > 0 {
                    writeln!(self.writer, ",")?;
                }
                let json = serde_json::to_string(result)?;
                write!(self.writer, "  {}", json)?;
            }
        }
        self.count += 1;
        Ok(())
    }
    
    fn flush(&mut self) -> Result<()> {
        if matches!(self.format, OutputFormat::JsonArray) {
            writeln!(self.writer, "\n]")?;
        }
        self.writer.flush()?;
        Ok(())
    }
    
    fn count(&self) -> u64 {
        self.count
    }
}

impl Drop for FileResultWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// メモリ内結果書き込み実装（テスト用）
pub struct MemoryResultWriter {
    results: Vec<SearchResult>,
}

impl MemoryResultWriter {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }
    
    pub fn results(&self) -> &[SearchResult] {
        &self.results
    }
}

impl Default for MemoryResultWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ResultWriter for MemoryResultWriter {
    fn write_result(&mut self, result: &SearchResult) -> Result<()> {
        self.results.push(result.clone());
        Ok(())
    }
    
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
    
    fn count(&self) -> u64 {
        self.results.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::board::BoardBits;
    use crate::domain::search::ChainCount;
    
    fn test_result() -> SearchResult {
        SearchResult {
            board: BoardBits::new(),
            chain_count: ChainCount::new(5).unwrap(),
            hash: 12345,
            is_mirror: false,
        }
    }
    
    #[test]
    fn memory_writer_stores_results() {
        let mut writer = MemoryResultWriter::new();
        let result = test_result();
        
        writer.write_result(&result).unwrap();
        writer.write_result(&result).unwrap();
        
        assert_eq!(writer.count(), 2);
        assert_eq!(writer.results().len(), 2);
    }
    
    #[test]
    fn memory_writer_batch_write() {
        let mut writer = MemoryResultWriter::new();
        let results = vec![test_result(), test_result(), test_result()];
        
        writer.write_batch(&results).unwrap();
        
        assert_eq!(writer.count(), 3);
    }
    
    #[test]
    fn memory_writer_flush_succeeds() {
        let mut writer = MemoryResultWriter::new();
        assert!(writer.flush().is_ok());
    }
}
