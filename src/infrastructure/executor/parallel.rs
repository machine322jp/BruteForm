// 並列実行管理

use anyhow::{Result, Context};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use crossbeam_channel::{Sender, Receiver, unbounded};

/// 並列タスクの実行結果
pub type TaskResult<T> = Result<T>;

/// 並列実行設定
#[derive(Clone, Debug)]
pub struct ParallelConfig {
    /// ワーカースレッド数
    pub num_workers: usize,
    /// バッチサイズ
    pub batch_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            batch_size: 100,
        }
    }
}

impl ParallelConfig {
    pub fn new(num_workers: usize) -> Self {
        Self {
            num_workers,
            batch_size: 100,
        }
    }
    
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

/// 並列実行エグゼキューター
pub struct ParallelExecutor {
    config: ParallelConfig,
    abort_flag: Arc<AtomicBool>,
}

impl ParallelExecutor {
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            abort_flag: Arc::new(AtomicBool::new(false)),
        }
    }
    
    /// 中断フラグを取得
    pub fn abort_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.abort_flag)
    }
    
    /// 実行を中断
    pub fn abort(&self) {
        self.abort_flag.store(true, Ordering::Relaxed);
    }
    
    /// 中断されたかチェック
    pub fn is_aborted(&self) -> bool {
        self.abort_flag.load(Ordering::Relaxed)
    }
    
    /// ワーカー数を取得
    pub fn num_workers(&self) -> usize {
        self.config.num_workers
    }
    
    /// バッチサイズを取得
    pub fn batch_size(&self) -> usize {
        self.config.batch_size
    }
    
    /// 設定を取得
    pub fn config(&self) -> &ParallelConfig {
        &self.config
    }
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        Self::new(ParallelConfig::default())
    }
}

/// ワーカープール（簡易実装）
pub struct WorkerPool<T, R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    task_tx: Sender<T>,
    result_rx: Receiver<R>,
    num_workers: usize,
}

impl<T, R> WorkerPool<T, R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    /// 新しいワーカープールを作成
    pub fn new<F>(num_workers: usize, worker_fn: F) -> Self
    where
        F: Fn(T) -> R + Send + Sync + Clone + 'static,
    {
        let (task_tx, task_rx) = unbounded::<T>();
        let (result_tx, result_rx) = unbounded::<R>();
        
        for _ in 0..num_workers {
            let task_rx = task_rx.clone();
            let result_tx = result_tx.clone();
            let worker_fn = worker_fn.clone();
            
            std::thread::spawn(move || {
                while let Ok(task) = task_rx.recv() {
                    let result = worker_fn(task);
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            });
        }
        
        Self {
            task_tx,
            result_rx,
            num_workers,
        }
    }
    
    /// タスクを送信
    pub fn send_task(&self, task: T) -> Result<()> {
        self.task_tx.send(task)
            .map_err(|e| anyhow::anyhow!("タスクの送信に失敗しました: {}", e))
    }
    
    /// 結果を受信（ブロッキング）
    pub fn recv_result(&self) -> Result<R> {
        self.result_rx.recv()
            .map_err(|e| anyhow::anyhow!("結果の受信に失敗しました: {}", e))
    }
    
    /// 結果を受信（ノンブロッキング）
    pub fn try_recv_result(&self) -> Option<R> {
        self.result_rx.try_recv().ok()
    }
    
    /// ワーカー数を取得
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.num_workers > 0);
        assert_eq!(config.batch_size, 100);
    }
    
    #[test]
    fn parallel_config_with_batch_size() {
        let config = ParallelConfig::new(4).with_batch_size(200);
        assert_eq!(config.num_workers, 4);
        assert_eq!(config.batch_size, 200);
    }
    
    #[test]
    fn executor_can_abort() {
        let executor = ParallelExecutor::default();
        assert!(!executor.is_aborted());
        
        executor.abort();
        assert!(executor.is_aborted());
    }
    
    #[test]
    fn worker_pool_processes_tasks() {
        let pool = WorkerPool::new(2, |x: i32| x * 2);
        
        pool.send_task(5).unwrap();
        pool.send_task(10).unwrap();
        
        let result1 = pool.recv_result().unwrap();
        let result2 = pool.recv_result().unwrap();
        
        let mut results = vec![result1, result2];
        results.sort();
        assert_eq!(results, vec![10, 20]);
    }
    
    #[test]
    fn worker_pool_multiple_workers() {
        let pool = WorkerPool::new(4, |x: i32| {
            std::thread::sleep(Duration::from_millis(10));
            x + 1
        });
        
        for i in 0..10 {
            pool.send_task(i).unwrap();
        }
        
        let mut results = Vec::new();
        for _ in 0..10 {
            results.push(pool.recv_result().unwrap());
        }
        
        results.sort();
        assert_eq!(results, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }
}
