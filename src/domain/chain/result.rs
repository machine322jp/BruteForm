// 連鎖結果の定義

/// 連鎖結果
#[derive(Clone, Debug, PartialEq)]
pub struct ChainResult {
    /// 連鎖数
    pub chain_count: u32,
    /// 消したぷよの総数
    pub total_erased: usize,
    /// 各ステップの詳細
    pub sequence: ChainSequence,
}

impl ChainResult {
    pub fn new(chain_count: u32) -> Self {
        Self {
            chain_count,
            total_erased: 0,
            sequence: ChainSequence::new(),
        }
    }
    
    pub fn with_erased(chain_count: u32, total_erased: usize) -> Self {
        Self {
            chain_count,
            total_erased,
            sequence: ChainSequence::new(),
        }
    }
}

/// 連鎖のシーケンス（各ステップの記録）
#[derive(Clone, Debug, PartialEq)]
pub struct ChainSequence {
    pub steps: Vec<ChainStep>,
}

impl ChainSequence {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }
    
    pub fn add_step(&mut self, step: ChainStep) {
        self.steps.push(step);
    }
}

impl Default for ChainSequence {
    fn default() -> Self {
        Self::new()
    }
}

/// 連鎖の1ステップ
#[derive(Clone, Debug, PartialEq)]
pub struct ChainStep {
    /// このステップで消えたぷよの数
    pub erased_count: usize,
    /// 消えた色
    pub erased_color: Option<u8>,
}

impl ChainStep {
    pub fn new(erased_count: usize, erased_color: Option<u8>) -> Self {
        Self {
            erased_count,
            erased_color,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn chain_result_new() {
        let result = ChainResult::new(5);
        assert_eq!(result.chain_count, 5);
        assert_eq!(result.total_erased, 0);
    }
    
    #[test]
    fn chain_sequence_add_step() {
        let mut seq = ChainSequence::new();
        seq.add_step(ChainStep::new(4, Some(0)));
        seq.add_step(ChainStep::new(5, Some(1)));
        
        assert_eq!(seq.steps.len(), 2);
        assert_eq!(seq.steps[0].erased_count, 4);
        assert_eq!(seq.steps[1].erased_color, Some(1));
    }
}
