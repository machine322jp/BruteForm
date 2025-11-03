// ドメイン層 - 制約・彩色ロジック

pub mod coloring;

pub use coloring::{build_abstract_info, enumerate_colorings_fast, AbstractInfo};
