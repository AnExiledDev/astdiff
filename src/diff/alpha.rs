//! Token-level alpha-equivalence for minified JavaScript.
//!
//! A re-minified bundle renames every identifier while leaving the code itself
//! untouched. Text-level comparison sees such a pair as fully changed; this
//! module sees it as identical by comparing the token stream with bound
//! identifiers replaced by their first-occurrence index (`%0`, `%1`, ...).
//!
//! What is normalized and what is kept literal:
//! - `identifier`, `statement_identifier` (labels) and
//!   `private_property_identifier` (`#x` class fields, which are class-scoped)
//!   are indexed: minifiers rename all three freely, and a consistent rename
//!   maps to the same index sequence on both sides.
//! - Public property names (`property_identifier`, shorthand object keys) stay
//!   literal: minifiers do not rename property accesses, and `a.push` vs
//!   `a.shift` must never compare equal.
//! - String fragments and escape sequences become [`NormTok::Str`], compared
//!   by content normally and ignored by the masked comparison, so "only
//!   string text changed" is detectable at the token level.
//! - Numbers, regexes, keywords and punctuation stay literal.
//!
//! Tokens come from a fresh tree-sitter parse of the declaration snippet.
//! The snippet is often a fragment (a lone `var` declarator, say); tree-sitter
//! still lexes valid tokens inside its error recovery, and both sides of a
//! pair mis-parse the same way, so the comparison stays symmetric.

use std::collections::HashMap;
use tree_sitter::{Node, Parser};

/// One normalized token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormTok {
    /// A renameable identifier, as its first-occurrence index within the snippet.
    Var(u32),
    /// Anything compared literally: keywords, punctuation, numbers, property names.
    Lit(Box<str>),
    /// String-literal content (string/template fragments and escapes).
    Str(Box<str>),
}

/// A declaration snippet reduced to normalized tokens, with the source line of
/// each token retained so the display diff can align original lines.
pub struct AlphaTokens {
    toks: Vec<NormTok>,
    line_of: Vec<u32>,
    line_count: usize,
}

/// A reusable tokenizer holding one tree-sitter parser.
pub struct AlphaTokenizer {
    parser: Parser,
}

impl AlphaTokenizer {
    pub fn new() -> Self {
        let mut parser = Parser::new();
        parser
            .set_language(tree_sitter_javascript::language())
            .expect("tree-sitter-javascript language must load");

        Self { parser }
    }

    pub fn tokenize(&mut self, src: &str) -> AlphaTokens {
        let mut toks = Vec::new();
        let mut line_of = Vec::new();
        let mut var_ids: HashMap<String, u32> = HashMap::new();

        if let Some(tree) = self.parser.parse(src, None) {
            collect_leaves(tree.root_node(), src, &mut toks, &mut line_of, &mut var_ids);
        }

        AlphaTokens {
            toks,
            line_of,
            line_count: src.lines().count(),
        }
    }
}

/// Whether two snippets are alpha-equivalent: same token stream modulo
/// consistent identifier renaming. Line boundaries are ignored, so a rename
/// that re-wraps lines still compares equal.
pub fn alpha_equal(a: &AlphaTokens, b: &AlphaTokens) -> bool {
    a.toks == b.toks
}

/// Alpha-equivalence with string content masked: true when the only
/// difference beyond renames is the text inside string literals.
pub fn alpha_equal_masked(a: &AlphaTokens, b: &AlphaTokens) -> bool {
    if a.toks.len() != b.toks.len() {
        return false;
    }

    a.toks.iter().zip(&b.toks).all(|(x, y)| match (x, y) {
        (NormTok::Str(_), NormTok::Str(_)) => true,
        _ => x == y,
    })
}

impl AlphaTokens {
    /// The normalized text of each source line (same line count as the input),
    /// for aligning original lines in the display diff. Distinct identifiers
    /// keep distinct indices, so lines stay distinguishable after
    /// normalization instead of collapsing into one degenerate blank form.
    pub fn norm_lines(&self) -> Vec<String> {
        let mut lines = vec![String::new(); self.line_count];

        for (tok, &line) in self.toks.iter().zip(&self.line_of) {
            let Some(slot) = lines.get_mut(line as usize) else {
                continue;
            };

            if !slot.is_empty() {
                slot.push(' ');
            }
            match tok {
                NormTok::Var(n) => {
                    slot.push('%');
                    slot.push_str(&n.to_string());
                }
                NormTok::Lit(s) | NormTok::Str(s) => slot.push_str(s),
            }
        }

        lines
    }
}

fn collect_leaves(
    node: Node,
    src: &str,
    toks: &mut Vec<NormTok>,
    line_of: &mut Vec<u32>,
    var_ids: &mut HashMap<String, u32>,
) {
    if node.child_count() == 0 {
        push_leaf(node, src, toks, line_of, var_ids);
        return;
    }

    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            collect_leaves(cursor.node(), src, toks, line_of, var_ids);

            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
}

fn push_leaf(
    node: Node,
    src: &str,
    toks: &mut Vec<NormTok>,
    line_of: &mut Vec<u32>,
    var_ids: &mut HashMap<String, u32>,
) {
    let kind = node.kind();

    if matches!(kind, "comment" | "hash_bang_line") {
        return;
    }

    let text = &src[node.byte_range()];

    let tok = match kind {
        "identifier" | "statement_identifier" | "private_property_identifier" => {
            let next_id = var_ids.len() as u32;
            NormTok::Var(*var_ids.entry(text.to_string()).or_insert(next_id))
        }
        "string_fragment" | "escape_sequence" => NormTok::Str(text.into()),
        _ => NormTok::Lit(text.into()),
    };

    toks.push(tok);
    line_of.push(node.start_position().row as u32);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> AlphaTokens {
        AlphaTokenizer::new().tokenize(src)
    }

    #[test]
    fn pure_rename_is_equal() {
        let a = toks("function aB(x, y) { return x + y * 2; }");
        let b = toks("function cD(u, v) { return u + v * 2; }");
        assert!(alpha_equal(&a, &b));
    }

    #[test]
    fn inconsistent_rename_is_not_equal() {
        // x maps to u in one place and v in the other: not a rename.
        let a = toks("function f(x, y) { return x + x; }");
        let b = toks("function g(u, v) { return u + v; }");
        assert!(!alpha_equal(&a, &b));
    }

    #[test]
    fn property_name_change_is_not_equal() {
        let a = toks("var r = q.push(1);");
        let b = toks("var s = w.shift(1);");
        assert!(!alpha_equal(&a, &b));
    }

    #[test]
    fn object_key_change_is_not_equal() {
        let a = toks("var o = { retries: n };");
        let b = toks("var p = { timeout: m };");
        assert!(!alpha_equal(&a, &b));
    }

    #[test]
    fn template_substitution_rename_is_equal() {
        let a = toks("function f(p) { return `got ${p} done`; }");
        let b = toks("function g(q) { return `got ${q} done`; }");
        assert!(alpha_equal(&a, &b));
    }

    #[test]
    fn template_text_change_is_not_equal_but_masked_equal() {
        let a = toks("function f(p) { return `got ${p} done`; }");
        let b = toks("function g(q) { return `took ${q} done`; }");
        assert!(!alpha_equal(&a, &b));
        assert!(alpha_equal_masked(&a, &b));
    }

    #[test]
    fn string_change_is_masked_equal_only() {
        let a = toks("function f() { throw new Error(\"old message\"); }");
        let b = toks("function g() { throw new Error(\"new message\"); }");
        assert!(!alpha_equal(&a, &b));
        assert!(alpha_equal_masked(&a, &b));
    }

    #[test]
    fn structural_change_is_not_masked_equal() {
        let a = toks("function f(x) { return x + 1; }");
        let b = toks("function g(y) { return y * 1; }");
        assert!(!alpha_equal(&a, &b));
        assert!(!alpha_equal_masked(&a, &b));
    }

    #[test]
    fn rewrapped_lines_still_equal() {
        let a = toks("var r = fn(aLongName1,\n  aLongName2);");
        let b = toks("var r = fn(\n  b1,\n  b2\n);");
        assert!(alpha_equal(&a, &b));
    }

    #[test]
    fn label_rename_is_equal() {
        let a = toks("e: { if (x) break e; run(x); }");
        let b = toks("f: { if (y) break f; run(y); }");
        assert!(alpha_equal(&a, &b));
    }

    #[test]
    fn private_field_rename_is_equal() {
        let a = toks("class A { #o = 1; get() { return this.#o; } }");
        let b = toks("class B { #i = 1; get() { return this.#i; } }");
        assert!(alpha_equal(&a, &b));
    }

    #[test]
    fn number_change_is_not_equal() {
        let a = toks("var t = wait(1000);");
        let b = toks("var u = wait(2000);");
        assert!(!alpha_equal(&a, &b));
        assert!(!alpha_equal_masked(&a, &b));
    }

    #[test]
    fn norm_lines_track_source_lines() {
        let t = toks("var ab = 1;\nvar cd = ab + 2;");
        let lines = t.norm_lines();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "var %0 = 1 ;");
        assert_eq!(lines[1], "var %1 = %0 + 2 ;");
    }
}
