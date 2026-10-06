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
//!
//! Alpha-equivalence alone picks its own identifier bijection, which is only
//! right for names the snippet binds itself. A name it never binds is a free
//! reference, usually to another top-level declaration, and those already
//! have a global pairing; [`free_renames`] exposes them so the caller can hold
//! them to it.

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
    /// Every indexed identifier in index order: `NormTok::Var(n)` is `idents[n]`.
    idents: Vec<Ident>,
}

/// The identifier behind one `NormTok::Var` index.
struct Ident {
    name: Box<str>,
    /// Declared somewhere in the snippet itself (a parameter, a local, a catch
    /// binding, a function or class name), or not a plain identifier at all
    /// (a label, a `#field`). Either way it never names an outside declaration
    /// and keeps plain alpha-equivalence. A name bound anywhere in the snippet
    /// counts as bound everywhere in it: missing a shadowed outer reference only
    /// falls back to the old behaviour, while a parameter mistaken for a free
    /// reference would flag a pure rename.
    is_local: bool,
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
        let mut out = AlphaTokens {
            toks: Vec::new(),
            line_of: Vec::new(),
            line_count: src.lines().count(),
            idents: Vec::new(),
        };
        let mut var_ids: HashMap<String, u32> = HashMap::new();

        if let Some(tree) = self.parser.parse(src, None) {
            collect_leaves(tree.root_node(), src, &mut out, &mut var_ids, false);
        }

        out
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

/// The free references two snippets trade under their alpha bijection, as
/// `(index, name in a, name in b)`: index `n` in `a` is read as index `n` in
/// `b`. An index local to either side is left out. Only meaningful when
/// [`alpha_equal_masked`] holds, since only then do the indices line up.
pub fn free_renames<'t>(
    a: &'t AlphaTokens,
    b: &'t AlphaTokens,
) -> impl Iterator<Item = (u32, &'t str, &'t str)> + 't {
    a.idents
        .iter()
        .zip(&b.idents)
        .enumerate()
        .filter(|(_, (x, y))| !x.is_local && !y.is_local)
        .map(|(n, (x, y))| (n as u32, &*x.name, &*y.name))
}

impl AlphaTokens {
    /// The normalized text of each source line (same line count as the input),
    /// for aligning original lines in the display diff. Distinct identifiers
    /// keep distinct indices, so lines stay distinguishable after
    /// normalization instead of collapsing into one degenerate blank form.
    pub fn norm_lines(&self) -> Vec<String> {
        self.render_lines(&[], ' ')
    }

    /// [`AlphaTokens::norm_lines`], with every index in `flagged` suffixed by
    /// `tag`. Tagging the two sides differently keeps any line that uses a
    /// flagged index from aligning with its counterpart, so the display diff
    /// shows exactly those lines even when the token streams are equal.
    pub fn norm_lines_flagging(&self, flagged: &[u32], tag: char) -> Vec<String> {
        self.render_lines(flagged, tag)
    }

    /// Whether the snippet only binds one name to another (`var a = b;`, or a
    /// later declarator `a = b,`). Such a declaration carries nothing of its
    /// own for a matcher to pair it by. Zero-width tokens are skipped: they
    /// are what the parser inserts to recover from a fragment, not source.
    pub fn is_bare_alias(&self) -> bool {
        let source_toks = self.toks.iter().filter(|tok| !self.is_zero_width(tok));
        let toks: Vec<&NormTok> = source_toks.collect();
        let toks = match toks.split_first() {
            Some((first, rest)) if is_lit(first, &["var", "let", "const"]) => rest,
            _ => &toks[..],
        };
        let toks = match toks.split_last() {
            Some((last, rest)) if is_lit(last, &[",", ";"]) => rest,
            _ => toks,
        };

        matches!(toks, [NormTok::Var(_), eq, NormTok::Var(_)] if is_lit(eq, &["="]))
    }

    fn is_zero_width(&self, tok: &NormTok) -> bool {
        match tok {
            NormTok::Var(n) => self.idents[*n as usize].name.is_empty(),
            NormTok::Lit(text) | NormTok::Str(text) => text.is_empty(),
        }
    }

    fn render_lines(&self, flagged: &[u32], tag: char) -> Vec<String> {
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

                    if flagged.contains(n) {
                        slot.push(tag);
                    }
                }
                NormTok::Lit(s) | NormTok::Str(s) => slot.push_str(s),
            }
        }

        lines
    }
}

/// Whether `tok` is literal text equal to one of `texts`.
fn is_lit(tok: &NormTok, texts: &[&str]) -> bool {
    matches!(tok, NormTok::Lit(text) if texts.contains(&&**text))
}

/// `in_binding_slot` says whether `node` sits where its parent declares a name
/// (see [`is_binding_slot`]); it only matters once `node` is an identifier leaf.
fn collect_leaves(
    node: Node,
    src: &str,
    out: &mut AlphaTokens,
    var_ids: &mut HashMap<String, u32>,
    in_binding_slot: bool,
) {
    if node.child_count() == 0 {
        push_leaf(node, src, out, var_ids, in_binding_slot);
        return;
    }

    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            let child_binds = is_binding_slot(node.kind(), cursor.field_name());
            collect_leaves(cursor.node(), src, out, var_ids, child_binds);

            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
}

/// Whether a child in `field` of a `parent_kind` node declares a name: a
/// parameter, a declarator or function/class name, a catch or loop binding, or
/// a name inside a destructuring pattern. Errs toward "declares" (a plain
/// `for (x in y)` or a destructuring assignment counts too), because a name
/// wrongly taken as local only loses the free-reference check.
fn is_binding_slot(parent_kind: &str, field: Option<&str>) -> bool {
    match parent_kind {
        "formal_parameters" | "array_pattern" | "rest_pattern" => true,
        "variable_declarator"
        | "function_declaration"
        | "function_expression"
        | "function"
        | "generator_function_declaration"
        | "generator_function"
        | "class_declaration"
        | "class" => field == Some("name"),
        "arrow_function" | "catch_clause" => field == Some("parameter"),
        "for_in_statement" | "assignment_pattern" => field == Some("left"),
        "pair_pattern" => field == Some("value"),
        _ => false,
    }
}

fn push_leaf(
    node: Node,
    src: &str,
    out: &mut AlphaTokens,
    var_ids: &mut HashMap<String, u32>,
    in_binding_slot: bool,
) {
    let kind = node.kind();

    if matches!(kind, "comment" | "hash_bang_line") {
        return;
    }

    let text = &src[node.byte_range()];

    let tok = match kind {
        "identifier" | "statement_identifier" | "private_property_identifier" => {
            let is_local = in_binding_slot || kind != "identifier";
            NormTok::Var(index_ident(out, var_ids, text, is_local))
        }
        "string_fragment" | "escape_sequence" => NormTok::Str(text.into()),
        _ => NormTok::Lit(text.into()),
    };

    out.toks.push(tok);
    out.line_of.push(node.start_position().row as u32);
}

/// The index of identifier `text`, recording it on first sight and marking it
/// local once any occurrence is.
fn index_ident(
    out: &mut AlphaTokens,
    var_ids: &mut HashMap<String, u32>,
    text: &str,
    is_local: bool,
) -> u32 {
    let next_id = var_ids.len() as u32;
    let id = *var_ids.entry(text.to_string()).or_insert(next_id);

    if id == next_id {
        out.idents.push(Ident {
            name: text.into(),
            is_local,
        });
    } else if is_local {
        out.idents[id as usize].is_local = true;
    }

    id
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

    fn free_names(a: &AlphaTokens, b: &AlphaTokens) -> Vec<(String, String)> {
        free_renames(a, b)
            .map(|(_, x, y)| (x.to_string(), y.to_string()))
            .collect()
    }

    #[test]
    fn free_renames_pair_outside_references_only() {
        // `run` and `base` are declared elsewhere; `x`, `y` and the arrow's
        // `e` are bound by the snippet and keep plain alpha-equivalence.
        let a = toks("function f(x, { k: y }) { return run(base, x, y, (e) => e); }");
        let b = toks("function g(p, { k: q }) { return go(table, p, q, (s) => s); }");
        assert!(alpha_equal(&a, &b));
        assert_eq!(
            free_names(&a, &b),
            vec![
                ("run".to_string(), "go".to_string()),
                ("base".to_string(), "table".to_string()),
            ]
        );
    }

    #[test]
    fn a_name_bound_anywhere_in_the_snippet_is_never_free() {
        // `base` is a parameter here, even though the same text could name
        // a top-level declaration; it must not be held to a global pairing.
        let a = toks("var r = function (base) { return base.low; };");
        let b = toks("var s = function (other) { return other.low; };");
        assert!(free_names(&a, &b).is_empty());
    }

    #[test]
    fn a_declarator_binding_one_name_to_another_is_a_bare_alias() {
        assert!(toks("var alias = base;").is_bare_alias());
        assert!(toks("  alias = base,").is_bare_alias());
        assert!(toks("  alias = base;").is_bare_alias());
    }

    #[test]
    fn a_declarator_doing_anything_more_is_not_a_bare_alias() {
        assert!(!toks("var alias = base.size;").is_bare_alias());
        assert!(!toks("var alias = wrap(base);").is_bare_alias());
        assert!(!toks("var alias = 1;").is_bare_alias());
        assert!(!toks("var alias = base, other = next;").is_bare_alias());
    }

    #[test]
    fn flagged_lines_stop_aligning() {
        let t = toks("var ab = 1;\nvar cd = ab + 2;");
        let lines = t.norm_lines_flagging(&[0], '-');
        assert_eq!(lines[0], "var %0- = 1 ;");
        assert_eq!(lines[1], "var %1 = %0- + 2 ;");
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
