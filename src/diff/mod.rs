use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use anyhow::Result;
use tree_sitter::Node;
use serde::{Serialize, Deserialize};

pub mod fingerprint;
pub mod matching_report;
pub mod parallel_matching_v2;
pub mod profiling;

use fingerprint::*;

/// Number of MinHash lanes per declaration. These values feed Jaccard estimation and the
/// LSH banding gate, so changing this changes which declarations match.
const MINHASH_LANES: usize = 128;

/// Read once: this is checked per declaration during extraction, and env::var takes the
/// process env lock and allocates on every call.
fn debug_enabled() -> bool {
    static DEBUG_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DEBUG_ENABLED.get_or_init(|| std::env::var("ASTDIFF_DEBUG").is_ok())
}

/// Represents a structural diff between two JavaScript ASTs
pub struct StructuralDiff {
    mappings1: Option<HashMap<String, String>>,
    mappings2: Option<HashMap<String, String>>,
    use_fingerprints: bool,
    generate_report: bool,
    report_path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Declaration {
    pub name: String,
    pub kind: DeclarationKind,
    pub line: usize,
    pub end_line: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub node_kind: String,
    pub signature: String,
    /// Sorted and deduplicated. Both consumers (MinHash, intersection count) are
    /// order-independent, and sequential u64s intersect far faster than a hash set
    /// whose RandomState re-hashes values that are already uniformly distributed.
    pub structural_hashes: Vec<u64>,
    pub size: usize,
    pub minhash_signature: Vec<u64>,
    pub fingerprint: Option<FunctionFingerprint>,
}

// Serializable version (same fields now that Node is gone)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerializableDeclaration {
    pub name: String,
    pub kind: DeclarationKind,
    pub line: usize,
    pub end_line: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub signature: String,
    pub structural_hashes: HashSet<u64>,
    pub size: usize,
    pub minhash_signature: Vec<u64>,
    pub fingerprint: Option<FunctionFingerprint>,
}

impl From<&Declaration> for SerializableDeclaration {
    fn from(decl: &Declaration) -> Self {
        SerializableDeclaration {
            name: decl.name.clone(),
            kind: decl.kind.clone(),
            line: decl.line,
            end_line: decl.end_line,
            start_byte: decl.start_byte,
            end_byte: decl.end_byte,
            signature: decl.signature.clone(),
            // Converted here rather than changing the field type: the dump is a
            // bincode format on disk and existing dumps must stay loadable.
            structural_hashes: decl.structural_hashes.iter().copied().collect(),
            size: decl.size,
            minhash_signature: decl.minhash_signature.clone(),
            fingerprint: decl.fingerprint.clone(),
        }
    }
}


// Thread-safe declaration data for parallel processing
#[derive(Debug, Clone)]
pub struct DeclarationData {
    name: String,
    kind: DeclarationKind,
    line: usize,
    end_line: usize,
    signature: String,
    structural_hashes: Vec<u64>,
    size: usize,
    minhash_signature: Vec<u64>,
    fingerprint: Option<FunctionFingerprint>,
}

impl Declaration {
    fn to_data(&self) -> DeclarationData {
        DeclarationData {
            name: self.name.clone(),
            kind: self.kind.clone(),
            line: self.line,
            end_line: self.end_line,
            signature: self.signature.clone(),
            structural_hashes: self.structural_hashes.clone(),
            size: self.size,
            minhash_signature: self.minhash_signature.clone(),
            fingerprint: self.fingerprint.clone(),
        }
    }

    /// Same fields as `to_data`, but moved. On a large bundle the cloned copy of
    /// every structural hash set doubles peak memory for the whole matching phase,
    /// so any caller that owns its declarations should convert this way.
    fn into_data(self) -> DeclarationData {
        DeclarationData {
            name: self.name,
            kind: self.kind,
            line: self.line,
            end_line: self.end_line,
            signature: self.signature,
            structural_hashes: self.structural_hashes,
            size: self.size,
            minhash_signature: self.minhash_signature,
            fingerprint: self.fingerprint,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DeclarationKind {
    Function,
    Variable,
    Class,
    Import,
    Export,
}

impl std::fmt::Display for DeclarationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeclarationKind::Function => write!(f, "function"),
            DeclarationKind::Variable => write!(f, "variable"),
            DeclarationKind::Class => write!(f, "class"),
            DeclarationKind::Import => write!(f, "import"),
            DeclarationKind::Export => write!(f, "export"),
        }
    }
}

/// Apply cross-kind penalty to similarity score.
/// Function <-> Variable swaps are common in minified code (small penalty).
/// Other kind mismatches get a larger penalty.
fn apply_kind_penalty(similarity: f64, kind1: &DeclarationKind, kind2: &DeclarationKind) -> f64 {
    if kind1 == kind2 {
        return similarity;
    }
    let is_func_var_swap = matches!(
        (kind1, kind2),
        (DeclarationKind::Function, DeclarationKind::Variable)
        | (DeclarationKind::Variable, DeclarationKind::Function)
    );
    similarity * if is_func_var_swap { 0.9 } else { 0.7 }
}

/// Number of values present in both sorted, deduplicated slices.
///
/// A straight merge rather than galloping: the only caller has already rejected any pair
/// whose size ratio is below 0.3, so the two sides are never skewed enough for binary
/// search to beat two sequential scans.
fn sorted_intersection_count(a: &[u64], b: &[u64]) -> usize {
    let mut count = 0;
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }

    count
}

/// Classification of a matched declaration pair based on normalized diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffClassification {
    /// Empty normalized diff — pure rename or identical
    Unchanged,
    /// Only string literal values changed (code skeleton identical)
    StringOnly,
    /// Code logic changed (structural differences beyond strings)
    Structural,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffResult {
    pub identical: bool,
    pub similarity: f64,
    pub changes: Vec<Change>,
    pub matched_declarations: usize,
    pub total_declarations1: usize,
    pub total_declarations2: usize,
    /// Rename map: new_name → old_name (file2 → file1) for normalizing source2 references
    #[serde(skip)]
    pub rename_map: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Change {
    pub change_type: ChangeType,
    pub location1: Option<Location>,
    pub location2: Option<Location>,
    pub description: String,
    pub structural_path: String,
    /// Classification derived from normalized diff (None for Add/Delete)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification: Option<DiffClassification>,
    /// The display diff (original text, normalized comparison). Empty if Unchanged.
    #[serde(skip)]
    pub display_diff: String,
    /// Similarity score for matched pairs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Addition,
    Deletion,
    Modification,
    Reorder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub line: usize,
    pub column: usize,
    pub code_snippet: String,
    pub end_line: Option<usize>,  // Optional end line for line ranges
}

/// Extract source text by line range. O(1) with pre-built line vector.
/// Lines are 1-indexed (matching tree-sitter convention).
pub fn extract_source_range(lines: &[&str], start_line: usize, end_line: usize) -> String {
    if start_line == 0 || start_line > lines.len() {
        return String::new();
    }
    let start = start_line - 1; // Convert to 0-indexed
    let end = end_line.min(lines.len());
    if start >= end {
        return String::new();
    }
    lines[start..end].join("\n")
}

impl StructuralDiff {
    pub fn new() -> Self {
        Self {
            mappings1: None,
            mappings2: None,
            use_fingerprints: true,  // Default to true for better accuracy
            generate_report: false,
            report_path: None,
        }
    }
    
    pub fn extract_declarations_for_inspection<'a>(&self, root: Node<'a>, source: &str) -> Vec<Declaration> {
        self.extract_declarations(root, source)
    }
    
    
    pub fn set_use_fingerprints(&mut self, use_fingerprints: bool) {
        self.use_fingerprints = use_fingerprints;
    }
    
    pub fn set_generate_report(&mut self, generate_report: bool) {
        self.generate_report = generate_report;
    }
    
    pub fn set_report_path(&mut self, path: std::path::PathBuf) {
        self.report_path = Some(path.to_string_lossy().to_string());
        self.generate_report = true;  // Automatically enable report if path is set
    }

    /// Format string diff for display, highlighting important changes
    fn calculate_line_statistics(&self, result: &DiffResult, _source1: &str, _source2: &str) -> (usize, usize, usize) {
        let mut declarations_added = 0;
        let mut declarations_removed = 0;
        let mut declarations_modified = 0;

        for change in &result.changes {
            match change.change_type {
                ChangeType::Addition => declarations_added += 1,
                ChangeType::Deletion => declarations_removed += 1,
                ChangeType::Modification => {
                    match change.classification.as_ref() {
                        Some(DiffClassification::Structural) | Some(DiffClassification::StringOnly) => {
                            declarations_modified += 1;
                        }
                        _ => {}
                    }
                }
                ChangeType::Reorder => {}
            }
        }

        (declarations_added, declarations_removed, declarations_added + declarations_removed + declarations_modified)
    }
    
    
    pub fn set_mappings1(&mut self, mappings: HashMap<String, String>) {
        self.mappings1 = Some(mappings);
    }
    
    pub fn set_mappings2(&mut self, mappings: HashMap<String, String>) {
        self.mappings2 = Some(mappings);
    }
    
    
    /// Takes declarations rather than trees so the caller can drop each syntax tree
    /// as soon as its declarations are extracted; nothing below here reads a Node.
    pub fn compare(&self, source1: &str, source2: &str,
                  declarations1: Vec<Declaration>, declarations2: Vec<Declaration>,
                  dump: Option<&std::path::Path>,
                  file1_path: &std::path::Path, file2_path: &std::path::Path) -> Result<DiffResult> {
        use crate::dump::{AstDiffDump, DiffConfig};

        // Compare declarations
        let need_dump = dump.is_some();
        let declarations1_clone = if need_dump { Some(declarations1.clone()) } else { None };
        let declarations2_clone = if need_dump { Some(declarations2.clone()) } else { None };

        let result = self.compare_declarations(declarations1, declarations2, source1, source2)?;

        // Dump if requested
        if let Some(dump_path) = dump {
            eprintln!("Creating comprehensive dump at {}", dump_path.display());

            let decls1_clone = declarations1_clone.unwrap();
            let decls2_clone = declarations2_clone.unwrap();
            let decls1: Vec<SerializableDeclaration> = decls1_clone.iter().map(|d| d.into()).collect();
            let decls2: Vec<SerializableDeclaration> = decls2_clone.iter().map(|d| d.into()).collect();

            // Reconstruct match data from changes
            let matches: Vec<(usize, usize, f64)> = result.changes.iter()
                .filter(|c| matches!(c.change_type, ChangeType::Modification))
                .filter_map(|c| {
                    let loc1 = c.location1.as_ref()?;
                    let loc2 = c.location2.as_ref()?;
                    let idx1 = decls1_clone.iter().position(|d| d.line == loc1.line)?;
                    let idx2 = decls2_clone.iter().position(|d| d.line == loc2.line)?;
                    let sim = c.similarity_score.unwrap_or(0.0);
                    Some((idx1, idx2, sim))
                })
                .collect();

            let config = DiffConfig {
                use_fingerprints: self.use_fingerprints,
                parallel_matching: true,
                threshold: 0.5,
            };

            let dump = AstDiffDump::new(
                file1_path.to_path_buf(),
                file2_path.to_path_buf(),
                decls1,
                decls2,
                matches,
                result.clone(),
                config,
            )?;

            dump.save(dump_path)?;
        }

        Ok(result)
    }
    
    pub fn compare_declarations(&self, declarations1: Vec<Declaration>, declarations2: Vec<Declaration>,
                              source1: &str, source2: &str) -> Result<DiffResult> {
        use profiling::Timer;

        eprintln!("Extracted {} declarations from file1, {} from file2",
                 declarations1.len(), declarations2.len());

        let total_declarations1 = declarations1.len();
        let total_declarations2 = declarations2.len();

        // Match declarations — now returns rename map and pre-classified changes
        let (matches, changes, rename_map) = {
            let _timer = Timer::new("match_declarations_total");
            self.match_owned_declarations(declarations1, declarations2, source1, source2)
        };

        let matched_declarations = matches.len();

        let similarity = if total_declarations1 == 0 && total_declarations2 == 0 {
            1.0
        } else {
            matched_declarations as f64 / total_declarations1.max(total_declarations2) as f64
        };

        Ok(DiffResult {
            identical: changes.is_empty(),
            similarity,
            changes,
            matched_declarations,
            total_declarations1,
            total_declarations2,
            rename_map,
        })
    }

    pub fn extract_declarations<'a>(&self, root: Node<'a>, source: &str) -> Vec<Declaration> {
        use rayon::prelude::*;

        let mut declarations = Vec::new();
        self.extract_declarations_recursive(root, source, &mut declarations, true);

        // Each signature depends only on its own declaration's hash set, so this pass is
        // order-independent. It runs here rather than in create_declaration because the
        // recursive walk holds tree-sitter Nodes and cannot cross threads.
        declarations.par_iter_mut().for_each(|decl| {
            decl.minhash_signature = self.compute_minhash(&decl.structural_hashes, MINHASH_LANES);
        });

        declarations
    }

    fn create_declaration(&self, name: String, kind: DeclarationKind,
                         line: usize, end_line: usize, start_byte: usize, end_byte: usize,
                         node_kind: &str, signature: String, structural_hashes: Vec<u64>,
                         fingerprint: Option<FunctionFingerprint>) -> Declaration {
        let size = structural_hashes.len();

        // Left empty on purpose: extract_declarations fills every signature in one
        // parallel pass once the whole vector exists.
        let minhash_signature = Vec::new();

        Declaration {
            name,
            kind,
            line,
            end_line,
            start_byte,
            end_byte,
            node_kind: node_kind.to_string(),
            signature,
            structural_hashes,
            size,
            minhash_signature,
            fingerprint,
        }
    }

    fn extract_fingerprint(&self, node: Node, source: &str, kind: &DeclarationKind, name: &str) -> Option<FunctionFingerprint> {
        if !matches!(kind, DeclarationKind::Function | DeclarationKind::Variable) {
            return None;
        }
        let _timer = profiling::Timer::new("extract_fingerprint");
        let extractor = FingerprintExtractor::new(source);
        let fp = extractor.extract_function_fingerprint(node);

        if debug_enabled() && !fp.strings.is_empty() {
            eprintln!("Fingerprint for {} '{}': {} strings, {} constants, {} API calls",
                kind, name, fp.strings.len(), fp.constants.len(), fp.api_calls.len());
            for s in &fp.strings {
                eprintln!("  String: '{}' ({:?})", s.value, s.context);
            }
        }

        Some(fp)
    }
    
    fn extract_declarations_recursive<'a>(&self, node: Node<'a>, source: &str, declarations: &mut Vec<Declaration>, is_global: bool) {
        match node.kind() {
            "function_declaration" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    let name = &source[name_node.byte_range()];
                    let kind = DeclarationKind::Function;
                    let fp = self.extract_fingerprint(node, source, &kind, name);
                    let signature = self.get_function_signature(node, source);
                    let structural_hashes = self.collect_structural_hashes(node, source);
                    declarations.push(self.create_declaration(
                        name.to_string(), kind,
                        node.start_position().row + 1, node.end_position().row + 1,
                        node.start_byte(), node.end_byte(), node.kind(),
                        signature, structural_hashes, fp,
                    ));
                }
            }
            "variable_declaration" if is_global => {
                for child in node.children(&mut node.walk()) {
                    if child.kind() == "variable_declarator" {
                        if child.child_by_field_name("value").is_none() {
                            continue;
                        }
                        if let Some(name_node) = child.child_by_field_name("name") {
                            if name_node.kind() == "identifier" {
                                let name = &source[name_node.byte_range()];
                                let kind = DeclarationKind::Variable;
                                let fp = self.extract_fingerprint(child, source, &kind, name);
                                let signature = self.get_variable_signature(child, source);
                                let structural_hashes = if let Some(value_node) = child.child_by_field_name("value") {
                                    self.collect_structural_hashes(value_node, source)
                                } else {
                                    Vec::new()
                                };
                                declarations.push(self.create_declaration(
                                    name.to_string(), kind,
                                    child.start_position().row + 1, child.end_position().row + 1,
                                    child.start_byte(), child.end_byte(), child.kind(),
                                    signature, structural_hashes, fp,
                                ));
                            }
                        }
                    }
                }
            }
            "class_declaration" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    let name = &source[name_node.byte_range()];
                    let kind = DeclarationKind::Class;
                    let fp = self.extract_fingerprint(node, source, &kind, name);
                    let signature = self.get_class_signature(node, source);
                    let structural_hashes = self.collect_structural_hashes(node, source);
                    declarations.push(self.create_declaration(
                        name.to_string(), kind,
                        node.start_position().row + 1, node.end_position().row + 1,
                        node.start_byte(), node.end_byte(), node.kind(),
                        signature, structural_hashes, fp,
                    ));
                }
            }
            "import_statement" => {
                let kind = DeclarationKind::Import;
                let name = format!("import@{}", node.start_position().row);
                let fp = self.extract_fingerprint(node, source, &kind, &name);
                let signature = self.get_import_signature(node, source);
                let structural_hashes = self.collect_structural_hashes(node, source);
                declarations.push(self.create_declaration(
                    name, kind,
                    node.start_position().row + 1, node.end_position().row + 1,
                    node.start_byte(), node.end_byte(), node.kind(),
                    signature, structural_hashes, fp,
                ));
            }
            "export_statement" => {
                if let Some(decl) = node.child_by_field_name("declaration") {
                    self.extract_declarations_recursive(decl, source, declarations, is_global);
                } else {
                    let kind = DeclarationKind::Export;
                    let name = format!("export@{}", node.start_position().row);
                    let fp = self.extract_fingerprint(node, source, &kind, &name);
                    let signature = self.get_export_signature(node, source);
                    let structural_hashes = self.collect_structural_hashes(node, source);
                    declarations.push(self.create_declaration(
                        name, kind,
                        node.start_position().row + 1, node.end_position().row + 1,
                        node.start_byte(), node.end_byte(), node.kind(),
                        signature, structural_hashes, fp,
                    ));
                }
            }
            _ => {
                // Only look for global declarations at the top level
                if is_global && node == node.parent().map(|p| p.child(0)).flatten().unwrap_or(node) {
                    for child in node.children(&mut node.walk()) {
                        self.extract_declarations_recursive(child, source, declarations, 
                            child.kind() != "function_declaration" && 
                            child.kind() != "class_declaration");
                    }
                }
            }
        }
    }
    
    fn collect_structural_hashes(&self, node: Node, source: &str) -> Vec<u64> {
        let mut hashes = Vec::new();
        let mut scratch = Vec::new();
        self.collect_structural_hashes_recursive(node, source, &mut hashes, &mut scratch);

        // Dedup here is not an optimization: `size` and the Jaccard denominator are
        // both set cardinalities, so the vector has to carry each hash once.
        hashes.sort_unstable();
        hashes.dedup();

        // Structural hashes repeat heavily, so dedup frees most of the vector. Without this
        // every declaration would hold its pre-dedup capacity for the rest of the run.
        hashes.shrink_to_fit();

        hashes
    }

    /// Hashes `node`, inserts its hash plus every descendant's into `hashes`, and returns it.
    ///
    /// A node's hash depends only on its own subtree, so the old separate `compute_structural_hash`
    /// pass re-walked a subtree the collector was already walking; folding the two into one
    /// post-order pass visits each node exactly once for the same result.
    ///
    /// `scratch` is a shared stack of child hashes: each frame owns the slice from `base` onwards
    /// and truncates back to it before returning, so the whole traversal shares one allocation
    /// instead of allocating a `Vec` per internal node.
    fn collect_structural_hashes_recursive(&self, node: Node, source: &str, hashes: &mut Vec<u64>,
                                           scratch: &mut Vec<u64>) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let is_literal = self.is_literal(node);
        let is_identifier = node.kind() == "identifier";
        let base = scratch.len();

        // Literals and identifiers ignore their children when hashing, but the walk still descends
        // into them so that descendants (a template_string's substitutions, say) reach the set.
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                if !matches!(child.kind(), "comment") {
                    let child_hash = self.collect_structural_hashes_recursive(child, source, hashes, scratch);
                    let contributes = !is_literal && !is_identifier
                        && !matches!(child.kind(), ";" | "," | "(" | ")" | "{" | "}" | "[" | "]");
                    if contributes {
                        scratch.push(child_hash);
                    }
                }

                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }

        let mut hasher = DefaultHasher::new();

        // Hash node type
        node.kind().hash(&mut hasher);

        // For literals, include the value
        if is_literal {
            source[node.byte_range()].hash(&mut hasher);
        } else if is_identifier {
            // For identifiers, just use a placeholder
            "<ID>".hash(&mut hasher);
        } else {
            // Sort child hashes for order-independent nodes
            if self.is_order_independent(node) {
                scratch[base..].sort();
            }
            for hash in &scratch[base..] {
                hash.hash(&mut hasher);
            }
        }

        scratch.truncate(base);

        let hash = hasher.finish();
        hashes.push(hash);

        hash
    }

    fn get_function_signature(&self, node: Node, _source: &str) -> String {
        let params = if let Some(params_node) = node.child_by_field_name("parameters") {
            let param_count = params_node.children(&mut params_node.walk())
                .filter(|n| n.kind() == "identifier" || n.kind() == "formal_parameters")
                .count();
            format!("params:{}", param_count)
        } else {
            "params:0".to_string()
        };
        
        let body = if let Some(body_node) = node.child_by_field_name("body") {
            let statement_count = body_node.children(&mut body_node.walk())
                .filter(|n| !matches!(n.kind(), "{" | "}" | ";"))
                .count();
            format!("stmts:{}", statement_count)
        } else {
            "stmts:0".to_string()
        };
        
        format!("function({},{})", params, body)
    }
    
    fn get_variable_signature(&self, node: Node, source: &str) -> String {
        if let Some(init) = node.child_by_field_name("value") {
            match init.kind() {
                "number" => format!("var=number:{}", &source[init.byte_range()]),
                "string" => format!("var=string:len{}", source[init.byte_range()].len()),
                "true" | "false" => format!("var=bool:{}", init.kind()),
                "array" => format!("var=array:len{}", init.children(&mut init.walk()).count()),
                "object" => format!("var=object:props{}", init.children(&mut init.walk())
                    .filter(|n| n.kind() == "pair").count()),
                "arrow_function" | "function" => {
                    let param_count = if let Some(params) = init.child_by_field_name("parameters") {
                        params.children(&mut params.walk()).count()
                    } else if init.child_by_field_name("parameter").is_some() {
                        1
                    } else {
                        0
                    };
                    format!("var=function:params{}", param_count)
                }
                _ => format!("var={}", init.kind()),
            }
        } else {
            "var=undefined".to_string()
        }
    }
    
    fn get_class_signature(&self, node: Node, _source: &str) -> String {
        if let Some(body) = node.child_by_field_name("body") {
            let method_count = body.children(&mut body.walk())
                .filter(|n| n.kind() == "method_definition")
                .count();
            let field_count = body.children(&mut body.walk())
                .filter(|n| n.kind() == "field_definition")
                .count();
            format!("class(methods:{},fields:{})", method_count, field_count)
        } else {
            "class()".to_string()
        }
    }
    
    fn get_import_signature(&self, node: Node, source: &str) -> String {
        let source_path = node.children(&mut node.walk())
            .find(|n| n.kind() == "string")
            .map(|n| &source[n.byte_range()])
            .unwrap_or("");
        format!("import from {}", source_path)
    }
    
    fn get_export_signature(&self, node: Node, _source: &str) -> String {
        if node.child_by_field_name("declaration").is_some() {
            "export declaration".to_string()
        } else if let Some(clause) = node.child_by_field_name("clause") {
            let export_count = clause.children(&mut clause.walk())
                .filter(|n| n.kind() == "export_specifier")
                .count();
            format!("export {} items", export_count)
        } else {
            "export".to_string()
        }
    }
    
    fn compute_minhash(&self, hashes: &[u64], num_hashes: usize) -> Vec<u64> {
        use std::collections::hash_map::DefaultHasher;

        let mut signature = vec![u64::MAX; num_hashes];

        for &hash in hashes {
            // The seeded hash is `value` then `seed` written into a fresh DefaultHasher,
            // so the state after writing the value is a prefix shared by all num_hashes
            // seeds. Cloning it feeds the hasher exactly the same byte sequence per seed
            // while paying for the value write once instead of num_hashes times.
            let mut base = DefaultHasher::new();
            hash.hash(&mut base);

            for i in 0..num_hashes {
                let mut hasher = base.clone();
                i.hash(&mut hasher);
                signature[i] = signature[i].min(hasher.finish());
            }
        }

        signature
    }

    pub fn calculate_declaration_similarity(&self, decl1: &Declaration, decl2: &Declaration, _source1: &str, _source2: &str) -> f64 {
        // Delegate to the _data version via conversion
        let d1 = decl1.to_data();
        let d2 = decl2.to_data();
        self.calculate_declaration_similarity_data(&d1, &d2, "", "")
    }
    
    
    
    fn is_literal(&self, node: Node) -> bool {
        matches!(node.kind(), 
            "string" | "number" | "true" | "false" | "null" | "undefined" | "regex" | "template_string"
        )
    }
    
    fn is_order_independent(&self, node: Node) -> bool {
        matches!(node.kind(), 
            "object" | "object_pattern" | "named_imports" | "export_clause"
        )
    }
    
    
        
    pub fn print_summary(&self, result: &DiffResult, file1: &std::path::PathBuf, file2: &std::path::PathBuf,
                         source1: &str, source2: &str) {
        println!("--- {}", file1.display());
        println!("+++ {}", file2.display());
        println!("Structural similarity: {:.1}%", result.similarity * 100.0);
        println!("Matched declarations: {}/{} vs {}",
                 result.matched_declarations, result.total_declarations1, result.total_declarations2);

        // Calculate and print line statistics
        let (lines_added, lines_removed, total_diff) = self.calculate_line_statistics(result, source1, source2);
        println!("Diff size: {} declarations (+{} added, -{} removed)", total_diff, lines_added, lines_removed);

        // Group changes by type using classification
        let mut additions = Vec::new();
        let mut deletions = Vec::new();
        let mut structural_changes = Vec::new();
        let mut string_changes = Vec::new();

        for change in &result.changes {
            match change.change_type {
                ChangeType::Addition => additions.push(change),
                ChangeType::Deletion => deletions.push(change),
                ChangeType::Modification => {
                    match change.classification.as_ref() {
                        Some(DiffClassification::Structural) => structural_changes.push(change),
                        Some(DiffClassification::StringOnly) => string_changes.push(change),
                        _ => {} // Unchanged — not shown
                    }
                }
                ChangeType::Reorder => {}
            }
        }

        let total_unchanged = result.matched_declarations
            .saturating_sub(structural_changes.len())
            .saturating_sub(string_changes.len());

        println!("Changes: {} added, {} removed, {} structural, {} string-only ({} unchanged)",
            additions.len(), deletions.len(), structural_changes.len(),
            string_changes.len(), total_unchanged);
        println!();

        // Show deletions
        if !deletions.is_empty() {
            println!("=== Removed ===");
            for change in &deletions {
                println!("--- {}", change.description);
                if let Some(loc) = &change.location1 {
                    println!("    at line {}: {}", loc.line, loc.code_snippet);
                }
            }
            println!();
        }

        // Show additions
        if !additions.is_empty() {
            println!("=== Added ===");
            for change in &additions {
                println!("+++ {}", change.description);
                if let Some(loc) = &change.location2 {
                    println!("    at line {}: {}", loc.line, loc.code_snippet);
                }
            }
            println!();
        }

        // Show structural changes
        if !structural_changes.is_empty() {
            println!("=== Structural Changes ===");
            for change in &structural_changes {
                println!("@@@ {}", change.description);
                if let Some(loc) = &change.location1 {
                    println!("  - at line {}: {}", loc.line, loc.code_snippet);
                }
                if let Some(loc) = &change.location2 {
                    println!("  + at line {}: {}", loc.line, loc.code_snippet);
                }
            }
            println!();
        }

        // Show string changes
        if !string_changes.is_empty() {
            println!("=== String Changes ===");
            for change in &string_changes {
                println!("@@@ {}", change.description);
                if let Some(loc) = &change.location1 {
                    println!("  - at line {}: {}", loc.line, loc.code_snippet);
                }
                if let Some(loc) = &change.location2 {
                    println!("  + at line {}: {}", loc.line, loc.code_snippet);
                }
            }
            println!();
        }
    }
    
    pub fn generate_normalized_display_diff(
        orig1: &str, orig2: &str,
        norm1: &str, norm2: &str,
        context_lines: usize,
    ) -> String {
        use similar::{ChangeTag, TextDiff};

        let diff = TextDiff::from_lines(norm1, norm2);
        let orig_lines1: Vec<&str> = orig1.lines().collect();
        let orig_lines2: Vec<&str> = orig2.lines().collect();

        let mut output = String::new();
        let mut has_changes = false;

        for hunk in diff.unified_diff().context_radius(context_lines).iter_hunks() {
            has_changes = true;
            output.push_str(&format!("{}\n", hunk.header()));
            for change in hunk.iter_changes() {
                let sign = match change.tag() {
                    ChangeTag::Delete => "-",
                    ChangeTag::Insert => "+",
                    ChangeTag::Equal => " ",
                };
                // Look up the original (non-normalized) line at the same index
                let orig_line = match change.tag() {
                    ChangeTag::Delete | ChangeTag::Equal => {
                        change.old_index()
                            .and_then(|i| orig_lines1.get(i))
                            .copied()
                            .unwrap_or("")
                    }
                    ChangeTag::Insert => {
                        change.new_index()
                            .and_then(|i| orig_lines2.get(i))
                            .copied()
                            .unwrap_or("")
                    }
                };
                output.push_str(sign);
                output.push_str(orig_line);
                if !orig_line.ends_with('\n') {
                    output.push('\n');
                }
            }
        }

        if has_changes { output } else { String::new() }
    }

    pub fn print_default(&self, result: &DiffResult, file1: &std::path::PathBuf, file2: &std::path::PathBuf,
                         source1: &str, source2: &str) -> Result<()> {
        let file1_name = file1.file_name().unwrap_or(file1.as_os_str()).to_string_lossy();
        let file2_name = file2.file_name().unwrap_or(file2.as_os_str()).to_string_lossy();

        println!("--- {}", file1.display());
        println!("+++ {}", file2.display());
        println!("Structural similarity: {:.1}%", result.similarity * 100.0);
        println!("Matched: {}/{} vs {}",
            result.matched_declarations, result.total_declarations1, result.total_declarations2);

        // Classify changes
        let mut additions = Vec::new();
        let mut deletions = Vec::new();
        let mut structural = Vec::new();
        let mut string_only = Vec::new();
        for change in &result.changes {
            match change.change_type {
                ChangeType::Addition => additions.push(change),
                ChangeType::Deletion => deletions.push(change),
                ChangeType::Modification => {
                    match change.classification.as_ref() {
                        Some(DiffClassification::Structural) => structural.push(change),
                        Some(DiffClassification::StringOnly) => string_only.push(change),
                        Some(DiffClassification::Unchanged) | None => {}
                    }
                }
                ChangeType::Reorder => {} // Implicit in location
            }
        }

        // Count unchanged as: matched - structural - string_only
        let total_unchanged = result.matched_declarations
            .saturating_sub(structural.len())
            .saturating_sub(string_only.len());

        println!("Changes: {} added, {} removed, {} structural, {} string-only ({} unchanged)",
            additions.len(), deletions.len(), structural.len(), string_only.len(), total_unchanged);
        println!();

        // === Removed ===
        if !deletions.is_empty() {
            println!("=== Removed ===");
            let lines1: Vec<&str> = source1.lines().collect();
            for change in &deletions {
                if let Some(loc) = &change.location1 {
                    let end = loc.end_line.unwrap_or(loc.line);
                    println!("\n--- Removed {} ({}:{}-{})",
                        Self::extract_name_from_desc(&change.description),
                        file1_name, loc.line, end);
                    let body = extract_source_range(&lines1, loc.line, end);
                    for line in body.lines() {
                        println!("- {}", line);
                    }
                }
            }
            println!();
        }

        // === Added ===
        if !additions.is_empty() {
            println!("=== Added ===");
            let lines2: Vec<&str> = source2.lines().collect();
            for change in &additions {
                if let Some(loc) = &change.location2 {
                    let end = loc.end_line.unwrap_or(loc.line);
                    println!("\n+++ Added {} ({}:{}-{})",
                        Self::extract_name_from_desc(&change.description),
                        file2_name, loc.line, end);
                    let body = extract_source_range(&lines2, loc.line, end);
                    for line in body.lines() {
                        println!("+ {}", line);
                    }
                }
            }
            println!();
        }

        // === Structural Changes ===
        if !structural.is_empty() {
            println!("=== Structural Changes ===");
            for change in &structural {
                if let (Some(loc1), Some(loc2)) = (&change.location1, &change.location2) {
                    println!("\n@@@ {}", change.description);
                    println!("--- {}:{}", file1_name, loc1.line);
                    println!("+++ {}:{}", file2_name, loc2.line);
                    if !change.display_diff.is_empty() {
                        print!("{}", change.display_diff);
                    }
                }
            }
            println!();
        }

        // === String Changes ===
        if !string_only.is_empty() {
            println!("=== String Changes ===");
            for change in &string_only {
                if let (Some(loc1), Some(loc2)) = (&change.location1, &change.location2) {
                    println!("\n@@@ {}", change.description);
                    println!("--- {}:{}", file1_name, loc1.line);
                    println!("+++ {}:{}", file2_name, loc2.line);
                    if !change.display_diff.is_empty() {
                        print!("{}", change.display_diff);
                    }
                }
            }
            println!();
        }

        Ok(())
    }

    /// Compact output: location-only summary grouped by classification.
    pub fn print_compact_locations(&self, result: &DiffResult, file1: &std::path::PathBuf, file2: &std::path::PathBuf) {
        let file1_name = file1.file_name().unwrap_or(file1.as_os_str()).to_string_lossy();
        let file2_name = file2.file_name().unwrap_or(file2.as_os_str()).to_string_lossy();

        // Classify changes
        let mut additions = Vec::new();
        let mut deletions = Vec::new();
        let mut structural = Vec::new();
        let mut string_only = Vec::new();

        for change in &result.changes {
            match change.change_type {
                ChangeType::Addition => additions.push(change),
                ChangeType::Deletion => deletions.push(change),
                ChangeType::Modification => {
                    match change.classification.as_ref() {
                        Some(DiffClassification::Structural) => structural.push(change),
                        Some(DiffClassification::StringOnly) => string_only.push(change),
                        _ => {} // Unchanged — not shown
                    }
                }
                ChangeType::Reorder => {}
            }
        }

        let total_unchanged = result.matched_declarations
            .saturating_sub(structural.len())
            .saturating_sub(string_only.len());

        // Removed
        if !deletions.is_empty() {
            println!("Removed: {}", deletions.len());
            for change in &deletions {
                if let Some(loc) = &change.location1 {
                    let end = loc.end_line.unwrap_or(loc.line);
                    let name = Self::extract_name_from_desc(&change.description);
                    println!("  {} ({}:{}-{})", name, file1_name, loc.line, end);
                }
            }
            println!();
        }

        // Added
        if !additions.is_empty() {
            println!("Added: {}", additions.len());
            for change in &additions {
                if let Some(loc) = &change.location2 {
                    let end = loc.end_line.unwrap_or(loc.line);
                    let name = Self::extract_name_from_desc(&change.description);
                    println!("  {} ({}:{}-{})", name, file2_name, loc.line, end);
                }
            }
            println!();
        }

        // Structural
        if !structural.is_empty() {
            println!("Structural: {}", structural.len());
            for change in &structural {
                if let (Some(loc1), Some(loc2)) = (&change.location1, &change.location2) {
                    let name = Self::extract_name_from_desc(&change.description);
                    let sim = change.similarity_score.map(|s| format!(" {:.1}%", s * 100.0)).unwrap_or_default();
                    println!("  {} ({}:{} -> {}:{}){}",
                        name, file1_name, loc1.line, file2_name, loc2.line, sim);
                }
            }
            println!();
        }

        // String-only
        if !string_only.is_empty() {
            println!("String-only: {}", string_only.len());
            for change in &string_only {
                if let (Some(loc1), Some(loc2)) = (&change.location1, &change.location2) {
                    let name = Self::extract_name_from_desc(&change.description);
                    println!("  {} ({}:{} -> {}:{})",
                        name, file1_name, loc1.line, file2_name, loc2.line);
                }
            }
            println!();
        }

        println!("Unchanged: {} (not shown)", total_unchanged);
    }

    /// Extract declaration name from a description string.
    fn extract_name_from_desc(desc: &str) -> &str {
        // Try patterns like "function 'foo' ..." or "Removed function 'foo'"
        if let Some(start) = desc.find('\'') {
            if let Some(end) = desc[start+1..].find('\'') {
                return &desc[start+1..start+1+end];
            }
        }
        desc
    }

    pub fn print_side_by_side(&self, result: &DiffResult, file1: &std::path::PathBuf, file2: &std::path::PathBuf,
                               source1: &str, source2: &str) {
        println!("Structural similarity: {:.1}%", result.similarity * 100.0);
        println!();
        // Simplified implementation
        self.print_summary(result, file1, file2, source1, source2);
    }
    
    pub fn print_json(&self, result: &DiffResult) -> Result<()> {
        let json = serde_json::to_string_pretty(result)?;
        println!("{}", json);
        Ok(())
    }
    
    pub fn generate_rename_mapping(&self, result: &DiffResult) -> HashMap<String, String> {
        let mut mappings = HashMap::new();
        
        for change in &result.changes {
            if let ChangeType::Modification = change.change_type {
                if change.description.contains("matched with") {
                    // Extract the rename relationship from the structural_path
                    if let Some((from, to)) = change.structural_path
                        .strip_prefix("global.")
                        .and_then(|s| s.split_once("->")) {
                        mappings.insert(from.to_string(), to.to_string());
                    }
                }
            }
        }
        
        mappings
    }
    
    pub fn match_declarations(&self, decls1: &[Declaration], decls2: &[Declaration], source1: &str, source2: &str)
        -> (Vec<(usize, usize)>, Vec<Change>, HashMap<String, String>) {
        use profiling::Timer;

        eprintln!("Using parallel matching v2 for {} x {} declarations", decls1.len(), decls2.len());

        let scorer = self.build_rarity_scorer(decls1.iter().chain(decls2.iter()));

        // Convert to thread-safe data structures
        let data1: Vec<DeclarationData> = {
            let _timer = Timer::new("convert_to_data_structures");
            decls1.iter().map(|d| d.to_data()).collect()
        };
        let data2: Vec<DeclarationData> = decls2.iter().map(|d| d.to_data()).collect();

        self.run_matcher(data1, data2, source1, source2, scorer)
    }

    /// Consuming counterpart of `match_declarations`: each declaration's hash set and
    /// fingerprint are moved into its `DeclarationData`, so the source vectors free as
    /// they are converted instead of living alongside a full clone.
    fn match_owned_declarations(&self, decls1: Vec<Declaration>, decls2: Vec<Declaration>, source1: &str, source2: &str)
        -> (Vec<(usize, usize)>, Vec<Change>, HashMap<String, String>) {
        use profiling::Timer;

        eprintln!("Using parallel matching v2 for {} x {} declarations", decls1.len(), decls2.len());

        let scorer = self.build_rarity_scorer(decls1.iter().chain(decls2.iter()));

        let data1: Vec<DeclarationData> = {
            let _timer = Timer::new("convert_to_data_structures");
            decls1.into_iter().map(|d| d.into_data()).collect()
        };
        let data2: Vec<DeclarationData> = decls2.into_iter().map(|d| d.into_data()).collect();

        self.run_matcher(data1, data2, source1, source2, scorer)
    }

    fn build_rarity_scorer<'a>(&self, decls: impl Iterator<Item = &'a Declaration>) -> Option<RarityScorer> {
        use profiling::Timer;

        if !self.use_fingerprints {
            return None;
        }

        let _timer = Timer::new("build_rarity_scorer_parallel");
        let mut scorer = RarityScorer::new();
        for decl in decls {
            if let Some(ref fp) = decl.fingerprint {
                scorer.add_fingerprint(fp);
            }
        }

        Some(scorer)
    }

    fn run_matcher(&self, data1: Vec<DeclarationData>, data2: Vec<DeclarationData>,
                   source1: &str, source2: &str, scorer: Option<RarityScorer>)
        -> (Vec<(usize, usize)>, Vec<Change>, HashMap<String, String>) {
        use parallel_matching_v2::ParallelMatcherV2;

        let matcher = ParallelMatcherV2::new(self.use_fingerprints);

        matcher.match_declarations(
            &data1,
            &data2,
            source1,
            source2,
            scorer.as_ref(),
            |d1, d2, s1, s2| self.calculate_declaration_similarity_data(d1, d2, s1, s2),
        )
    }

    fn calculate_declaration_similarity_data(&self, decl1: &DeclarationData, decl2: &DeclarationData, _source1: &str, _source2: &str) -> f64 {
        // For imports and exports, use signature similarity regardless of kind
        if matches!(decl1.kind, DeclarationKind::Import | DeclarationKind::Export)
            || matches!(decl2.kind, DeclarationKind::Import | DeclarationKind::Export) {
            return if decl1.signature == decl2.signature { 1.0 } else { 0.3 };
        }

        let size1 = decl1.structural_hashes.len();
        let size2 = decl2.structural_hashes.len();

        if size1 == 0 && size2 == 0 {
            let base = if decl1.signature == decl2.signature { 1.0 } else { 0.5 };
            return apply_kind_penalty(base, &decl1.kind, &decl2.kind);
        }

        // If one is much larger than the other, they can't be similar enough
        let size_ratio = size1.min(size2) as f64 / size1.max(size2) as f64;
        if size_ratio < 0.3 {
            return 0.2;
        }

        // Jaccard similarity from structural hash intersection.
        // Count directly instead of materializing the two sets: this runs once per
        // surviving candidate pair (50M+ on a large bundle), and collecting sets
        // just to read .len() off them was the single hottest allocation in the
        // tool. |A u B| = |A| + |B| - |A n B| makes the union free.
        let intersection = sorted_intersection_count(&decl1.structural_hashes, &decl2.structural_hashes);
        let union = size1 + size2 - intersection;
        let base_similarity = intersection as f64 / union as f64;

        apply_kind_penalty(base_similarity, &decl1.kind, &decl2.kind)
    }
}