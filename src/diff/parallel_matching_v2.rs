use std::collections::HashMap;
use super::{DeclarationData, DeclarationKind, Change, ChangeType, DiffClassification, MINHASH_LANES};
use super::alpha;
use super::fingerprint::{self, calculate_fingerprint_similarity, RarityScorer};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Instant, Duration};

/// Estimated MinHash similarity a pair must reach to survive the candidate filter.
const LSH_SIMILARITY_THRESHOLD: f64 = 0.3;

/// Lanes compared before the gate checks whether the pair is still reachable.
///
/// Must divide MINHASH_LANES, otherwise the tail lanes would go uncompared; the
/// gate checks that at runtime and falls back to the general path if it ever stops
/// holding.
const LSH_GATE_BLOCK_LANES: usize = 32;

/// Smallest number of agreeing lanes that clears LSH_SIMILARITY_THRESHOLD.
///
/// The gate is `matches / lanes >= threshold`. `matches` is a small integer and
/// `lanes` is a power of two, so that quotient is exact in f64 and the comparison
/// has a single integer crossing point: below it every pair fails, at or above it
/// every pair passes. Finding that point once at compile time lets the hot loop
/// count lanes and compare integers instead of dividing 230 million times, and it
/// is the reason the loop can bail early: once the lanes still uncompared cannot
/// carry the count this far, the pair is already rejected. For the current 128
/// lanes and 0.3 threshold the point is 39, since 38/128 = 0.296875 fails and
/// 39/128 = 0.3046875 passes.
const LSH_MIN_MATCHING_LANES: usize = min_matching_lanes(MINHASH_LANES, LSH_SIMILARITY_THRESHOLD);

const fn min_matching_lanes(lanes: usize, threshold: f64) -> usize {
    let mut matching = 0;

    while matching < lanes {
        if matching as f64 / lanes as f64 >= threshold {
            break;
        }

        matching += 1;
    }

    matching
}

/// One decls2 entry as the window scan sees it, in sorted2 order.
///
/// The scan touches this instead of the full DeclarationData, which is ~200 bytes
/// and drags a String heap allocation in behind every probe. Everything the scan
/// decides on lives here, so a rejected pair costs one sequential read.
struct Decl2Probe {
    size: usize,
    name_id: u32,
    i2: u32,
    kind: DeclarationKind,
    /// False when this declaration's signature is not MINHASH_LANES long, in which
    /// case it has no slot in the flat signature buffer and the scan reads the
    /// declaration's own signature.
    has_flat_signature: bool,
}

/// A pair that survived LSH filtering.
///
/// Indices are u32 because this list is the largest live allocation in the tool
/// (50M+ entries on a 34 MB bundle) and no input has come close to 4 billion
/// declarations. The old struct also carried the LSH score, which nothing ever
/// read; dropping it and narrowing the indices takes the entry from 32 to 12 bytes.
#[derive(Debug, Clone)]
pub struct CandidateMatch {
    pub i1: u32,
    pub i2: u32,
    pub name_match: bool,  // True if names match exactly
}

#[derive(Debug, Clone)]
pub struct SimilarityResult {
    pub i1: usize,
    pub i2: usize,
    pub similarity: f64,
    pub evidence_count: usize,
    pub name_match: bool,  // True if names match exactly
}

pub struct ParallelMatcherV2 {
    use_fingerprints: bool,
    batch_size: usize,
}

impl ParallelMatcherV2 {
    pub fn new(use_fingerprints: bool) -> Self {
        Self {
            use_fingerprints,
            batch_size: 1000, // Process LSH in batches of 1000
        }
    }
    
    pub fn match_declarations(
        &self,
        decls1: &[DeclarationData],
        decls2: &[DeclarationData],
        source1: &str,
        source2: &str,
        scorer: Option<&RarityScorer>,
        calculate_similarity: impl Fn(&DeclarationData, &DeclarationData, &str, &str) -> f64 + Sync,
    ) -> (Vec<(usize, usize)>, Vec<Change>, HashMap<String, String>) {
        use super::profiling::Timer;

        // Steps 1+2: generate candidate pairs and LSH-filter them in one pass, so
        // only survivors are ever stored (see build_and_filter_candidates).
        let lsh_candidates = {
            let _timer = Timer::new("build_and_filter_candidates");
            self.build_and_filter_candidates(decls1, decls2)
        };

        eprintln!("LSH filtering reduced to {} candidates", lsh_candidates.len());

        // Step 3: Parallel full similarity calculation for remaining candidates
        let similarity_results = {
            let _timer = Timer::new("parallel_full_similarity");
            self.parallel_full_similarity(
                &lsh_candidates,
                decls1,
                decls2,
                source1,
                source2,
                scorer,
                &calculate_similarity,
            )
        };

        // Step 4: Resolve best matches + normalize/diff all pairs
        let (matches, changes, rename_map) = {
            let _timer = Timer::new("resolve_matches");
            self.resolve_best_matches(similarity_results, decls1, decls2, source1, source2)
        };

        (matches, changes, rename_map)
    }
    
    /// Generate the candidate pairs and LSH-filter them in a single pass.
    ///
    /// This used to be two steps: materialize every (i1, i2) pair whose sizes and
    /// kinds were compatible, then filter that list down. On a 34 MB bundle the
    /// intermediate list held 230 million pairs, 3.7 GB, and it stayed alive while
    /// the filtered list was built next to it. Worse, a Vec that large grows by
    /// doubling, so the last reallocation needs the old and new buffers resident at
    /// the same time. That transient spike is what aborted the process on hosts
    /// where the memory was not there.
    ///
    /// Testing each pair as it is generated means only the survivors are ever
    /// stored, which is roughly a fifth of the pairs on real input.
    ///
    /// The scan itself reads a flat copy of the decls2 signatures and a compact side
    /// table, both laid out in the same size order the window walks, so a probe costs a
    /// sequential read instead of a pointer chase into a 37k-allocation heap. Neither
    /// copy changes any value the scan compares, only where it reads them from.
    ///
    /// Output order is unchanged (i1 ascending, then decls2 in size order):
    /// resolve_best_matches sorts these by similarity with a stable sort, so the
    /// order here decides tie-breaks and therefore the final diff.
    fn build_and_filter_candidates(&self, decls1: &[DeclarationData], decls2: &[DeclarationData]) -> Vec<CandidateMatch> {
        // Sort declarations by size for efficient window search
        let mut sorted2: Vec<(usize, usize)> = decls2.iter()
            .enumerate()
            .map(|(i, d)| (i, d.size))
            .collect();
        sorted2.sort_by_key(|(_, size)| *size);

        // Names are interned across both files so the scan compares u32 ids instead of
        // two String heap reads per probe. Ids come from one table over both inputs, so
        // two names share an id exactly when the strings are equal, which is the test
        // the id comparison replaces.
        let mut name_ids: HashMap<&str, u32> = HashMap::with_capacity(decls1.len() + decls2.len());
        let mut next_name_id: u32 = 0;

        let name_ids1: Vec<u32> = decls1.iter()
            .map(|decl| *name_ids.entry(decl.name.as_str()).or_insert_with(|| {
                let id = next_name_id;
                next_name_id += 1;
                id
            }))
            .collect();

        // Signatures are copied into one flat buffer in sorted2 order so the window scan
        // reads them front to back. Each declaration otherwise owns its own ~1 KB
        // allocation, which turns the scan into one random pointer chase per probe over a
        // working set far larger than cache.
        let mut flat_signatures: Vec<u64> = Vec::with_capacity(sorted2.len() * MINHASH_LANES);
        let mut probes: Vec<Decl2Probe> = Vec::with_capacity(sorted2.len());

        for &(i2, size) in &sorted2 {
            let decl2 = &decls2[i2];
            let has_flat_signature = decl2.minhash_signature.len() == MINHASH_LANES;

            // The stride stays fixed so `idx` alone locates a signature; an odd-length
            // signature keeps its slot as padding and is read from the declaration.
            if has_flat_signature {
                flat_signatures.extend_from_slice(&decl2.minhash_signature);
            } else {
                flat_signatures.resize(flat_signatures.len() + MINHASH_LANES, 0);
            }

            let name_id = *name_ids.entry(decl2.name.as_str()).or_insert_with(|| {
                let id = next_name_id;
                next_name_id += 1;
                id
            });

            probes.push(Decl2Probe {
                size,
                name_id,
                i2: i2 as u32,
                kind: decl2.kind.clone(),
                has_flat_signature,
            });
        }

        let examined = AtomicUsize::new(0);
        let last_update = Mutex::new(Instant::now());

        let results: Vec<CandidateMatch> = decls1.par_iter()
            .enumerate()
            .flat_map_iter(|(i1, decl1)| {
                let min_size = ((decl1.size as f64) * 0.5).max(1.0) as usize;
                let max_size = ((decl1.size as f64) * 1.5) as usize;
                let name_id1 = name_ids1[i1];

                // Binary search for window start
                let start_idx = probes.partition_point(|probe| probe.size < min_size);

                let mut local_results = Vec::new();
                let mut local_examined = 0usize;

                for idx in start_idx..probes.len() {
                    let probe = &probes[idx];
                    if probe.size > max_size {
                        break;
                    }

                    if decl1.kind != probe.kind {
                        continue;
                    }

                    local_examined += 1;

                    // Always include pairs with matching names - they're almost certainly
                    // the same function/variable and need to be compared for string diffs
                    // even if structural similarity is low (e.g., template string content changed)
                    if name_id1 == probe.name_id {
                        local_results.push(CandidateMatch {
                            i1: i1 as u32,
                            i2: probe.i2,
                            name_match: true,
                        });
                        continue;
                    }

                    let sig2: &[u64] = if probe.has_flat_signature {
                        &flat_signatures[idx * MINHASH_LANES..(idx + 1) * MINHASH_LANES]
                    } else {
                        &decls2[probe.i2 as usize].minhash_signature
                    };

                    if passes_lsh_gate(&decl1.minhash_signature, sig2) {
                        local_results.push(CandidateMatch {
                            i1: i1 as u32,
                            i2: probe.i2,
                            name_match: false,
                        });
                    }
                }

                // Report progress every second
                let done = examined.fetch_add(local_examined, Ordering::Relaxed) + local_examined;

                if let Ok(mut last) = last_update.try_lock() {
                    if last.elapsed() >= Duration::from_secs(1) {
                        eprint!("\r  LSH filtering: {} pairs examined", done);
                        *last = Instant::now();
                    }
                }

                local_results.into_iter()
            })
            .collect();

        // Clear the progress line with a final update
        eprintln!("\r  LSH filtering: {} pairs examined - Complete", examined.load(Ordering::Relaxed));

        results
    }
    
    fn parallel_full_similarity(
        &self,
        candidates: &[CandidateMatch],
        decls1: &[DeclarationData],
        decls2: &[DeclarationData],
        source1: &str,
        source2: &str,
        scorer: Option<&RarityScorer>,
        calculate_similarity: &(impl Fn(&DeclarationData, &DeclarationData, &str, &str) -> f64 + Sync),
    ) -> Vec<SimilarityResult> {
        let progress = AtomicUsize::new(0);
        let total = candidates.len();
        let last_update = Mutex::new(Instant::now());
        
        let results = candidates.par_chunks(self.batch_size / 10) // Smaller batches for expensive calculations
            .flat_map(|batch| {
                let mut results = Vec::with_capacity(batch.len());
                
                for candidate in batch {
                    let decl1 = &decls1[candidate.i1 as usize];
                    let decl2 = &decls2[candidate.i2 as usize];
                    
                    let (similarity, evidence_count) =
                        if self.use_fingerprints {
                            if let (Some(ref fp1), Some(ref fp2), Some(s)) =
                                (&decl1.fingerprint, &decl2.fingerprint, scorer) {
                                let (fp_score, ev_count) = calculate_fingerprint_similarity(fp1, fp2, s);

                                // The structural term can only lower the combined score, so a
                                // pair whose best case already misses the threshold cannot
                                // survive and does not need its structural similarity computed.
                                // The bound is written as the real expression with struct_sim
                                // pinned at its maximum, so it is bit-identical to what the
                                // pair would have scored, and this skips exactly the pairs the
                                // threshold below would have rejected.
                                //
                                // Name matches are exempt: they are kept regardless of score,
                                // and their similarity value decides their place in the sort.
                                let best_case = fp_score * 0.7 + 1.0 * 0.3;
                                if !candidate.name_match
                                    && !should_match_with_score(best_case, ev_count, decl1.size)
                                {
                                    continue;
                                }

                                let struct_sim = calculate_similarity(decl1, decl2, source1, source2);
                                (fp_score * 0.7 + struct_sim * 0.3, ev_count)
                            } else {
                                (calculate_similarity(decl1, decl2, source1, source2), 0)
                            }
                        } else {
                            (calculate_similarity(decl1, decl2, source1, source2), 0)
                        };

                    // Apply thresholds - always include name matches
                    if candidate.name_match || should_match_with_score(similarity, evidence_count, decl1.size) {
                        results.push(SimilarityResult {
                            i1: candidate.i1 as usize,
                            i2: candidate.i2 as usize,
                            similarity,
                            evidence_count,
                            name_match: candidate.name_match,
                        });
                    }
                }
                
                // Report progress every second
                let done = progress.fetch_add(batch.len(), Ordering::Relaxed) + batch.len();
                
                if let Ok(mut last) = last_update.try_lock() {
                    if last.elapsed() >= Duration::from_secs(1) || done == total {
                        eprint!("\r  Full similarity: {}/{} ({:.1}%)", done, total, done as f64 / total as f64 * 100.0);
                        *last = Instant::now();
                    }
                }
                
                results
            })
            .collect();
            
        // Clear the progress line with a final update
        eprintln!("\r  Full similarity: {}/{} (100.0%) - Complete", total, total);
        
        results
    }
    
    fn resolve_best_matches(
        &self,
        mut results: Vec<SimilarityResult>,
        decls1: &[DeclarationData],
        decls2: &[DeclarationData],
        source1: &str,
        source2: &str,
    ) -> (Vec<(usize, usize)>, Vec<Change>, HashMap<String, String>) {
        use super::profiling::Timer;

        // Pre-compute source lines to avoid repeated parsing
        let _timer = Timer::new("precompute_source_lines");
        let lines1: Vec<&str> = source1.lines().collect();
        let lines2: Vec<&str> = source2.lines().collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        let mut matches = Vec::new();
        let mut matched1 = vec![false; decls1.len()];
        let mut matched2 = vec![false; decls2.len()];
        let mut changes = Vec::new();

        // ── Phase A: Greedy matching + build rename map ──
        let mut rename_map: HashMap<String, String> = HashMap::new();
        let mut match_data: Vec<(usize, usize, f64)> = Vec::new(); // (i1, i2, similarity)

        for result in &results {
            if !matched1[result.i1] && !matched2[result.i2] {
                matched1[result.i1] = true;
                matched2[result.i2] = true;
                matches.push((result.i1, result.i2));
                match_data.push((result.i1, result.i2, result.similarity));

                let decl1 = &decls1[result.i1];
                let decl2 = &decls2[result.i2];

                // Build rename map inline: new_name → old_name
                if decl1.name != decl2.name {
                    rename_map.insert(decl2.name.clone(), decl1.name.clone());
                }
            }
        }

        eprintln!("Phase A: {} matches, {} renames", matches.len(), rename_map.len());

        let pairing = TopLevelPairing::new(&matches, decls1, decls2, &lines1, &lines2);

        // ── Phase B: Normalize + diff all matched pairs ──
        // Once the rename map is fixed the pairs are independent, so they are diffed in
        // parallel, one tokenizer per rayon split since a tree-sitter Parser is not Sync.
        // The indexed collect keeps match order, so changes come out as they did serially.
        let outcomes: Vec<(PairTally, Option<Change>)> = match_data
            .par_iter()
            .map_init(
                alpha::AlphaTokenizer::new,
                |tokenizer, &(i1, i2, similarity)| {
                    diff_matched_pair(
                        tokenizer,
                        &decls1[i1],
                        &decls2[i2],
                        similarity,
                        &lines1,
                        &lines2,
                        &rename_map,
                        &pairing,
                    )
                },
            )
            .collect();

        let mut unchanged_count = 0usize;
        let mut string_only_count = 0usize;
        let mut structural_count = 0usize;

        for (tally, change) in outcomes {
            match tally {
                PairTally::Uncounted => {}
                PairTally::Unchanged => unchanged_count += 1,
                PairTally::StringOnly => string_only_count += 1,
                PairTally::Structural => structural_count += 1,
            }

            changes.extend(change);
        }

        eprintln!("Phase B: {} unchanged, {} string-only, {} structural",
            unchanged_count, string_only_count, structural_count);

        // Add deletions and additions
        for (i, decl) in decls1.iter().enumerate() {
            if !matched1[i] {
                changes.push(create_change(
                    ChangeType::Deletion,
                    Some(create_location_with_lines(decl, &lines1)),
                    None,
                    format!("Removed {} '{}'", kind_to_string(&decl.kind), decl.name),
                    format!("global.{}", decl.name),
                ));
            }
        }

        for (i, decl) in decls2.iter().enumerate() {
            if !matched2[i] {
                changes.push(create_change(
                    ChangeType::Addition,
                    None,
                    Some(create_location_with_lines(decl, &lines2)),
                    format!("Added {} '{}'", kind_to_string(&decl.kind), decl.name),
                    format!("global.{}", decl.name),
                ));
            }
        }

        (matches, changes, rename_map)
    }
}

// Helper functions

/// How many lines from the referencing declaration a free reference looks for
/// the declaration it names, and how near the Phase A partner that contradicts
/// it has to sit. A minified bundle concatenates scopes that reuse the same
/// short names at top level, and many bindings (`var x;` assigned later, `let`,
/// `const`) are never extracted, so a same-named declaration far away is more
/// often a stranger than the binding. Past this distance a reference is left
/// to plain alpha-equivalence.
const NEIGHBOURHOOD_LINES: usize = 200;

/// The declarations Phase A matched, by identity rather than by name, so that
/// a free reference can be held to the pairing. Unlike `rename_map` it also
/// holds the pairs whose name did not change.
struct TopLevelPairing<'a> {
    old: TopLevelSide<'a>,
    new: TopLevelSide<'a>,
}

/// One build's declarations, indexed for resolving a name near a given
/// declaration, with each one's Phase A partner on the other side.
struct TopLevelSide<'a> {
    decls: &'a [DeclarationData],
    lines: &'a [&'a str],
    /// Declaration indices per name, in line order.
    by_name: HashMap<&'a str, Vec<usize>>,
    partner: Vec<Option<usize>>,
}

impl<'a> TopLevelPairing<'a> {
    fn new(
        matches: &[(usize, usize)],
        decls1: &'a [DeclarationData],
        decls2: &'a [DeclarationData],
        lines1: &'a [&'a str],
        lines2: &'a [&'a str],
    ) -> Self {
        let mut old = TopLevelSide::new(decls1, lines1);
        let mut new = TopLevelSide::new(decls2, lines2);

        for &(i1, i2) in matches {
            old.partner[i1] = Some(i2);
            new.partner[i2] = Some(i1);
        }

        Self { old, new }
    }

    /// The indices of the free references in a same-shape pair that now name
    /// a different declaration than the pairing says (see [`Self::is_repointed`]).
    fn repointed_refs(
        &self,
        tokenizer: &mut alpha::AlphaTokenizer,
        t1: &alpha::AlphaTokens,
        t2: &alpha::AlphaTokens,
        decl1: &DeclarationData,
        decl2: &DeclarationData,
    ) -> Vec<u32> {
        alpha::free_renames(t1, t2)
            .filter(|&(_, old_name, new_name)| {
                self.is_repointed(tokenizer, old_name, new_name, decl1, decl2)
            })
            .map(|(index, _, _)| index)
            .collect()
    }

    /// Whether the reference read as `old_name` in `decl1` and `new_name` in
    /// `decl2` changed what it points at. Each name stands for every
    /// declaration of that name near its pair on its own side, since nearness
    /// alone cannot tell a reused short name's binding from a neighbour. The
    /// reference is re-pointed only when no old target was matched to a new
    /// one, a contradicting partner sits near the pair (so the pairing has a
    /// local answer that differs), and no old target is identical to a new one
    /// up to renaming, a look-alike Phase A could have cross-paired. String
    /// text counts here: siblings differing only in their strings are exactly
    /// the targets a swapped reference trades between. A name with no
    /// declaration nearby (removed, or bound where nothing is extracted)
    /// contradicts nothing.
    fn is_repointed(
        &self,
        tokenizer: &mut alpha::AlphaTokenizer,
        old_name: &str,
        new_name: &str,
        decl1: &DeclarationData,
        decl2: &DeclarationData,
    ) -> bool {
        let old_targets = self.old.near(old_name, decl1);
        let new_targets = self.new.near(new_name, decl2);

        if old_targets.is_empty() || new_targets.is_empty() {
            return false;
        }

        let is_paired = old_targets
            .iter()
            .filter_map(|&i1| self.old.partner[i1])
            .any(|i2| new_targets.contains(&i2));

        if is_paired {
            return false;
        }

        let is_contradicted_nearby = self.old.has_partner_near(old_targets, &self.new, decl2)
            || self.new.has_partner_near(new_targets, &self.old, decl1);

        if !is_contradicted_nearby {
            return false;
        }

        let old_shapes = self.old.shapes(tokenizer, old_targets);
        let new_shapes = self.new.shapes(tokenizer, new_targets);
        let is_look_alike = old_shapes
            .iter()
            .any(|a| new_shapes.iter().any(|b| alpha::alpha_equal(a, b)));

        !is_look_alike
    }
}

impl<'a> TopLevelSide<'a> {
    fn new(decls: &'a [DeclarationData], lines: &'a [&'a str]) -> Self {
        let mut by_name: HashMap<&'a str, Vec<usize>> = HashMap::new();

        for (index, decl) in decls.iter().enumerate() {
            by_name.entry(decl.name.as_str()).or_default().push(index);
        }

        for indices in by_name.values_mut() {
            indices.sort_by_key(|&index| decls[index].line);
        }

        Self {
            decls,
            lines,
            by_name,
            partner: vec![None; decls.len()],
        }
    }

    /// The declarations named `name` within [`NEIGHBOURHOOD_LINES`] of
    /// `from`, in line order. Top-level declarations do not overlap, so their
    /// end lines rise with their start lines and two binary searches bound
    /// the window.
    fn near(&self, name: &str, from: &DeclarationData) -> &[usize] {
        let Some(candidates) = self.by_name.get(name) else {
            return &[];
        };
        let window_start = from.line.saturating_sub(NEIGHBOURHOOD_LINES);
        let window_end = from.end_line + NEIGHBOURHOOD_LINES;
        let start = candidates.partition_point(|&index| self.decls[index].end_line < window_start);
        let end = candidates.partition_point(|&index| self.decls[index].line <= window_end);

        // Overlapping spans would break the ordering; an empty window beats a panic.
        &candidates[start..end.max(start)]
    }

    fn is_near(&self, index: usize, from: &DeclarationData) -> bool {
        line_gap(&self.decls[index], from) <= NEIGHBOURHOOD_LINES
    }

    /// Whether any of `targets` was matched to a declaration on the `other`
    /// side within [`NEIGHBOURHOOD_LINES`] of `from`.
    fn has_partner_near(
        &self,
        targets: &[usize],
        other: &TopLevelSide,
        from: &DeclarationData,
    ) -> bool {
        targets
            .iter()
            .any(|&index| self.partner[index].is_some_and(|p| other.is_near(p, from)))
    }

    /// The declarator text of each of `targets`, tokenized for comparing
    /// declarations by shape.
    fn shapes(
        &self,
        tokenizer: &mut alpha::AlphaTokenizer,
        targets: &[usize],
    ) -> Vec<alpha::AlphaTokens> {
        targets
            .iter()
            .map(|&index| tokenizer.tokenize(&self.declarator_text(index)))
            .collect()
    }

    /// The source of declaration `index` without the `var`/`let`/`const` a
    /// first declarator carries or the `,`/`;` that ends it, so two
    /// declarators compare by what they declare, not by where they sit in
    /// their statement.
    fn declarator_text(&self, index: usize) -> String {
        let decl = &self.decls[index];
        let source = super::extract_source_range(self.lines, decl.line, decl.end_line);
        let text = source.trim();
        let text = ["var ", "let ", "const "]
            .iter()
            .find_map(|keyword| text.strip_prefix(keyword))
            .unwrap_or(text);

        text.trim_end_matches([',', ';']).to_string()
    }
}

/// Lines between two declarations' spans, zero when they overlap.
fn line_gap(decl: &DeclarationData, from: &DeclarationData) -> usize {
    if decl.end_line < from.line {
        from.line - decl.end_line
    } else {
        decl.line.saturating_sub(from.end_line)
    }
}

/// How one matched pair counts toward the Phase B summary line.
enum PairTally {
    /// Source could not be extracted and the names agree: no change, not counted.
    Uncounted,
    Unchanged,
    StringOnly,
    Structural,
}

/// Classify and diff one matched pair: the per-pair body of Phase B.
#[allow(clippy::too_many_arguments)]
fn diff_matched_pair(
    tokenizer: &mut alpha::AlphaTokenizer,
    decl1: &DeclarationData,
    decl2: &DeclarationData,
    similarity: f64,
    lines1: &[&str],
    lines2: &[&str],
    rename_map: &HashMap<String, String>,
    pairing: &TopLevelPairing,
) -> (PairTally, Option<Change>) {
    use super::StructuralDiff;

    // Extract source for both declarations
    let src1 = super::extract_source_range(lines1, decl1.line, decl1.end_line);
    let src2 = super::extract_source_range(lines2, decl2.line, decl2.end_line);

    if src1.is_empty() || src2.is_empty() {
        // Can't extract source — skip diffing
        if decl1.name != decl2.name {
            return (PairTally::Unchanged, Some(create_classified_change(
                ChangeType::Modification,
                Some(create_location_with_lines(decl1, lines1)),
                Some(create_location_with_lines(decl2, lines2)),
                format!("{} '{}' matched with '{}' (was '{}')",
                    kind_to_string(&decl1.kind), decl2.name, decl1.name, decl1.name),
                format!("global.{}->{}", decl1.name, decl2.name),
                DiffClassification::Unchanged,
                String::new(),
                Some(similarity),
            )));
        }

        return (PairTally::Uncounted, None);
    }

    let is_import = matches!(decl1.kind, DeclarationKind::Import);

    let (classification, display_diff) = if is_import {
        // Imports keep the string-normalization path: import canonicalization
        // collapses multiline import lists, which token comparison would
        // misread as churn.
        let pre_s1 = fingerprint::normalize_for_comparison(&src1, true);
        let pre_s2 = fingerprint::normalize_for_comparison(&src2, true);

        let renamed = if rename_map.is_empty() {
            pre_s2
        } else {
            fingerprint::normalize_string_with_renames(&pre_s2, rename_map)
        };
        let comp_s1 = fingerprint::normalize_minified_identifiers(&pre_s1);
        let comp_s2 = fingerprint::normalize_minified_identifiers(&renamed);

        if comp_s1 == comp_s2 {
            (DiffClassification::Unchanged, String::new())
        } else {
            let display_diff = StructuralDiff::generate_normalized_display_diff(
                &src1, &src2, &comp_s1, &comp_s2, 3,
            );

            if display_diff.is_empty() {
                (DiffClassification::Unchanged, String::new())
            } else {
                (fingerprint::classify_diff_lines(&display_diff), display_diff)
            }
        }
    } else {
        // Token-level alpha-equivalence: a consistent rename (top-level or
        // function-local, any identifier length) compares equal, and masking
        // string content separates string-only edits from structural ones.
        // The bijection it finds is its own, though, so free references to
        // top-level declarations are also held to the Phase A pairing: a
        // table re-pointed at different declarations is not a rename. A bare
        // alias (`var a = b;`) is exempt: it has no body or strings for Phase
        // A to pair it by, so a reference that disagrees with the pairing
        // there says the alias pair is a guess, not that the code changed.
        let t1 = tokenizer.tokenize(&src1);
        let t2 = tokenizer.tokenize(&src2);
        let is_same_shape = alpha::alpha_equal_masked(&t1, &t2);
        let repointed = if is_same_shape && !t1.is_bare_alias() {
            pairing.repointed_refs(tokenizer, &t1, &t2, decl1, decl2)
        } else {
            Vec::new()
        };
        let is_rename_only = is_same_shape && repointed.is_empty();

        if is_rename_only && alpha::alpha_equal(&t1, &t2) {
            (DiffClassification::Unchanged, String::new())
        } else {
            let classification = if is_rename_only {
                DiffClassification::StringOnly
            } else {
                DiffClassification::Structural
            };
            let (norm1, norm2) = if repointed.is_empty() {
                (t1.norm_lines(), t2.norm_lines())
            } else {
                (
                    t1.norm_lines_flagging(&repointed, '-'),
                    t2.norm_lines_flagging(&repointed, '+'),
                )
            };
            let display_diff =
                StructuralDiff::generate_alpha_display_diff(&src1, &src2, &norm1, &norm2, 3);

            (classification, display_diff)
        }
    };

    if matches!(classification, DiffClassification::Unchanged) {
        return (PairTally::Unchanged, None);
    }

    let desc = if decl1.name != decl2.name {
        match classification {
            DiffClassification::StringOnly =>
                format!("{} '{}' (was '{}') — string-only",
                    kind_to_string(&decl1.kind), decl2.name, decl1.name),
            DiffClassification::Structural =>
                format!("{} '{}' (was '{}') — structural ({:.1}%)",
                    kind_to_string(&decl1.kind), decl2.name, decl1.name, similarity * 100.0),
            DiffClassification::Unchanged => unreachable!(),
        }
    } else {
        match classification {
            DiffClassification::StringOnly =>
                format!("{} '{}' — string-only",
                    kind_to_string(&decl1.kind), decl1.name),
            DiffClassification::Structural =>
                format!("{} '{}' — structural ({:.1}%)",
                    kind_to_string(&decl1.kind), decl1.name, similarity * 100.0),
            DiffClassification::Unchanged => unreachable!(),
        }
    };

    let structural_path = if decl1.name != decl2.name {
        format!("global.{}->{}", decl1.name, decl2.name)
    } else {
        format!("global.{}", decl1.name)
    };

    let tally = match classification {
        DiffClassification::StringOnly => PairTally::StringOnly,
        DiffClassification::Structural => PairTally::Structural,
        DiffClassification::Unchanged => unreachable!(),
    };

    (tally, Some(create_classified_change(
        ChangeType::Modification,
        Some(create_location_with_lines(decl1, lines1)),
        Some(create_location_with_lines(decl2, lines2)),
        desc,
        structural_path,
        classification,
        display_diff,
        Some(similarity),
    )))
}

fn estimate_minhash_similarity(sig1: &[u64], sig2: &[u64]) -> f64 {
    let matches = sig1.iter().zip(sig2).filter(|(a, b)| a == b).count();
    matches as f64 / sig1.len() as f64
}

/// Whether a pair's MinHash signatures agree on enough lanes to stay a candidate.
///
/// Equivalent to `estimate_minhash_similarity(sig1, sig2) >= LSH_SIMILARITY_THRESHOLD`,
/// but it never finishes counting a pair that has already lost: after each block the
/// lanes still uncompared are added to the count as if all of them agreed, and if even
/// that best case falls short the pair is rejected. On real input most pairs die in the
/// first block, which is most of the 230 million pair scan.
///
/// Signatures of any other length go through the original division, so a future change
/// to the lane count cannot silently change which pairs survive.
fn passes_lsh_gate(sig1: &[u64], sig2: &[u64]) -> bool {
    let blocked = sig1.len() == MINHASH_LANES
        && sig2.len() == MINHASH_LANES
        && MINHASH_LANES % LSH_GATE_BLOCK_LANES == 0;

    if !blocked {
        return estimate_minhash_similarity(sig1, sig2) >= LSH_SIMILARITY_THRESHOLD;
    }

    let mut matching = 0usize;
    let mut uncompared = MINHASH_LANES;

    for (block1, block2) in sig1.chunks_exact(LSH_GATE_BLOCK_LANES)
        .zip(sig2.chunks_exact(LSH_GATE_BLOCK_LANES))
    {
        matching += block1.iter().zip(block2).filter(|(a, b)| a == b).count();
        uncompared -= LSH_GATE_BLOCK_LANES;

        if matching + uncompared < LSH_MIN_MATCHING_LANES {
            return false;
        }
    }

    matching >= LSH_MIN_MATCHING_LANES
}

fn should_match_with_score(similarity: f64, evidence_count: usize, size: usize) -> bool {
    if evidence_count > 0 {
        match evidence_count {
            1 => similarity >= 0.6,
            2 => similarity >= 0.45,
            3..=4 => similarity >= 0.4,
            _ => similarity >= 0.35,
        }
    } else {
        if similarity >= 0.85 {
            true
        } else if size < 10 {
            similarity >= 0.7
        } else if size < 50 {
            similarity >= 0.5
        } else {
            similarity >= 0.4
        }
    }
}

fn create_change(
    change_type: ChangeType,
    location1: Option<super::Location>,
    location2: Option<super::Location>,
    description: String,
    structural_path: String,
) -> super::Change {
    super::Change {
        change_type,
        location1,
        location2,
        description,
        structural_path,
        classification: None,
        display_diff: String::new(),
        similarity_score: None,
    }
}

fn create_classified_change(
    change_type: ChangeType,
    location1: Option<super::Location>,
    location2: Option<super::Location>,
    description: String,
    structural_path: String,
    classification: super::DiffClassification,
    display_diff: String,
    similarity_score: Option<f64>,
) -> super::Change {
    super::Change {
        change_type,
        location1,
        location2,
        description,
        structural_path,
        classification: Some(classification),
        display_diff,
        similarity_score,
    }
}

fn create_location_with_lines(decl: &DeclarationData, lines: &[&str]) -> super::Location {
    let snippet = if decl.line > 0 && decl.line <= lines.len() {
        lines[decl.line - 1].trim().to_string()
    } else {
        String::new()
    };
    
    super::Location {
        line: decl.line,
        column: 0,
        code_snippet: snippet,
        end_line: Some(decl.end_line),
    }
}

fn kind_to_string(kind: &DeclarationKind) -> &'static str {
    match kind {
        DeclarationKind::Function => "function",
        DeclarationKind::Class => "class",
        DeclarationKind::Variable => "variable",
        DeclarationKind::Import => "import",
        DeclarationKind::Export => "export",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::StructuralDiff;

    /// A base table, an override that spreads it and changes one key, and a
    /// dispatch table pointing its default row at the override and one model
    /// row at the base.
    const OLD_TABLES: &str = r#"var pick = (cell) => ({ cell: cell, mode: "typed" }),
  tierLow = { cell: "low", mode: "typed" },
  baseTable = {
    low: tierLow,
    medium: { ...pick("alpha-min"), measured: !0 },
    high: { ...pick("alpha-min"), measured: !0 },
  },
  overrideTable = { ...baseTable, high: { ...pick("beta-high"), measured: !0 } },
  dispatchTable = {
    default: overrideTable,
    "model-y": { low: pick("y-low"), high: pick("y-high") },
    "model-x": baseTable,
  };
"#;

    /// The next build: the base is renamed and its strings edited, the old
    /// override is gone, a new override is added, and the dispatch table is
    /// re-pointed (default at the base, the model row at the new override).
    /// Token for token the dispatch table is a pure rename of the old one.
    const REPOINTED_TABLES: &str = r#"var choose = (cell) => ({ cell: cell, mode: "typed" }),
  lowTier = { cell: "low", mode: "typed" },
  rootTable = {
    low: lowTier,
    medium: { ...choose("beta-high"), measured: !0 },
    high: { ...choose("beta-high"), measured: !0 },
  },
  patchTable = {
    ...rootTable,
    medium: { ...choose("alpha-min"), measured: !0 },
    high: { ...choose("alpha-min"), measured: !0 },
  },
  routeTable = {
    default: rootTable,
    "model-y": { low: choose("y-low"), high: choose("y-high") },
    "model-x": patchTable,
  };
"#;

    /// OLD_TABLES with every name changed and nothing else.
    const RENAMED_TABLES: &str = r#"var choose = (cell) => ({ cell: cell, mode: "typed" }),
  lowTier = { cell: "low", mode: "typed" },
  rootTable = {
    low: lowTier,
    medium: { ...choose("alpha-min"), measured: !0 },
    high: { ...choose("alpha-min"), measured: !0 },
  },
  patchTable = { ...rootTable, high: { ...choose("beta-high"), measured: !0 } },
  routeTable = {
    default: patchTable,
    "model-y": { low: choose("y-low"), high: choose("y-high") },
    "model-x": rootTable,
  };
"#;

    /// RENAMED_TABLES with the default and model rows swapped.
    const SWAPPED_TABLES: &str = r#"var choose = (cell) => ({ cell: cell, mode: "typed" }),
  lowTier = { cell: "low", mode: "typed" },
  rootTable = {
    low: lowTier,
    medium: { ...choose("alpha-min"), measured: !0 },
    high: { ...choose("alpha-min"), measured: !0 },
  },
  patchTable = { ...rootTable, high: { ...choose("beta-high"), measured: !0 } },
  routeTable = {
    default: rootTable,
    "model-y": { low: choose("y-low"), high: choose("y-high") },
    "model-x": patchTable,
  };
"#;

    const ALL_RENAMED: &[(&str, &str)] = &[
        ("pick", "choose"),
        ("tierLow", "lowTier"),
        ("baseTable", "rootTable"),
        ("overrideTable", "patchTable"),
        ("dispatchTable", "routeTable"),
    ];

    /// Phase A's outcome for OLD_TABLES -> REPOINTED_TABLES: the old override
    /// was removed and the new one added, so neither is paired.
    const REPOINTED_PAIRS: &[(&str, &str)] = &[
        ("pick", "choose"),
        ("tierLow", "lowTier"),
        ("baseTable", "rootTable"),
        ("dispatchTable", "routeTable"),
    ];

    fn declarations(src: &str) -> Vec<DeclarationData> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(tree_sitter_javascript::language())
            .expect("tree-sitter-javascript language must load");
        let tree = parser.parse(src, None).expect("fixture parses");

        StructuralDiff::new()
            .extract_declarations(tree.root_node(), src)
            .into_iter()
            .map(|decl| decl.into_data())
            .collect()
    }

    fn index_of(decls: &[DeclarationData], name: &str) -> usize {
        decls
            .iter()
            .position(|decl| decl.name == name)
            .unwrap_or_else(|| panic!("fixture declares '{name}'"))
    }

    /// A fixture declaration by name and occurrence (0 for the first
    /// declaration of that name), for fixtures that declare a name twice.
    type At = (&'static str, usize);

    fn index_at(decls: &[DeclarationData], (name, nth): At) -> usize {
        decls
            .iter()
            .enumerate()
            .filter(|(_, decl)| decl.name == name)
            .nth(nth)
            .map(|(index, _)| index)
            .unwrap_or_else(|| panic!("fixture declares '{name}' more than {nth} times"))
    }

    /// Phase B for the pair `old_name -> new_name`, with Phase A's outcome
    /// given as `pairs` instead of left to the matcher's scoring.
    fn diff_pair(
        old_src: &str,
        new_src: &str,
        pairs: &[(&str, &str)],
        old_name: &str,
        new_name: &str,
    ) -> (PairTally, Option<Change>) {
        let decls1 = declarations(old_src);
        let decls2 = declarations(new_src);
        let matches: Vec<(usize, usize)> = pairs
            .iter()
            .map(|&(old, new)| (index_of(&decls1, old), index_of(&decls2, new)))
            .collect();
        let pair = (index_of(&decls1, old_name), index_of(&decls2, new_name));

        diff_indexed(old_src, new_src, &decls1, &decls2, &matches, pair)
    }

    /// [`diff_pair`] with every declaration named by its occurrence as well.
    fn diff_at(
        old_src: &str,
        new_src: &str,
        pairs: &[(At, At)],
        old: At,
        new: At,
    ) -> (PairTally, Option<Change>) {
        let decls1 = declarations(old_src);
        let decls2 = declarations(new_src);
        let matches: Vec<(usize, usize)> = pairs
            .iter()
            .map(|&(old_at, new_at)| (index_at(&decls1, old_at), index_at(&decls2, new_at)))
            .collect();
        let pair = (index_at(&decls1, old), index_at(&decls2, new));

        diff_indexed(old_src, new_src, &decls1, &decls2, &matches, pair)
    }

    /// Phase B for the declarations at `(i1, i2)`, given Phase A's `matches`,
    /// with the rename map built the way Phase A builds it.
    fn diff_indexed(
        old_src: &str,
        new_src: &str,
        decls1: &[DeclarationData],
        decls2: &[DeclarationData],
        matches: &[(usize, usize)],
        (i1, i2): (usize, usize),
    ) -> (PairTally, Option<Change>) {
        let rename_map: HashMap<String, String> = matches
            .iter()
            .map(|&(m1, m2)| (&decls1[m1].name, &decls2[m2].name))
            .filter(|(old, new)| old != new)
            .map(|(old, new)| (new.clone(), old.clone()))
            .collect();
        let lines1: Vec<&str> = old_src.lines().collect();
        let lines2: Vec<&str> = new_src.lines().collect();
        let pairing = TopLevelPairing::new(matches, decls1, decls2, &lines1, &lines2);

        diff_matched_pair(
            &mut alpha::AlphaTokenizer::new(),
            &decls1[i1],
            &decls2[i2],
            1.0,
            &lines1,
            &lines2,
            &rename_map,
            &pairing,
        )
    }

    #[test]
    fn a_table_repointed_at_other_declarations_is_structural() {
        // The dispatch table now reads its default row from the base and its
        // model row from the new override, which a local bijection would
        // excuse as a rename (old override <-> base, base <-> new override).
        let (tally, change) = diff_pair(
            OLD_TABLES,
            REPOINTED_TABLES,
            REPOINTED_PAIRS,
            "dispatchTable",
            "routeTable",
        );
        let change = change.expect("a re-pointed table is reported");

        assert!(matches!(tally, PairTally::Structural));
        assert_eq!(change.classification, Some(DiffClassification::Structural));
        assert!(
            change.display_diff.contains(r#""model-x": patchTable"#),
            "display diff shows the re-pointed model row:\n{}",
            change.display_diff
        );
        assert!(
            change.display_diff.contains("default: rootTable"),
            "display diff shows the re-pointed default row:\n{}",
            change.display_diff
        );
    }

    #[test]
    fn the_renamed_base_keeps_its_string_only_classification() {
        // Its own references (the low tier, the helper) follow the pairing,
        // so only its string edit is reported.
        let (tally, change) = diff_pair(
            OLD_TABLES,
            REPOINTED_TABLES,
            REPOINTED_PAIRS,
            "baseTable",
            "rootTable",
        );

        assert!(matches!(tally, PairTally::StringOnly));
        assert_eq!(
            change.and_then(|c| c.classification),
            Some(DiffClassification::StringOnly)
        );
    }

    #[test]
    fn a_table_whose_references_follow_the_pairing_is_unchanged() {
        let (tally, change) = diff_pair(
            OLD_TABLES,
            RENAMED_TABLES,
            ALL_RENAMED,
            "dispatchTable",
            "routeTable",
        );

        assert!(matches!(tally, PairTally::Unchanged));
        assert!(change.is_none());
    }

    #[test]
    fn swapping_two_paired_references_is_structural() {
        // Both targets are paired, each with the other's successor.
        let (tally, change) = diff_pair(
            OLD_TABLES,
            SWAPPED_TABLES,
            ALL_RENAMED,
            "dispatchTable",
            "routeTable",
        );
        let change = change.expect("a swap is reported");

        assert!(matches!(tally, PairTally::Structural));
        assert!(change.display_diff.contains(r#""model-x": patchTable"#));
    }

    #[test]
    fn a_parameter_named_like_a_top_level_declaration_keeps_alpha_equivalence() {
        // `baseTable` is a parameter here, not the top-level table, so its
        // rename to `patchTable` is local even though both names are paired
        // with other declarations.
        let old = "var baseTable = { low: 1 },\n  overrideTable = { low: 2 },\n  lookup = (baseTable) => baseTable.low;\n";
        let new = "var rootTable = { low: 1 },\n  patchTable = { low: 2 },\n  find = (patchTable) => patchTable.low;\n";
        let pairs = [
            ("baseTable", "rootTable"),
            ("overrideTable", "patchTable"),
            ("lookup", "find"),
        ];

        let (tally, change) = diff_pair(old, new, &pairs, "lookup", "find");

        assert!(matches!(tally, PairTally::Unchanged));
        assert!(change.is_none());
    }

    /// Two regions that reuse the same short names at top level, as
    /// concatenated scopes in a minified bundle do, split by a blank line
    /// that [`spread`] widens past the neighbourhood. The first holds a base,
    /// an override and a table; the second, three helpers.
    const OLD_REGIONS: &str = r#"var e = { low: "a", high: "b" },
  f = { ...e, high: "c" },
  m = {
    default: f,
    "model-x": e,
  };

var e = (n) => n + 1,
  f = (n) => n * 2,
  m = (n) => f(e(n));
"#;

    /// The next build: the first region's override is replaced and its table
    /// re-pointed (default at the base, the model row at the new override);
    /// the second region is renamed and nothing else.
    const REPOINTED_REGIONS: &str = r#"var g = { low: "a", high: "c" },
  x = { ...g, high: "b" },
  k = {
    default: g,
    "model-x": x,
  };

var x = (n) => n + 1,
  g = (n) => n * 2,
  k = (n) => g(x(n));
"#;

    /// Phase A's outcome for OLD_REGIONS -> REPOINTED_REGIONS: the first
    /// region's old override was removed and its new one added.
    const REGION_PAIRS: &[(At, At)] = &[
        (("e", 0), ("g", 0)),
        (("m", 0), ("k", 0)),
        (("e", 1), ("x", 1)),
        (("f", 1), ("g", 1)),
        (("m", 1), ("k", 1)),
    ];

    /// `regions` with the blank line between its regions widened so that
    /// neither region is within the other's neighbourhood. Same-named
    /// declarations inside one neighbourhood are deliberately read as
    /// consistent, so the regions have to sit apart, as they do in a bundle.
    fn spread(regions: &str) -> String {
        regions.replace("\n\n", &"\n".repeat(NEIGHBOURHOOD_LINES + 50))
    }

    #[test]
    fn a_table_repointed_among_names_reused_elsewhere_is_structural() {
        // Every name the table reads is declared again in the other region,
        // so the check has to follow each reference to its own neighbour.
        let (tally, change) = diff_at(
            &spread(OLD_REGIONS),
            &spread(REPOINTED_REGIONS),
            REGION_PAIRS,
            ("m", 0),
            ("k", 0),
        );
        let change = change.expect("a re-pointed table is reported");

        assert!(matches!(tally, PairTally::Structural));
        assert!(
            change.display_diff.contains(r#""model-x": x"#),
            "display diff shows the re-pointed model row:\n{}",
            change.display_diff
        );
        assert!(
            change.display_diff.contains("default: g"),
            "display diff shows the re-pointed default row:\n{}",
            change.display_diff
        );
    }

    #[test]
    fn a_renamed_region_stays_unchanged_beside_a_repointed_one() {
        // The helper reads the second region's `f` and `e`, which follow the
        // pairing; the first region's same-named declarations do not, and must
        // not be taken for them.
        let (tally, change) = diff_at(
            &spread(OLD_REGIONS),
            &spread(REPOINTED_REGIONS),
            REGION_PAIRS,
            ("m", 1),
            ("k", 1),
        );

        assert!(matches!(tally, PairTally::Unchanged));
        assert!(change.is_none());
    }

    #[test]
    fn a_reference_moved_between_cross_paired_look_alikes_is_unchanged() {
        // Phase A paired the two factories by position, each with the other's
        // successor, but they differ only in a string, so the table's
        // references still name the same code.
        let old = r#"var u = () => ({ id: "one" }),
  v = () => ({ id: "two" }),
  w = { first: u, second: v };
"#;
        let new = r#"var z = () => ({ id: "two" }),
  s = () => ({ id: "one" }),
  y = { first: s, second: z };
"#;
        let pairs = [("u", "z"), ("v", "s"), ("w", "y")];

        let (tally, change) = diff_pair(old, new, &pairs, "w", "y");

        assert!(matches!(tally, PairTally::Unchanged));
        assert!(change.is_none());
    }

    #[test]
    fn a_reference_to_a_name_declared_twice_nearby_follows_either_declaration() {
        // The next build declares `store` twice: the function the caller
        // reads, and a class just below the caller. The class is the nearer
        // of the two, but the function is the old one's partner, so the
        // caller is a pure rename.
        let old = r#"function loadStore() {
  return { kind: "store" };
}
function pad(n) {
  const out = n;
  const twice = out * 2;
  const thrice = out * 3;
  return twice + thrice;
}
function start() {
  return loadStore().open();
}
class Shelf {
  open() {}
}
"#;
        let new = r#"function store() {
  return { kind: "store" };
}
function pad(n) {
  const out = n;
  const twice = out * 2;
  const thrice = out * 3;
  return twice + thrice;
}
function begin() {
  return store().open();
}
class store {
  open() {}
}
"#;
        let pairs = [
            (("loadStore", 0), ("store", 0)),
            (("pad", 0), ("pad", 0)),
            (("start", 0), ("begin", 0)),
            (("Shelf", 0), ("store", 1)),
        ];

        let (tally, change) = diff_at(old, new, &pairs, ("start", 0), ("begin", 0));

        assert!(matches!(tally, PairTally::Unchanged));
        assert!(change.is_none());
    }

    #[test]
    fn a_bare_alias_is_not_held_to_the_pairing() {
        // The alias reads a different table's successor than the pairing
        // says, but an alias has nothing of its own for Phase A to pair it
        // by, so the disagreement indicts the alias pair, not the code.
        let old = "var small = { size: 1 },\n  other = { size: 2 };\nvar alias = small;\n";
        let new = "var tiny = { size: 1 },\n  large = { size: 2 };\nvar link = large;\n";
        let pairs = [("small", "tiny"), ("other", "large"), ("alias", "link")];

        let (tally, change) = diff_pair(old, new, &pairs, "alias", "link");

        assert!(matches!(tally, PairTally::Unchanged));
        assert!(change.is_none());
    }

    #[test]
    fn a_swap_between_string_only_siblings_is_structural() {
        // The two rows' targets differ only in a string, but they are still
        // different code: swapping which row reads which is a real change.
        let old = r#"var alpha = { medium: label("x") },
  beta = { medium: label("y") },
  table = { default: alpha, "m": beta };
"#;
        let new = r#"var first = { medium: label("x") },
  second = { medium: label("y") },
  routes = { default: second, "m": first };
"#;
        let pairs = [("alpha", "first"), ("beta", "second"), ("table", "routes")];

        let (tally, change) = diff_pair(old, new, &pairs, "table", "routes");
        let change = change.expect("a swap is reported");

        assert!(matches!(tally, PairTally::Structural));
        assert!(
            change.display_diff.contains("default: second"),
            "display diff shows the swapped default row:\n{}",
            change.display_diff
        );
    }
}
