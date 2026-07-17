// Copyright 2024 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Minimal reproducer for the Hox divergence-minting bug.
//!
//! Hox runs N concurrent agents against ONE shared jj repo using per-agent
//! op-heads isolation (file-copy fork + file-copy merge-back). After the
//! merge-back reconcile, a single slice change can end up with multiple VISIBLE
//! commits that are different *generations* of the same linear rewrite chain
//! (same change_id) -> jj reports the change as divergent.
//!
//! v3 invariant: `dedup_evolved_heads` NEVER authors new commits. All hiding
//! is done exclusively via `remove_head` on the view. The subsequent
//! `rebase_descendants` after dedup is always a no-op for dedup-sourced
//! removals.
//!
//! These tests build the exact mechanisms with the `brevity` module primitives
//! (the same code Hox calls) and assert the v3 invariants.

use std::collections::HashMap;
use std::collections::HashSet;

use jj_lib::backend::ChangeId;
use jj_lib::backend::CommitId;
use jj_lib::brevity;
use jj_lib::object_id::ObjectId as _;
#[allow(unused_imports)]
use jj_lib::op_heads_store::OpHeadsStore as _;
use jj_lib::repo::ReadonlyRepo;
use jj_lib::repo::Repo as _;
use pollster::FutureExt as _;
use testutils::TestRepo;
use testutils::create_random_commit;
use testutils::create_tree;
use testutils::repo_path;

/// Count visible commits grouped by change_id in the head view.
fn visible_commits_by_change(repo: &ReadonlyRepo) -> HashMap<ChangeId, Vec<CommitId>> {
    // Walk all ancestors of the visible heads and group by change id.
    let mut map: HashMap<ChangeId, Vec<CommitId>> = HashMap::new();
    let heads: Vec<CommitId> = repo.view().heads().iter().cloned().collect();
    let mut visited = std::collections::HashSet::new();
    let mut stack = heads;
    while let Some(id) = stack.pop() {
        if !visited.insert(id.clone()) {
            continue;
        }
        let commit = repo.store().get_commit(&id).unwrap();
        map.entry(commit.change_id().clone())
            .or_default()
            .push(id.clone());
        for parent in commit.parent_ids() {
            stack.push(parent.clone());
        }
    }
    map
}

fn dump_chain(repo: &ReadonlyRepo, label: &str) {
    eprintln!("=== {label} ===");
    let by_change = visible_commits_by_change(repo);
    for (change, commits) in &by_change {
        eprintln!(
            "  change {} -> {} visible commit(s):",
            &change.reverse_hex()[..12.min(change.reverse_hex().len())],
            commits.len()
        );
        for c in commits {
            let commit = repo.store().get_commit(c).unwrap();
            eprintln!(
                "    {} desc={:?} parents={:?}",
                &c.hex()[..12.min(c.hex().len())],
                commit.description(),
                commit
                    .parent_ids()
                    .iter()
                    .map(|p| p.hex()[..12.min(p.hex().len())].to_string())
                    .collect::<Vec<_>>()
            );
        }
    }
}

/// Collect all commit ids visible (reachable from heads) in an op.
fn collect_visible_commits(
    loader: &jj_lib::repo::RepoLoader,
    op: &jj_lib::operation::Operation,
) -> HashSet<CommitId> {
    let repo = loader.load_at(op).block_on().unwrap();
    let heads: Vec<CommitId> = repo.view().heads().iter().cloned().collect();
    let mut result: HashSet<CommitId> = HashSet::new();
    let mut stack = heads;
    let mut visited: HashSet<CommitId> = HashSet::new();
    while let Some(id) = stack.pop() {
        if !visited.insert(id.clone()) {
            continue;
        }
        result.insert(id.clone());
        if let Ok(commit) = repo.store().get_commit(&id) {
            for pid in commit.parent_ids() {
                stack.push(pid.clone());
            }
        }
    }
    result
}

/// Returns the set of commit ids that appear in the merged result but were not
/// visible in ANY of the input ops (i.e. commits authored by merge_operations).
/// An empty set means the v3 no-authoring invariant holds.
fn authored_commits_in_merge(
    loader: &jj_lib::repo::RepoLoader,
    input_ops: &[jj_lib::operation::Operation],
    merged_op: &jj_lib::operation::Operation,
) -> HashSet<CommitId> {
    let all_input_commits: HashSet<CommitId> = input_ops
        .iter()
        .flat_map(|op| collect_visible_commits(loader, op))
        .collect();

    let merged_commits = collect_visible_commits(loader, merged_op);
    merged_commits
        .into_iter()
        .filter(|id| !all_input_commits.contains(id))
        .collect()
}

/// Strict: assert zero commits are authored by merge_operations.
/// Use for scenarios where the v3 no-authoring invariant must hold exactly.
fn assert_no_new_commits_authored(
    loader: &jj_lib::repo::RepoLoader,
    input_ops: &[jj_lib::operation::Operation],
    merged_op: &jj_lib::operation::Operation,
    label: &str,
) {
    let authored = authored_commits_in_merge(loader, input_ops, merged_op);
    assert!(
        authored.is_empty(),
        "[{label}] v3 no-authoring violated: merged result contains {} commit(s) not visible \
         in any input op: [{}]",
        authored.len(),
        authored
            .iter()
            .map(|id| id.hex()[..12.min(id.hex().len())].to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// Looser: assert that every commit authored by merge_operations has a
/// predecessor edge (in the op-store records) that chains back to a commit
/// that WAS visible in an input op. This allows pairwise-merge rebase to
/// author intermediate wc commits, but only if they are honest rewrites (not
/// orphans). Panics if any authored commit has no such chain.
fn assert_authored_commits_are_honest_rewrites(
    loader: &jj_lib::repo::RepoLoader,
    input_ops: &[jj_lib::operation::Operation],
    merged_op: &jj_lib::operation::Operation,
    label: &str,
) {
    let all_input_commits: HashSet<CommitId> = input_ops
        .iter()
        .flat_map(|op| collect_visible_commits(loader, op))
        .collect();

    let authored = authored_commits_in_merge(loader, input_ops, merged_op);
    if authored.is_empty() {
        return;
    }

    // Collect all predecessor edges from the merged op and its ancestry.
    let merged_repo = loader.load_at(merged_op).block_on().unwrap();
    let mut preds: HashMap<CommitId, Vec<CommitId>> = HashMap::new();
    let mut op_stack: Vec<jj_lib::operation::Operation> = vec![merged_op.clone()];
    let mut seen_ops: HashSet<jj_lib::op_store::OperationId> = HashSet::new();
    while let Some(op) = op_stack.pop() {
        if !seen_ops.insert(op.id().clone()) {
            continue;
        }
        if let Some(map) = &op.store_operation().commit_predecessors {
            for (new_id, old_ids) in map {
                preds
                    .entry(new_id.clone())
                    .or_default()
                    .extend(old_ids.iter().cloned());
            }
        }
        for parent in op.parents().block_on().unwrap() {
            op_stack.push(parent);
        }
    }
    drop(merged_repo);

    // For each authored commit, walk transitive predecessors to find a chain
    // back to a commit that was in an input op.
    for authored_id in &authored {
        let mut stack = vec![authored_id.clone()];
        let mut visited: HashSet<CommitId> = HashSet::new();
        let mut found_chain = false;
        while let Some(cur) = stack.pop() {
            if !visited.insert(cur.clone()) {
                continue;
            }
            if all_input_commits.contains(&cur) {
                found_chain = true;
                break;
            }
            if let Some(olds) = preds.get(&cur) {
                stack.extend(olds.iter().cloned());
            }
        }
        assert!(
            found_chain,
            "[{label}] authored commit {} has no predecessor chain back to any input-op \
             visible commit — orphan rewrite",
            &authored_id.hex()[..12.min(authored_id.hex().len())]
        );
    }
}

/// SCENARIO A: single agent, 3 in-private rewrites, no extra lineage.
/// (Control: does the plain fork -> rewrite x3 -> merge-back reconcile already
/// mint divergence?)
#[test]
fn repro_single_agent_three_rewrites() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path().to_path_buf();

    // 1. Orchestrator pre-creates slice change C1 on the SHARED store.
    let mut tx = repo.start_transaction();
    let c1 = create_random_commit(tx.repo_mut())
        .set_description("slice[0] original")
        .write()
        .block_on()
        .unwrap();
    let change_id = c1.change_id().clone();
    let shared_after_c1 = tx.commit("new empty commit").block_on().unwrap();

    // 2. Fork op heads by file copy (mimics hox exactly).
    brevity::fork_agent_oplog(
        &repo_path,
        "agent-0",
        shared_after_c1.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();

    // 3. Agent loop (PRIVATE store) rewrites C1 several times.
    let agent_loader =
        brevity::agent_repo_loader(shared_after_c1.loader(), &repo_path, "agent-0").unwrap();
    let agent_repo = agent_loader.load_at_head().block_on().unwrap();

    // describe #1
    let mut tx = agent_repo.start_transaction();
    let g2 = tx
        .repo_mut()
        .rewrite_commit(&c1)
        .set_description("slice[0] intermediate")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let agent_repo = tx.commit("describe commit C1").block_on().unwrap();

    // describe #2
    let mut tx = agent_repo.start_transaction();
    let g3 = tx
        .repo_mut()
        .rewrite_commit(&g2)
        .set_description("slice[0] intermediate 2")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let agent_repo = tx.commit("describe commit G2").block_on().unwrap();

    // squash-equivalent rewrite #3
    let mut tx = agent_repo.start_transaction();
    let _g4 = tx
        .repo_mut()
        .rewrite_commit(&g3)
        .set_description("[Slice 0] final")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    tx.commit("squash commits into G3").block_on().unwrap();

    // 5. Merge-back: reconcile divergent operations on the SHARED store.
    brevity::merge_agent_oplog(shared_after_c1.loader(), &repo_path, "agent-0")
        .block_on()
        .unwrap();

    // 6. Load shared at head; assert exactly ONE visible commit for C1.
    let reloaded = repo.loader().load_at_head().block_on().unwrap();
    dump_chain(&reloaded, "SCENARIO A (single agent, 3 rewrites)");
    let by_change = visible_commits_by_change(&reloaded);
    let visible = by_change.get(&change_id).map(|v| v.len()).unwrap_or(0);
    assert_eq!(
        visible, 1,
        "expected exactly 1 visible commit for C1's change_id, got {visible} (DIVERGENT)"
    );
}

/// SCENARIO B: agent rewrites + a THIRD lineage on the shared store holding an
/// INTERMEDIATE generation (mimics a stray/orphaned op whose head file lingers,
/// referencing a non-final generation of the same change).
///
/// This tests EVERY ordering of the 3 op heads fed to merge_operations, because
/// suspect #1 (rebase_descendants clearing parent_mapping between pairwise
/// merges) is ORDER-DEPENDENT.
///
/// v3 behavior: all three op heads have the old/intermediate/final generations
/// as actual VIEW HEADS (no wc child pinning), so v3 can remove the stale ones
/// via remove_head. No new commits are authored. All orderings must converge to
/// exactly 1 visible commit per change id.
#[test]
fn repro_third_lineage_all_orderings() {
    use jj_lib::operation::Operation;

    // Build the 3 ops: G1 (original C1), G2 (intermediate), G4 (final), all on
    // distinct op heads but the SAME linear rewrite chain (same change_id).
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path().to_path_buf();

    let mut tx = repo.start_transaction();
    let c1 = create_random_commit(tx.repo_mut())
        .set_description("slice[0] original")
        .write()
        .block_on()
        .unwrap();
    let change_id = c1.change_id().clone();
    let shared_after_c1 = tx.commit("new empty commit").block_on().unwrap();
    let op_g1 = shared_after_c1.operation().clone();

    brevity::fork_agent_oplog(
        &repo_path,
        "agent-0",
        shared_after_c1.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();

    let agent_loader =
        brevity::agent_repo_loader(shared_after_c1.loader(), &repo_path, "agent-0").unwrap();
    let agent_repo = agent_loader.load_at_head().block_on().unwrap();
    let mut tx = agent_repo.start_transaction();
    let g2 = tx
        .repo_mut()
        .rewrite_commit(&c1)
        .set_description("slice[0] intermediate")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let agent_repo_g2 = tx.commit("describe commit C1 -> G2").block_on().unwrap();
    let op_g2 = agent_repo_g2.operation().clone();

    let agent_repo = agent_repo_g2;
    let mut tx = agent_repo.start_transaction();
    let g3 = tx
        .repo_mut()
        .rewrite_commit(&g2)
        .set_description("slice[0] intermediate 2")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let agent_repo = tx.commit("describe commit G2 -> G3").block_on().unwrap();

    let mut tx = agent_repo.start_transaction();
    let _g4 = tx
        .repo_mut()
        .rewrite_commit(&g3)
        .set_description("[Slice 0] final")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let agent_repo_g4 = tx
        .commit("squash commits into G3 -> G4")
        .block_on()
        .unwrap();
    let op_g4 = agent_repo_g4.operation().clone();

    // Try all 6 orderings of [G1, G2, G4].
    let labels = [("G1", &op_g1), ("G2", &op_g2), ("G4", &op_g4)];
    let perms: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut failures = Vec::new();
    for perm in perms {
        let ops: Vec<Operation> = perm.iter().map(|&i| labels[i].1.clone()).collect();
        let order: Vec<&str> = perm.iter().map(|&i| labels[i].0).collect();
        let merged = repo
            .loader()
            .merge_operations(ops.clone(), Some("reconcile divergent operations"))
            .block_on()
            .unwrap();

        // v3 no-authoring invariant.
        assert_no_new_commits_authored(
            repo.loader(),
            &ops,
            &merged,
            &format!("SCENARIO B order {order:?}"),
        );

        let reloaded = repo.loader().load_at(&merged).block_on().unwrap();
        let by_change = visible_commits_by_change(&reloaded);
        let visible = by_change.get(&change_id).map(|v| v.len()).unwrap_or(0);
        eprintln!("order {order:?} -> {visible} visible commit(s) for C1");
        if visible != 1 {
            dump_chain(&reloaded, &format!("DIVERGENT order {order:?}"));
            failures.push((order, visible));
        }
    }
    assert!(
        failures.is_empty(),
        "orderings that minted divergence: {failures:?}"
    );
}

/// Focused single-order reproducer: feed [G1, G4, G2] (intermediate generation
/// LAST). This is the minimal failing case for clean instrumented tracing.
#[test]
fn repro_minimal_g1_g4_g2() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path().to_path_buf();

    let mut tx = repo.start_transaction();
    let c1 = create_random_commit(tx.repo_mut())
        .set_description("slice[0] original")
        .write()
        .block_on()
        .unwrap();
    let change_id = c1.change_id().clone();
    let shared_after_c1 = tx.commit("new empty commit").block_on().unwrap();
    let op_g1 = shared_after_c1.operation().clone();
    eprintln!(
        "G1 op={} C1 commit={}",
        &op_g1.id().hex()[..8],
        &c1.id().hex()[..8]
    );

    brevity::fork_agent_oplog(
        &repo_path,
        "agent-0",
        shared_after_c1.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();
    let agent_loader =
        brevity::agent_repo_loader(shared_after_c1.loader(), &repo_path, "agent-0").unwrap();
    let agent_repo = agent_loader.load_at_head().block_on().unwrap();

    let mut tx = agent_repo.start_transaction();
    let g2 = tx
        .repo_mut()
        .rewrite_commit(&c1)
        .set_description("slice[0] intermediate")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let agent_repo_g2 = tx.commit("describe commit C1 -> G2").block_on().unwrap();
    let op_g2 = agent_repo_g2.operation().clone();
    eprintln!(
        "G2 op={} G2 commit={}",
        &op_g2.id().hex()[..8],
        &g2.id().hex()[..8]
    );

    let agent_repo = agent_repo_g2;
    let mut tx = agent_repo.start_transaction();
    let g3 = tx
        .repo_mut()
        .rewrite_commit(&g2)
        .set_description("slice[0] intermediate 2")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let agent_repo = tx.commit("describe commit G2 -> G3").block_on().unwrap();
    eprintln!("G3 commit={}", &g3.id().hex()[..8]);

    let mut tx = agent_repo.start_transaction();
    let g4 = tx
        .repo_mut()
        .rewrite_commit(&g3)
        .set_description("[Slice 0] final")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let agent_repo_g4 = tx
        .commit("squash commits into G3 -> G4")
        .block_on()
        .unwrap();
    let op_g4 = agent_repo_g4.operation().clone();
    eprintln!(
        "G4 op={} G4 commit={}",
        &op_g4.id().hex()[..8],
        &g4.id().hex()[..8]
    );

    // Feed [G1, G4, G2] — intermediate G2 last.
    let ops = vec![op_g1, op_g4, op_g2];
    let merged = repo
        .loader()
        .merge_operations(ops.clone(), Some("reconcile divergent operations"))
        .block_on()
        .unwrap();

    // v3 no-authoring invariant.
    assert_no_new_commits_authored(repo.loader(), &ops, &merged, "MINIMAL [G1,G4,G2]");

    let reloaded = repo.loader().load_at(&merged).block_on().unwrap();
    dump_chain(&reloaded, "MINIMAL [G1,G4,G2]");
    let by_change = visible_commits_by_change(&reloaded);
    let visible = by_change.get(&change_id).map(|v| v.len()).unwrap_or(0);
    assert_eq!(visible, 1, "got {visible} visible commits (DIVERGENT)");
}

/// SCENARIO C (PRODUCTION SHAPE): the slice change has TWO generations, and
/// the SAME workspace working-copy commit (one change id) sits on each
/// generation in two different reconcile lineages.
///
/// v3.1 behavior:
///   1. The pairwise-merge rebase_descendants DOES author one new wc commit
///      (W1') — it rebases A's W1 (on original C1) onto G4 when it sees the
///      C1→G4 rewrite. This is an honest rewrite (predecessor chain: W1'→W1→W)
///      and is expected.
///   2. After pairwise merges, the wc change has two head siblings: W1' (authored
///      by the merge step) and W_b (carried by lineage B). Both are tree-identical
///      (empty snapshot on the same tree). Both share W as a common evolution
///      predecessor.
///   3. The v3.1 mechanical-sibling cleanup in dedup_evolved_heads collapses the
///      wc change to 1 visible (removes the lexicographically-smaller sibling).
///   4. Once the stale wc sibling is removed, C1 (its parent) becomes unreachable,
///      so the slice change also converges to 1 visible: G4.
///
/// Assertions:
///   (i)  Every authored commit has a predecessor chain to an input-visible commit.
///   (ii) Exactly 1 visible commit for the slice change (G4).
///   (iii) Exactly 1 visible commit for the wc change.
#[test]
fn repro_shared_wc_change_pins_both_gens() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path().to_path_buf();

    // 1. Pre-create slice change C1 (the original generation).
    let mut tx = repo.start_transaction();
    let c1 = create_random_commit(tx.repo_mut())
        .set_description("slice[0] gamma original")
        .write()
        .block_on()
        .unwrap();
    let slice_change = c1.change_id().clone();
    let shared_after_c1 = tx.commit("new empty commit").block_on().unwrap();

    // 2. Create the workspace wc commit W on top of the ORIGINAL gen, on the
    //    SHARED store, so both lineages fork from a state that already has W.
    let mut tx = shared_after_c1.start_transaction();
    let w = tx
        .repo_mut()
        .new_commit(vec![c1.id().clone()], c1.tree())
        .set_description("")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let shared_after_w = tx
        .commit("create initial working-copy commit in workspace loop-0")
        .block_on()
        .unwrap();
    let wc_change = w.change_id().clone();
    let op_base = shared_after_w.operation().clone();

    // 3. LINEAGE A: leaves the workspace where it is (wc W on original gen).
    brevity::fork_agent_oplog(
        &repo_path,
        "agentA",
        shared_after_w.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();
    let loader_a =
        brevity::agent_repo_loader(shared_after_w.loader(), &repo_path, "agentA").unwrap();
    let repo_a = loader_a.load_at_head().block_on().unwrap();
    let mut tx = repo_a.start_transaction();
    let w1 = tx
        .repo_mut()
        .rewrite_commit(&w)
        .set_description("")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let op_lineage_a = tx
        .commit("snapshot working copy")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    // 4. LINEAGE B: rewrites the slice change C1 -> G4 (final squash) and
    //    carries the workspace wc commit onto the new generation each time.
    brevity::fork_agent_oplog(
        &repo_path,
        "agentB",
        shared_after_w.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();
    let loader_b =
        brevity::agent_repo_loader(shared_after_w.loader(), &repo_path, "agentB").unwrap();
    let repo_b = loader_b.load_at_head().block_on().unwrap();

    // describe C1 -> G2, reparent wc W onto G2
    let mut tx = repo_b.start_transaction();
    let g2 = tx
        .repo_mut()
        .rewrite_commit(&c1)
        .set_description("slice[0] gamma intermediate")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let repo_b = tx.commit("describe commit C1 -> G2").block_on().unwrap();

    // describe G2 -> G3
    let g2 = repo_b.store().get_commit(g2.id()).unwrap();
    let mut tx = repo_b.start_transaction();
    let g3 = tx
        .repo_mut()
        .rewrite_commit(&g2)
        .set_description("slice[0] gamma intermediate 2")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let repo_b = tx.commit("describe commit G2 -> G3").block_on().unwrap();

    // squash G3 -> G4
    let g3 = repo_b.store().get_commit(g3.id()).unwrap();
    let mut tx = repo_b.start_transaction();
    let g4 = tx
        .repo_mut()
        .rewrite_commit(&g3)
        .set_description("[Slice 0] gamma final")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let op_lineage_b = tx
        .commit("squash commits into G3 -> G4")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    eprintln!(
        "slice original={} squash={} ; wc_snaphot={}",
        &c1.id().hex()[..8],
        &g4.id().hex()[..8],
        &w1.id().hex()[..8]
    );

    // 5. Reconcile baseline + lineage A (original gen, wc snapshot) + lineage B
    //    (squash gen, wc carried). The squash lineage is LAST.
    let ops = vec![op_base, op_lineage_a, op_lineage_b];
    let merged = repo
        .loader()
        .merge_operations(ops.clone(), Some("reconcile divergent operations"))
        .block_on()
        .unwrap();

    // v3.1 contract for SCENARIO C:
    //
    // (i)  Every commit authored by merge_operations (pairwise-merge rebase_descendants
    //      authoring W1') must have a predecessor chain back to an input-visible commit.
    //      This is LOOSER than zero-authoring: the pairwise-merge rebase IS allowed to
    //      author W1' (an honest rewrite of W1) but must not author any orphan commits.
    assert_authored_commits_are_honest_rewrites(repo.loader(), &ops, &merged, "SCENARIO C");

    let reloaded = repo.loader().load_at(&merged).block_on().unwrap();
    dump_chain(&reloaded, "SCENARIO C (shared wc change pins both gens)");

    let by_change = visible_commits_by_change(&reloaded);
    let slice_visible = by_change.get(&slice_change).map(|v| v.len()).unwrap_or(0);
    let wc_visible = by_change.get(&wc_change).map(|v| v.len()).unwrap_or(0);
    eprintln!("SCENARIO C: slice_visible={slice_visible} wc_visible={wc_visible}");

    // (ii) v3.1 mechanical-sibling cleanup collapses the wc siblings (W1' and W_b)
    //      to exactly 1 visible wc commit. They are tree-identical (both empty
    //      snapshots on the same tree), share W as a common predecessor, are both
    //      heads (not wc-referenced in the test repo), and pass the exclusive-ancestor
    //      fixpoint — so the lex-smaller sibling is removed.
    assert_eq!(
        wc_visible, 1,
        "SCENARIO C: wc change must converge to exactly 1 visible commit after v3.1 cleanup; \
         got {wc_visible}"
    );

    // (iii) Once the stale wc sibling is removed, C1 (parent of the removed sibling)
    //       becomes unreachable from remaining heads, so the slice change converges
    //       to exactly 1 visible commit: G4.
    assert_eq!(
        slice_visible, 1,
        "SCENARIO C: slice change must converge to exactly 1 visible commit after v3.1 cleanup; \
         got {slice_visible}"
    );
}

/// SCENARIO D (DIRECT, isolates the dedup from the merge path): build a view
/// in which the slice change is ALREADY divergent with the OLD generation kept
/// visible-but-not-a-head by a wc-commit child, then invoke the public
/// `dedup_evolved_heads` and assert v3 fail-open behavior.
///
/// v3 change from v2: v2 called `set_rewritten_commit(C1, G4)` and then
/// `rebase_descendants` authored a new wc commit on top of the survivor.
/// v3 does NOT author new commits. Since C1 is a non-head pinned by W1 (a
/// non-stale visible descendant), dedup fails open: nothing is hidden, nothing
/// is authored, no error is raised.
///
/// The no-authoring invariant holds by construction (we never set_rewritten).
#[test]
fn repro_direct_nonhead_old_gen_divergence() {
    use jj_lib::repo::MutableRepo;

    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;

    // Establish C1 (original gen) and capture the op (predecessor source).
    let mut tx = repo.start_transaction();
    let c1 = create_random_commit(tx.repo_mut())
        .set_description("slice[0] original")
        .write()
        .block_on()
        .unwrap();
    let slice_change = c1.change_id().clone();
    let repo1 = tx.commit("new empty commit").block_on().unwrap();
    let op_c1 = repo1.operation().clone();

    // Rewrite C1 -> G4 (records the predecessor edge) and capture the op.
    let mut tx = repo1.start_transaction();
    let g4 = tx
        .repo_mut()
        .rewrite_commit(&c1)
        .set_description("[Slice 0] final")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let repo_g4 = tx.commit("squash commit C1 -> G4").block_on().unwrap();
    let op_g4 = repo_g4.operation().clone();

    // Now hand-build the DIVERGENT view: both C1 and G4 visible, each pinned by
    // a wc-commit child, neither generation a head. This is the post-reconcile
    // state that the merge path can land in for real cascades.
    let mut tx = repo_g4.start_transaction();
    let mut_repo: &mut MutableRepo = tx.repo_mut();
    // wc child on the OLD gen C1 -> keeps C1 visible-but-not-head.
    let _w1 = mut_repo
        .new_commit(vec![c1.id().clone()], c1.tree())
        .set_description("wc commit (workspace loop-old)")
        .write()
        .block_on()
        .unwrap();
    // wc child on the NEW gen G4 -> keeps G4 visible-but-not-head.
    let _w2 = mut_repo
        .new_commit(vec![g4.id().clone()], g4.tree())
        .set_description("wc commit (workspace loop-new)")
        .write()
        .block_on()
        .unwrap();
    mut_repo.rebase_descendants().block_on().unwrap();

    // Sanity: the slice change is divergent right now (2 visible gens, both non-head).
    {
        let pre_view = tx.repo().view().clone();
        let mut stack: Vec<CommitId> = pre_view.heads().iter().cloned().collect();
        let mut seen = std::collections::HashSet::new();
        let mut slice_vis = 0;
        while let Some(id) = stack.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            let c = tx.repo().store().get_commit(&id).unwrap();
            if *c.change_id() == slice_change {
                slice_vis += 1;
            }
            for pid in c.parent_ids() {
                stack.push(pid.clone());
            }
        }
        eprintln!("PRE-dedup slice visible generations = {slice_vis} (expect 2)");
        assert_eq!(
            slice_vis, 2,
            "test setup failed: slice should be divergent before dedup"
        );
    }

    // Capture the set of all commit ids before dedup — the no-authoring check.
    let pre_dedup_commits: HashSet<CommitId> = {
        let mut s = HashSet::new();
        let mut stack: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
        let mut visited: HashSet<CommitId> = HashSet::new();
        while let Some(id) = stack.pop() {
            if !visited.insert(id.clone()) {
                continue;
            }
            s.insert(id.clone());
            if let Ok(c) = tx.repo().store().get_commit(&id) {
                for pid in c.parent_ids() {
                    stack.push(pid.clone());
                }
            }
        }
        s
    };

    // Invoke the dedup with the ops that carry the C1 -> G4 predecessor edge.
    // v3 signature requires the debug_log parameter (None = silent).
    tx.repo_mut()
        .dedup_evolved_heads(&[op_c1, op_g4], None)
        .block_on()
        .unwrap();
    let rebased = tx.repo_mut().rebase_descendants().block_on().unwrap();

    // v3 no-authoring: rebase_descendants must be a no-op after dedup.
    assert_eq!(
        rebased, 0,
        "v3 violation: rebase_descendants authored {rebased} commit(s) after dedup"
    );

    // Verify no new commits were created.
    let post_dedup_commits: HashSet<CommitId> = {
        let mut s = HashSet::new();
        let mut stack: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
        let mut visited: HashSet<CommitId> = HashSet::new();
        while let Some(id) = stack.pop() {
            if !visited.insert(id.clone()) {
                continue;
            }
            s.insert(id.clone());
            if let Ok(c) = tx.repo().store().get_commit(&id) {
                for pid in c.parent_ids() {
                    stack.push(pid.clone());
                }
            }
        }
        s
    };
    for id in &post_dedup_commits {
        assert!(
            pre_dedup_commits.contains(id),
            "v3 violation: dedup authored new commit {}",
            &id.hex()[..12.min(id.hex().len())]
        );
    }

    // v3 fail-open: C1 is pinned by W1 (a non-stale wc commit with different
    // change_id). Dedup cannot remove C1 without removing W1. The divergence
    // remains. This is intentional — the old behavior (rebasing W1) was the
    // PRIMARY source of sibling-divergence bugs.
    let mut stack: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
    let mut seen = std::collections::HashSet::new();
    let mut slice_vis = 0;
    while let Some(id) = stack.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        let c = tx.repo().store().get_commit(&id).unwrap();
        if *c.change_id() == slice_change {
            slice_vis += 1;
            eprintln!(
                "  POST visible slice gen: {} {:?}",
                &id.hex()[..8],
                c.description()
            );
        }
        for pid in c.parent_ids() {
            stack.push(pid.clone());
        }
    }
    eprintln!("POST-dedup slice visible generations = {slice_vis}");
    // v3: fail-open means the divergence persists (2 generations remain visible).
    // No new commits were authored — that is the key invariant.
    assert_eq!(
        slice_vis, 2,
        "v3 expected 2 visible slice generations (fail-open on wc-pinned stale gen), got \
         {slice_vis}"
    );
}

// =============================================================================
// NEW TESTS (v3-specific)
// =============================================================================

/// T1 CASCADE (PRIMARY MECHANISM): tests the exact R1/R2 shape that caused
/// rebase-minted sibling divergence in v2.
///
/// Shape:
///   - Reconcile R1: Op-head L1 has C-new (head), Op-head L2 has C-old (head)
///     with wc child WC-old (head). After pairwise merge: C-new is head, WC-old
///     still pins C-old visible. Dedup (v3) removes C-new from... wait, actually
///     in this simpler shape: C-old and C-new are both heads.
///
/// Simpler T1 shape: Build the interleaved-reconcile situation.
///   - Shared base has C-gen0 (the original).
///   - Agent A advances: C-gen0 -> C-gen1 -> C-gen2 (final).
///   - Shared store also has a stray op with C-gen1 as the head.
///   - Merge [C-gen0, C-gen2, C-gen1]: C-gen0 and C-gen1 are stale heads.
///     v3 removes them via remove_head. No new commits. 1 visible.
///   - Then merge [M1, L2] where L2 is the pre-R1 lineage still at C-gen0.
///     v3 must also converge to 1 visible without authoring.
#[test]
fn t1_cascade_primary_mechanism() {
    use jj_lib::operation::Operation;

    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path().to_path_buf();

    // Set up: shared base has C-gen0.
    let mut tx = repo.start_transaction();
    let c_gen0 = create_random_commit(tx.repo_mut())
        .set_description("c-gen0")
        .write()
        .block_on()
        .unwrap();
    let change_id = c_gen0.change_id().clone();
    let shared_base = tx.commit("initial: c-gen0").block_on().unwrap();
    let op_gen0 = shared_base.operation().clone();

    // Agent forks and produces C-gen0 -> C-gen1 -> C-gen2.
    brevity::fork_agent_oplog(&repo_path, "agent-a", shared_base.op_heads_store().as_ref())
        .block_on()
        .unwrap();
    let agent_loader =
        brevity::agent_repo_loader(shared_base.loader(), &repo_path, "agent-a").unwrap();
    let agent_repo = agent_loader.load_at_head().block_on().unwrap();

    let mut tx = agent_repo.start_transaction();
    let c_gen1 = tx
        .repo_mut()
        .rewrite_commit(&c_gen0)
        .set_description("c-gen1")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let agent_repo = tx.commit("rewrite gen0 -> gen1").block_on().unwrap();
    let op_gen1 = agent_repo.operation().clone();

    let c_gen1 = agent_repo.store().get_commit(c_gen1.id()).unwrap();
    let mut tx = agent_repo.start_transaction();
    let c_gen2 = tx
        .repo_mut()
        .rewrite_commit(&c_gen1)
        .set_description("c-gen2 final")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let agent_repo = tx.commit("rewrite gen1 -> gen2").block_on().unwrap();
    let op_gen2 = agent_repo.operation().clone();

    // M1 = merge of [op_gen0, op_gen2, op_gen1]: gen0 and gen1 are stale heads.
    // v3 must: remove gen0 and gen1 from heads, leave gen2. No new commits.
    let m1_ops: Vec<Operation> = vec![op_gen0.clone(), op_gen2.clone(), op_gen1.clone()];
    let m1 = repo
        .loader()
        .merge_operations(m1_ops.clone(), Some("reconcile R1"))
        .block_on()
        .unwrap();

    assert_no_new_commits_authored(repo.loader(), &m1_ops, &m1, "T1 R1");
    {
        let r1_repo = repo.loader().load_at(&m1).block_on().unwrap();
        let by_change = visible_commits_by_change(&r1_repo);
        let visible = by_change.get(&change_id).map(|v| v.len()).unwrap_or(0);
        assert_eq!(
            visible, 1,
            "T1 R1: expected 1 visible after M1, got {visible}"
        );
        // Verify the survivor is gen2.
        if let Some(ids) = by_change.get(&change_id) {
            let commit = r1_repo.store().get_commit(&ids[0]).unwrap();
            assert_eq!(
                commit.description(),
                "c-gen2 final",
                "T1 R1: survivor should be gen2"
            );
        }
    }

    // Now simulate L2: a concurrent op lineage forked from op_gen0 that still
    // carries the ORIGINAL c_gen0. Merge [M1, op_gen0].
    let m2_ops: Vec<Operation> = vec![m1.clone(), op_gen0.clone()];
    let m2 = repo
        .loader()
        .merge_operations(m2_ops.clone(), Some("reconcile R2"))
        .block_on()
        .unwrap();

    assert_no_new_commits_authored(repo.loader(), &m2_ops, &m2, "T1 R2");
    {
        let r2_repo = repo.loader().load_at(&m2).block_on().unwrap();
        dump_chain(&r2_repo, "T1 R2 (M1 + L2/gen0)");
        let by_change = visible_commits_by_change(&r2_repo);
        let visible = by_change.get(&change_id).map(|v| v.len()).unwrap_or(0);
        assert_eq!(
            visible, 1,
            "T1 R2: expected 1 visible after M2, got {visible} (sibling divergence)"
        );
    }

    drop(c_gen2);
}

/// T2 STACKED DUAL DIVERGENCE: lineage A has P-old←C-old (P is the parent
/// change, C is the child change). Lineage B has P-new←C-new with predecessor
/// edges P-old→P-new and C-old→C-new. After merge, both the parent change and
/// the child change should have exactly 1 visible each, and no new commits are
/// authored.
#[test]
fn t2_stacked_dual_divergence() {
    use jj_lib::operation::Operation;

    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path().to_path_buf();

    // Build shared base with P-old and C-old (C-old is a child commit of P-old).
    let mut tx = repo.start_transaction();
    let p_old = create_random_commit(tx.repo_mut())
        .set_description("P-old")
        .write()
        .block_on()
        .unwrap();
    let p_change = p_old.change_id().clone();
    let c_old = tx
        .repo_mut()
        .new_commit(vec![p_old.id().clone()], p_old.tree())
        .set_description("C-old")
        .write()
        .block_on()
        .unwrap();
    let c_change = c_old.change_id().clone();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let shared_base = tx.commit("initial: P-old + C-old").block_on().unwrap();
    let op_lineage_a = shared_base.operation().clone();

    // Lineage B: rewrites P-old -> P-new, then C-old -> C-new.
    brevity::fork_agent_oplog(&repo_path, "agent-b", shared_base.op_heads_store().as_ref())
        .block_on()
        .unwrap();
    let loader_b = brevity::agent_repo_loader(shared_base.loader(), &repo_path, "agent-b").unwrap();
    let repo_b = loader_b.load_at_head().block_on().unwrap();

    let mut tx = repo_b.start_transaction();
    let p_new = tx
        .repo_mut()
        .rewrite_commit(&p_old)
        .set_description("P-new")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let repo_b = tx.commit("rewrite P-old -> P-new").block_on().unwrap();

    // After rebasing descendants, C-old should have been rebased onto P-new.
    // Find the new C commit (same change_id).
    let by_change = visible_commits_by_change(&repo_b);
    let c_rebased_ids = by_change.get(&c_change).cloned().unwrap_or_default();
    assert_eq!(
        c_rebased_ids.len(),
        1,
        "expected exactly 1 C commit after P rewrite"
    );
    let c_rebased = repo_b.store().get_commit(&c_rebased_ids[0]).unwrap();

    // Now explicitly rewrite C -> C-new to record a predecessor edge.
    let mut tx = repo_b.start_transaction();
    let _c_new = tx
        .repo_mut()
        .rewrite_commit(&c_rebased)
        .set_description("C-new")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let repo_b = tx.commit("rewrite C-rebased -> C-new").block_on().unwrap();
    let op_lineage_b = repo_b.operation().clone();

    eprintln!(
        "p_change={} c_change={}",
        &p_change.reverse_hex()[..12],
        &c_change.reverse_hex()[..12]
    );

    // Merge lineage A (P-old + C-old heads) and lineage B (P-new + C-new heads).
    let ops: Vec<Operation> = vec![op_lineage_a, op_lineage_b];
    let merged = repo
        .loader()
        .merge_operations(ops.clone(), Some("reconcile stacked dual divergence"))
        .block_on()
        .unwrap();

    assert_no_new_commits_authored(repo.loader(), &ops, &merged, "T2");

    let merged_repo = repo.loader().load_at(&merged).block_on().unwrap();
    dump_chain(&merged_repo, "T2 STACKED DUAL DIVERGENCE");
    let by_change = visible_commits_by_change(&merged_repo);

    let p_visible = by_change.get(&p_change).map(|v| v.len()).unwrap_or(0);
    let c_visible = by_change.get(&c_change).map(|v| v.len()).unwrap_or(0);

    assert_eq!(
        p_visible, 1,
        "T2: parent change P should have 1 visible, got {p_visible}"
    );
    assert_eq!(
        c_visible, 1,
        "T2: child change C should have 1 visible, got {c_visible}"
    );

    // Verify survivors are the new gens.
    if let Some(ids) = by_change.get(&p_change) {
        let commit = merged_repo.store().get_commit(&ids[0]).unwrap();
        assert_eq!(
            commit.description(),
            "P-new",
            "T2: P survivor should be P-new"
        );
    }
    if let Some(ids) = by_change.get(&c_change) {
        let commit = merged_repo.store().get_commit(&ids[0]).unwrap();
        assert_eq!(
            commit.description(),
            "C-new",
            "T2: C survivor should be C-new"
        );
    }

    drop(p_new);
}

/// T3 HEAD-INVERSION + DEEP HARVEST (secondary mechanism): the edge-bearing
/// rewrite op sits at or below the CCA of the merged ops. Phase-1 harvest
/// (CCA-bounded) misses the edge; phase-2 (unbounded) must find it.
///
/// Shape:
///   - Shared history: X-orig written, then X-new written (rewrite X-orig->X-new,
///     records the predecessor edge). This forms the "deep" op that is the CCA.
///   - Two lineages fork from X-new. L1 touches something else (no X-related
///     edges), while L2 re-introduces X-orig as a bare head (stray op).
///   - Merge [L1, L2]: X-orig and X-new are both visible. The predecessor edge
///     X-orig->X-new was recorded in an op that is at/below the CCA. Phase-1
///     misses it; phase-2 finds it. X-orig removed. 1 visible.
#[test]
fn t3_head_inversion_deep_harvest() {
    use jj_lib::operation::Operation;

    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path().to_path_buf();

    // Build shared base: X-orig -> X-new (rewrite, records predecessor edge).
    let mut tx = repo.start_transaction();
    let x_orig = create_random_commit(tx.repo_mut())
        .set_description("X-orig")
        .write()
        .block_on()
        .unwrap();
    let x_change = x_orig.change_id().clone();
    let repo_after_x_orig = tx.commit("X-orig").block_on().unwrap();

    let mut tx = repo_after_x_orig.start_transaction();
    let x_new = tx
        .repo_mut()
        .rewrite_commit(&x_orig)
        .set_description("X-new")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    // This op records the X-orig -> X-new predecessor edge.
    let shared_after_x_new = tx.commit("rewrite X-orig -> X-new").block_on().unwrap();
    // op_rewrite is the CCA of L1 and L2 below.

    // L1 fork: touches something else, no X edges.
    brevity::fork_agent_oplog(
        &repo_path,
        "l1",
        shared_after_x_new.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();
    let loader_l1 =
        brevity::agent_repo_loader(shared_after_x_new.loader(), &repo_path, "l1").unwrap();
    let repo_l1 = loader_l1.load_at_head().block_on().unwrap();
    let mut tx = repo_l1.start_transaction();
    let _other = create_random_commit(tx.repo_mut())
        .set_description("other change L1")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let op_l1 = tx
        .commit("L1: add other change")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    // L2 fork: re-introduces X-orig as a bare head by adding it back.
    brevity::fork_agent_oplog(
        &repo_path,
        "l2",
        shared_after_x_new.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();
    let loader_l2 =
        brevity::agent_repo_loader(shared_after_x_new.loader(), &repo_path, "l2").unwrap();
    let repo_l2 = loader_l2.load_at_head().block_on().unwrap();
    // Re-add X-orig as a head by using add_head. We do this via a transaction
    // that directly manipulates the view.
    let mut tx = repo_l2.start_transaction();
    // Get the actual x_orig commit from the store.
    let x_orig_from_store = tx.repo_mut().store().get_commit(x_orig.id()).unwrap();
    tx.repo_mut()
        .add_head(&x_orig_from_store)
        .block_on()
        .unwrap();
    let op_l2 = tx
        .commit("L2: re-add X-orig as stray head")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    eprintln!(
        "x_change={} x_orig={} x_new={}",
        &x_change.reverse_hex()[..12],
        &x_orig.id().hex()[..8],
        &x_new.id().hex()[..8]
    );

    // Merge L1 and L2. The predecessor edge X-orig->X-new is in an op that is
    // BELOW the CCA of [op_l1, op_l2] (the CCA is shared_after_x_new's op).
    // Phase-1 harvest (bounded to the merge cone) misses the edge.
    // Phase-2 harvest (unbounded) must find it and resolve.
    let ops: Vec<Operation> = vec![op_l1, op_l2];
    let merged = repo
        .loader()
        .merge_operations(ops.clone(), Some("reconcile head-inversion"))
        .block_on()
        .unwrap();

    assert_no_new_commits_authored(repo.loader(), &ops, &merged, "T3");

    let merged_repo = repo.loader().load_at(&merged).block_on().unwrap();
    dump_chain(&merged_repo, "T3 HEAD-INVERSION + DEEP HARVEST");
    let by_change = visible_commits_by_change(&merged_repo);

    let x_visible = by_change.get(&x_change).map(|v| v.len()).unwrap_or(0);
    assert_eq!(
        x_visible, 1,
        "T3: X change should have 1 visible after phase-2 harvest, got {x_visible}"
    );
    if let Some(ids) = by_change.get(&x_change) {
        let commit = merged_repo.store().get_commit(&ids[0]).unwrap();
        assert_eq!(
            commit.description(),
            "X-new",
            "T3: survivor should be X-new (the newer generation)"
        );
    }
}

/// T4 PINNED FAIL-OPEN: stale generation with a non-stale visible child (the
/// child has no successor anywhere). v3 must leave the group alone — nothing
/// hidden, nothing authored, no error.
#[test]
fn t4_pinned_fail_open() {
    use jj_lib::repo::MutableRepo;

    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;

    // Build: A-old (original), A-new (rewrite, records edge), then A-old has
    // a non-stale child CHILD (different change_id) visible as a head.
    let mut tx = repo.start_transaction();
    let a_old = create_random_commit(tx.repo_mut())
        .set_description("A-old")
        .write()
        .block_on()
        .unwrap();
    let a_change = a_old.change_id().clone();
    let repo_after_a_old = tx.commit("A-old").block_on().unwrap();
    let op_a_old = repo_after_a_old.operation().clone();

    let mut tx = repo_after_a_old.start_transaction();
    let _a_new = tx
        .repo_mut()
        .rewrite_commit(&a_old)
        .set_description("A-new")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let repo_after_a_new = tx.commit("rewrite A-old -> A-new").block_on().unwrap();
    let op_a_new = repo_after_a_new.operation().clone();

    // Hand-build: A-old is visible-but-not-head (has non-stale child CHILD).
    // A-new is also visible.
    let mut tx = repo_after_a_new.start_transaction();
    let mut_repo: &mut MutableRepo = tx.repo_mut();
    // Add CHILD on top of A-old — a non-stale descendant that pins A-old.
    let _child = mut_repo
        .new_commit(vec![a_old.id().clone()], a_old.tree())
        .set_description("CHILD (pins A-old, non-stale)")
        .write()
        .block_on()
        .unwrap();
    mut_repo.rebase_descendants().block_on().unwrap();

    // Pre-dedup: A-old and A-new should both be visible.
    let a_old_id = a_old.id().clone();
    let pre_a_vis: usize = {
        let heads: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
        let mut stack = heads;
        let mut seen: HashSet<CommitId> = HashSet::new();
        let mut count = 0;
        while let Some(id) = stack.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Ok(c) = tx.repo().store().get_commit(&id) {
                if *c.change_id() == a_change {
                    count += 1;
                }
                for pid in c.parent_ids() {
                    stack.push(pid.clone());
                }
            }
        }
        count
    };
    eprintln!("T4 PRE-dedup A visible = {pre_a_vis}");

    // Invoke dedup.
    tx.repo_mut()
        .dedup_evolved_heads(&[op_a_old, op_a_new], None)
        .block_on()
        .unwrap();
    let rebased = tx.repo_mut().rebase_descendants().block_on().unwrap();

    // v3 no-authoring invariant: rebase_descendants must be a no-op.
    assert_eq!(
        rebased, 0,
        "T4: v3 violation: rebase authored {rebased} commit(s)"
    );

    // Verify: A-old is still pinned (CHILD is non-stale), so both A-old and A-new
    // should remain visible (fail-open). Nothing was hidden.
    let a_post_vis: usize = {
        let heads: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
        let mut stack = heads;
        let mut seen: HashSet<CommitId> = HashSet::new();
        let mut count = 0;
        while let Some(id) = stack.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Ok(c) = tx.repo().store().get_commit(&id) {
                if *c.change_id() == a_change {
                    count += 1;
                }
                for pid in c.parent_ids() {
                    stack.push(pid.clone());
                }
            }
        }
        count
    };
    eprintln!("T4 POST-dedup A visible = {a_post_vis}");

    // The A-old commit must still be visible (pinned fail-open).
    let is_a_old_still_visible = {
        let heads: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
        let mut stack = heads;
        let mut seen: HashSet<CommitId> = HashSet::new();
        let mut found = false;
        while let Some(id) = stack.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            if id == a_old_id {
                found = true;
                break;
            }
            if let Ok(c) = tx.repo().store().get_commit(&id) {
                for pid in c.parent_ids() {
                    stack.push(pid.clone());
                }
            }
        }
        found
    };
    assert!(
        is_a_old_still_visible,
        "T4: A-old should remain visible (fail-open: pinned by non-stale CHILD)"
    );
    assert_eq!(
        a_post_vis, 2,
        "T4: both A-old and A-new should remain visible (fail-open), got {a_post_vis}"
    );
}

/// T5 WC-COMMIT PROTECTION: the stale head IS a workspace's wc commit in the
/// merged view. v3 must NOT remove it even though it is stale.
///
/// This tests the (b) guard in the removal logic: wc commits are never removed
/// even if they are identified as stale.
#[test]
fn t5_wc_commit_protection() {
    use jj_lib::repo::MutableRepo;

    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;

    let mut tx = repo.start_transaction();
    let b_old = create_random_commit(tx.repo_mut())
        .set_description("B-old")
        .write()
        .block_on()
        .unwrap();
    let b_change = b_old.change_id().clone();
    let repo_after_b_old = tx.commit("B-old").block_on().unwrap();
    let op_b_old = repo_after_b_old.operation().clone();

    let mut tx = repo_after_b_old.start_transaction();
    let b_new = tx
        .repo_mut()
        .rewrite_commit(&b_old)
        .set_description("B-new")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let repo_after_b_new = tx.commit("rewrite B-old -> B-new").block_on().unwrap();
    let op_b_new = repo_after_b_new.operation().clone();

    // Hand-build: B-old is BOTH a view head AND a wc commit for workspace "main".
    // B-new is also a view head. Both are divergent.
    let mut tx = repo_after_b_new.start_transaction();
    let mut_repo: &mut MutableRepo = tx.repo_mut();

    // Ensure B-old is in the view heads (it may have been removed by the
    // rewrite+rebase above; we explicitly re-add it to simulate a stale op
    // that brought it back).
    let b_old_commit = mut_repo.store().get_commit(b_old.id()).unwrap();
    mut_repo.add_head(&b_old_commit).block_on().unwrap();

    // Set B-old as the workspace wc commit so the protection guard fires.
    mut_repo
        .set_wc_commit(
            jj_lib::ref_name::WorkspaceNameBuf::from("test-workspace"),
            b_old.id().clone(),
        )
        .unwrap();
    mut_repo.rebase_descendants().block_on().unwrap();

    // Pre-dedup: both B-old and B-new should be visible.
    let pre_b_vis = {
        let heads: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
        let mut stack = heads;
        let mut seen: HashSet<CommitId> = HashSet::new();
        let mut count = 0;
        while let Some(id) = stack.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Ok(c) = tx.repo().store().get_commit(&id) {
                if *c.change_id() == b_change {
                    count += 1;
                }
                for pid in c.parent_ids() {
                    stack.push(pid.clone());
                }
            }
        }
        count
    };
    eprintln!("T5 PRE-dedup B visible = {pre_b_vis}");

    // Invoke dedup.
    tx.repo_mut()
        .dedup_evolved_heads(&[op_b_old, op_b_new], None)
        .block_on()
        .unwrap();
    let rebased = tx.repo_mut().rebase_descendants().block_on().unwrap();

    // v3 no-authoring invariant.
    assert_eq!(
        rebased, 0,
        "T5: v3 violation: rebase authored {rebased} commit(s)"
    );

    // B-old must still be a view head (wc-commit protection).
    let b_old_still_head = tx.repo().view().heads().contains(b_old.id());
    assert!(
        b_old_still_head,
        "T5: B-old (wc commit) must not be removed from view heads"
    );

    // B-old must still be the wc commit.
    let wc_id = tx
        .repo()
        .view()
        .get_wc_commit_id(jj_lib::ref_name::WorkspaceName::new("test-workspace"));
    assert_eq!(
        wc_id,
        Some(b_old.id()),
        "T5: B-old must still be the wc commit after dedup"
    );

    eprintln!("T5: wc-commit protection verified — B-old not removed");
    drop(b_new);
}

/// T6 MECHANICAL-SIBLING DIVERGENCE WITH NON-IDENTICAL TREES (FAIL-OPEN).
///
/// Builds the same sibling shape (two view heads with the same change_id,
/// neither wc-referenced) but gives the two siblings DIFFERENT trees using
/// real file content. Guard (b) fires: `member.tree_ids() != survivor.tree_ids()`
/// → dedup must FAIL OPEN (both siblings remain visible, zero extra authoring).
///
/// This verifies that v3.1 never silently destroys content.
#[test]
fn t6_mechanical_sibling_different_trees_fail_open() {
    use jj_lib::repo::MutableRepo;

    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;

    let path_a = repo_path("file_a.txt");
    let path_b = repo_path("file_b.txt");

    // Create two trees with genuinely different content.
    let tree_a = create_tree(repo, &[(path_a, "content from lineage A\n")]);
    let tree_b = create_tree(repo, &[(path_b, "content from lineage B\n")]);

    // Common root commit R (empty tree — predecessor origin for both siblings).
    let mut tx = repo.start_transaction();
    let root = create_random_commit(tx.repo_mut())
        .set_description("root commit — common predecessor")
        .write()
        .block_on()
        .unwrap();
    let wc_change = root.change_id().clone();
    let repo_after_root = tx.commit("root").block_on().unwrap();
    let op_root = repo_after_root.operation().clone();

    // Sibling A: rewrite root with tree_a content.
    let mut tx = repo_after_root.start_transaction();
    let sibling_a = tx
        .repo_mut()
        .rewrite_commit(&root)
        .set_description("sibling A")
        .set_tree(tree_a)
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let repo_sa = tx.commit("rewrite root -> sibling-A").block_on().unwrap();
    let op_a = repo_sa.operation().clone();

    // Sibling B: rewrite root with tree_b content (genuinely different).
    let mut tx = repo_after_root.start_transaction();
    let sibling_b = tx
        .repo_mut()
        .rewrite_commit(&root)
        .set_description("sibling B")
        .set_tree(tree_b)
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let repo_sb = tx.commit("rewrite root -> sibling-B").block_on().unwrap();
    let op_b = repo_sb.operation().clone();

    // Force-build divergent view: both sibling_a and sibling_b are view heads.
    let mut tx = repo_sb.start_transaction();
    let mut_repo: &mut MutableRepo = tx.repo_mut();
    let sa_commit = mut_repo.store().get_commit(sibling_a.id()).unwrap();
    mut_repo.add_head(&sa_commit).block_on().unwrap();
    mut_repo.rebase_descendants().block_on().unwrap();

    // Confirm the trees differ at the point of dedup.
    let sa_reloaded = tx.repo().store().get_commit(sibling_a.id()).unwrap();
    let sb_reloaded = tx.repo().store().get_commit(sibling_b.id()).unwrap();
    assert_ne!(
        sa_reloaded.tree_ids(),
        sb_reloaded.tree_ids(),
        "T6 setup error: siblings must have different tree_ids after rewrite"
    );

    // Pre-dedup: both siblings visible.
    let pre_vis = {
        let mut heads: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
        let mut seen: HashSet<CommitId> = HashSet::new();
        let mut count = 0;
        while let Some(id) = heads.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Ok(c) = tx.repo().store().get_commit(&id) {
                if *c.change_id() == wc_change {
                    count += 1;
                }
                for pid in c.parent_ids() {
                    heads.push(pid.clone());
                }
            }
        }
        count
    };
    eprintln!("T6 PRE-dedup wc_change visible = {pre_vis}");
    assert!(
        pre_vis >= 2,
        "T6 setup: both siblings must be visible before dedup"
    );

    // Invoke dedup.
    let merged_ops = [op_root, op_a, op_b];
    tx.repo_mut()
        .dedup_evolved_heads(&merged_ops, None)
        .block_on()
        .unwrap();
    let rebased = tx.repo_mut().rebase_descendants().block_on().unwrap();

    // No extra authoring.
    assert_eq!(
        rebased, 0,
        "T6: v3.1 violation: dedup authored {rebased} commit(s) for non-identical-tree siblings"
    );

    // FAIL-OPEN: both siblings must remain visible (guard (b) fires).
    let post_vis = {
        let mut heads: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
        let mut seen: HashSet<CommitId> = HashSet::new();
        let mut count = 0;
        while let Some(id) = heads.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Ok(c) = tx.repo().store().get_commit(&id) {
                if *c.change_id() == wc_change {
                    count += 1;
                }
                for pid in c.parent_ids() {
                    heads.push(pid.clone());
                }
            }
        }
        count
    };
    eprintln!("T6 POST-dedup wc_change visible = {post_vis}");
    assert_eq!(
        post_vis, 2,
        "T6: non-identical-tree siblings must both remain visible (fail-open); got {post_vis}"
    );

    eprintln!("T6: non-identical trees → fail-open confirmed");
    drop((sibling_a, sibling_b));
}

/// T7 MECHANICAL-SIBLING CLEANUP — NEITHER SIBLING IS WC-REFERENCED.
///
/// Uses the SAME merge_operations path as SCENARIO C, but the divergent change
/// is a SLICE change (not the wc change). The slice has two generations. Lineage
/// A leaves the original; lineage B squashes to the final generation. After merge,
/// the slice change has one head (the final gen — no wc commit sits on the old gen
/// in this test). The v3 stale-gen dedup removes the old generation head directly.
///
/// This verifies the stale-gen path (steps 1-4) works when neither stale head is
/// wc-referenced — the exclusive-ancestor check passes and the stale head is
/// removed, leaving exactly 1 visible slice commit. Survivor is the final gen;
/// no lex-greatest selection is needed here (stale-gen dedup, not mechanical-sibling).
///
/// This is a regression test for the survivor-selection path that doesn't involve wc
/// commits. The slice old-gen (C1) is a view head in lineage A's op. After merge
/// with lineage B (which squashed to G4), C1 becomes stale. v3 removes C1. Result:
/// 1 visible slice commit (G4). No authoring.
#[test]
fn t7_mechanical_sibling_no_wc_lex_greatest_survives() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path().to_path_buf();

    // 1. Create slice change C1 on the shared store.
    let mut tx = repo.start_transaction();
    let c1 = create_random_commit(tx.repo_mut())
        .set_description("slice[0] original")
        .write()
        .block_on()
        .unwrap();
    let slice_change = c1.change_id().clone();
    let repo_after_c1 = tx.commit("new empty commit").block_on().unwrap();
    let op_base = repo_after_c1.operation().clone();

    // 2. LINEAGE A: leaves C1 as-is (agent does nothing to the slice).
    brevity::fork_agent_oplog(
        &repo_path,
        "agentA-t7",
        repo_after_c1.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();
    let loader_a =
        brevity::agent_repo_loader(repo_after_c1.loader(), &repo_path, "agentA-t7").unwrap();
    let repo_a = loader_a.load_at_head().block_on().unwrap();
    // Agent A does unrelated work; C1 remains the HEAD of its lineage.
    let op_lineage_a = repo_a.operation().clone();

    // 3. LINEAGE B: squashes C1 → G4 (the slice is finished).
    brevity::fork_agent_oplog(
        &repo_path,
        "agentB-t7",
        repo_after_c1.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();
    let loader_b =
        brevity::agent_repo_loader(repo_after_c1.loader(), &repo_path, "agentB-t7").unwrap();
    let repo_b = loader_b.load_at_head().block_on().unwrap();
    let mut tx = repo_b.start_transaction();
    let g4 = tx
        .repo_mut()
        .rewrite_commit(&c1)
        .set_description("[Slice 0] final — squash")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let op_lineage_b = tx
        .commit("squash C1 -> G4")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    eprintln!(
        "T7: slice original={} squash={}",
        &c1.id().hex()[..8],
        &g4.id().hex()[..8]
    );

    // 4. Reconcile: base + A (C1 still a head) + B (G4 replaces C1).
    //    C1 and G4 are now BOTH view heads — divergent slice change.
    //    v3 stale-gen dedup identifies C1 as stale (C1 is a predecessor of G4).
    //    C1 is a view head and is not wc-referenced → v3 removes it.
    //    Result: exactly 1 visible slice commit (G4). Zero authored commits.
    let ops = vec![op_base, op_lineage_a, op_lineage_b];
    let merged = repo
        .loader()
        .merge_operations(ops.clone(), Some("reconcile T7"))
        .block_on()
        .unwrap();

    // Zero authoring (strict invariant — stale-gen path never authors).
    assert_no_new_commits_authored(repo.loader(), &ops, &merged, "T7");

    // Exactly 1 visible slice commit.
    let reloaded = repo.loader().load_at(&merged).block_on().unwrap();
    let by_change = visible_commits_by_change(&reloaded);
    let slice_visible = by_change.get(&slice_change).map(|v| v.len()).unwrap_or(0);
    eprintln!("T7: slice_visible={slice_visible}");
    assert_eq!(
        slice_visible, 1,
        "T7: expected exactly 1 visible slice commit after stale-gen dedup; got {slice_visible}"
    );

    // The surviving commit must be G4 (the final squash), not C1 (the stale original).
    let surviving = &by_change[&slice_change][0];
    assert_eq!(
        surviving,
        g4.id(),
        "T7: expected survivor to be G4 ({}); got {}",
        &g4.id().hex()[..8],
        &surviving.hex()[..8]
    );

    eprintln!("T7: stale-gen dedup (no-wc path) confirmed — G4 survives, C1 removed");
}

/// T8 EMPTY-vs-NONEMPTY SIBLING PAIR (run-11 shape).
///
/// Models the production scenario from run-11: the pairwise-merge carries the
/// workspace wc commit to a new slice generation (producing an EMPTY sibling —
/// it just moved parent, no content change), while the snapshot path produces a
/// NON-EMPTY sibling (the agent actually wrote content to the working copy).
///
/// v3.2 must:
///   - Remove the empty sibling (provably lossless).
///   - Keep the non-empty sibling (the content is real).
///   - Author zero new commits.
///   - Leave exactly 1 visible commit for the wc change.
#[test]
fn t8_empty_vs_nonempty_sibling_empty_hidden_nonempty_kept() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_dir = test_repo.repo_path().to_path_buf();

    let content_path = repo_path("content.txt");

    // 1. Create slice change C1 and wc commit W (empty — no content yet).
    let mut tx = repo.start_transaction();
    let c1 = create_random_commit(tx.repo_mut())
        .set_description("slice[0] original")
        .write()
        .block_on()
        .unwrap();
    let slice_change = c1.change_id().clone();
    // Wc commit W: empty (same tree as C1, which is the empty tree).
    let w = tx
        .repo_mut()
        .new_commit(vec![c1.id().clone()], c1.tree())
        .set_description("")
        .write()
        .block_on()
        .unwrap();
    let wc_change = w.change_id().clone();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let repo_shared = tx.commit("wc commit W on C1").block_on().unwrap();
    let op_base = repo_shared.operation().clone();

    // 2. LINEAGE A: agent writes real content to the wc → W1 (NON-EMPTY).
    brevity::fork_agent_oplog(
        &repo_dir,
        "agentA-t8",
        repo_shared.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();
    let loader_a =
        brevity::agent_repo_loader(repo_shared.loader(), &repo_dir, "agentA-t8").unwrap();
    let repo_a = loader_a.load_at_head().block_on().unwrap();
    let w_reloaded = repo_a.store().get_commit(w.id()).unwrap();
    let tree_with_content = create_tree(repo, &[(content_path, "agent wrote this\n")]);
    let mut tx = repo_a.start_transaction();
    let w1 = tx
        .repo_mut()
        .rewrite_commit(&w_reloaded)
        .set_description("")
        .set_tree(tree_with_content)
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let op_lineage_a = tx
        .commit("snapshot wc W → W1 (non-empty)")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    // Verify W1 is non-empty.
    {
        let repo_check = loader_a.load_at(&op_lineage_a).block_on().unwrap();
        let w1_check = repo_check.store().get_commit(w1.id()).unwrap();
        assert!(
            !w1_check.is_empty(&*repo_check).block_on().unwrap_or(true),
            "T8 setup: W1 must be non-empty"
        );
    }

    // 3. LINEAGE B: squashes C1 → G4 (slice done); rebase_descendants carries
    //    W (the wc commit) onto G4 → W_b (EMPTY: same tree, just new parent).
    brevity::fork_agent_oplog(
        &repo_dir,
        "agentB-t8",
        repo_shared.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();
    let loader_b =
        brevity::agent_repo_loader(repo_shared.loader(), &repo_dir, "agentB-t8").unwrap();
    let repo_b = loader_b.load_at_head().block_on().unwrap();
    let c1_b = repo_b.store().get_commit(c1.id()).unwrap();
    let mut tx = repo_b.start_transaction();
    let g4 = tx
        .repo_mut()
        .rewrite_commit(&c1_b)
        .set_description("[Slice 0] final")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let op_lineage_b = tx
        .commit("squash C1 → G4")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    eprintln!(
        "T8: slice={} -> {} ; wc_orig={}",
        &c1.id().hex()[..8],
        &g4.id().hex()[..8],
        &w.id().hex()[..8]
    );

    // 4. Reconcile: the pairwise-merge rebase of W1 onto G4 creates W1' (non-empty,
    //    merge-authored), and W_b (empty, from lineage B's carry) is also a head.
    //    v3.2 must remove W_b (empty) and keep W1' (non-empty).
    let ops = vec![op_base, op_lineage_a, op_lineage_b];
    let merged = repo
        .loader()
        .merge_operations(ops.clone(), Some("reconcile T8"))
        .block_on()
        .unwrap();

    // No authoring constraint: only HONEST rewrites (W1'→W1 chain) are allowed.
    assert_authored_commits_are_honest_rewrites(repo.loader(), &ops, &merged, "T8");

    let reloaded = repo.loader().load_at(&merged).block_on().unwrap();
    dump_chain(&reloaded, "T8 (empty-vs-nonempty wc siblings)");

    let by_change = visible_commits_by_change(&reloaded);
    let wc_visible = by_change.get(&wc_change).map(|v| v.len()).unwrap_or(0);
    let slice_visible = by_change.get(&slice_change).map(|v| v.len()).unwrap_or(0);
    eprintln!("T8: wc_visible={wc_visible} slice_visible={slice_visible}");

    // Exactly 1 visible wc commit (the non-empty one).
    assert_eq!(
        wc_visible, 1,
        "T8: expected exactly 1 visible wc commit after v3.2 empty-sibling cleanup; got \
         {wc_visible}"
    );

    // Verify the surviving wc commit is non-empty.
    let surviving_wc_id = &by_change[&wc_change][0];
    let surviving_commit = reloaded.store().get_commit(surviving_wc_id).unwrap();
    assert!(
        !surviving_commit
            .is_empty(&*reloaded)
            .block_on()
            .unwrap_or(true),
        "T8: the surviving wc commit must be non-empty (the content-bearing one); got empty"
    );

    // Exactly 1 visible slice commit.
    assert_eq!(
        slice_visible, 1,
        "T8: expected exactly 1 visible slice commit; got {slice_visible}"
    );

    eprintln!("T8: empty-vs-nonempty cleanup confirmed — non-empty survives");
}

/// T9 EMPTY SIBLING IS WC-REFERENCED — FAIL-OPEN (wc protection wins).
///
/// Same shape as T8, but the workspace's current wc commit is the EMPTY one.
/// Guard (a) (wc-referenced → not removable) must fire, leaving both siblings
/// visible. Zero new commits authored.
#[test]
fn t9_empty_sibling_is_wc_referenced_fail_open() {
    use jj_lib::repo::MutableRepo;

    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;

    let content_path = repo_path("content.txt");

    // Build: a non-empty commit and an empty commit for the same change id,
    // then register the EMPTY one as the workspace wc commit.
    let mut tx = repo.start_transaction();
    let base = create_random_commit(tx.repo_mut())
        .set_description("base")
        .write()
        .block_on()
        .unwrap();
    let wc_change = base.change_id().clone();
    let repo_base = tx.commit("base").block_on().unwrap();
    let op_base = repo_base.operation().clone();

    // Non-empty rewrite.
    let tree_ne = create_tree(repo, &[(content_path, "real content\n")]);
    let mut tx = repo_base.start_transaction();
    let nonempty_commit = tx
        .repo_mut()
        .rewrite_commit(&base)
        .set_description("")
        .set_tree(tree_ne)
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let repo_ne = tx.commit("non-empty rewrite").block_on().unwrap();
    let op_ne = repo_ne.operation().clone();

    // Empty rewrite (same tree as base — provably empty).
    let mut tx = repo_base.start_transaction();
    let empty_commit = tx
        .repo_mut()
        .rewrite_commit(&base)
        .set_description("")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let repo_em = tx.commit("empty rewrite").block_on().unwrap();
    let op_em = repo_em.operation().clone();

    // Hand-build divergent view: both nonempty_commit and empty_commit are heads.
    // Register the EMPTY commit as the workspace wc commit.
    let mut tx = repo_em.start_transaction();
    let mut_repo: &mut MutableRepo = tx.repo_mut();
    let ne_commit = mut_repo.store().get_commit(nonempty_commit.id()).unwrap();
    mut_repo.add_head(&ne_commit).block_on().unwrap();
    mut_repo
        .set_wc_commit(
            jj_lib::ref_name::WorkspaceNameBuf::from("test-wc"),
            empty_commit.id().clone(),
        )
        .unwrap();
    mut_repo.rebase_descendants().block_on().unwrap();

    // Pre-dedup: both visible.
    let pre_vis = {
        let mut heads: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
        let mut seen: HashSet<CommitId> = HashSet::new();
        let mut count = 0;
        while let Some(id) = heads.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Ok(c) = tx.repo().store().get_commit(&id) {
                if *c.change_id() == wc_change {
                    count += 1;
                }
                for pid in c.parent_ids() {
                    heads.push(pid.clone());
                }
            }
        }
        count
    };
    assert!(
        pre_vis >= 2,
        "T9 setup: both siblings must be visible before dedup"
    );

    // Invoke dedup.  The empty commit IS wc-referenced → guard (a) must fire.
    // The non-empty commit is the candidate; it fails guard (b) because it is
    // non-empty and non-tree-identical to the empty survivor → also fail-open.
    let merged_ops = [op_base, op_ne, op_em];
    tx.repo_mut()
        .dedup_evolved_heads(&merged_ops, None)
        .block_on()
        .unwrap();
    let rebased = tx.repo_mut().rebase_descendants().block_on().unwrap();

    // No authoring.
    assert_eq!(
        rebased, 0,
        "T9: dedup must not author commits; got {rebased}"
    );

    // Both siblings must remain (wc protection for the empty one, non-identical
    // trees for the non-empty candidate).
    let post_vis = {
        let mut heads: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
        let mut seen: HashSet<CommitId> = HashSet::new();
        let mut count = 0;
        while let Some(id) = heads.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Ok(c) = tx.repo().store().get_commit(&id) {
                if *c.change_id() == wc_change {
                    count += 1;
                }
                for pid in c.parent_ids() {
                    heads.push(pid.clone());
                }
            }
        }
        count
    };
    assert_eq!(
        post_vis, 2,
        "T9: both siblings must remain when empty sibling is wc-referenced; got {post_vis}"
    );

    // The wc commit must still be the empty one.
    let wc_id = tx
        .repo()
        .view()
        .get_wc_commit_id(jj_lib::ref_name::WorkspaceName::new("test-wc"));
    assert_eq!(
        wc_id,
        Some(empty_commit.id()),
        "T9: empty wc commit must not be removed"
    );

    eprintln!("T9: wc protection for empty sibling confirmed — fail-open");
    drop((nonempty_commit, empty_commit));
}

/// T10 BOTH NON-EMPTY, DIFFERENT TREES — STILL FAIL-OPEN.
///
/// Regression: v3.2 must not change the behavior for non-empty non-identical
/// siblings. Guard (b) must still fire → fail-open (both remain, no authoring).
/// This is the same check as T6 but exercised via the full merge_operations path.
#[test]
fn t10_both_nonempty_different_trees_still_fail_open() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path_buf = test_repo.repo_path().to_path_buf();

    let path_a = repo_path("file_a.txt");
    let path_b = repo_path("file_b.txt");

    // Create a shared base commit.
    let mut tx = repo.start_transaction();
    let base = create_random_commit(tx.repo_mut())
        .set_description("base")
        .write()
        .block_on()
        .unwrap();
    let wc_change = base.change_id().clone();
    let repo_base = tx.commit("base").block_on().unwrap();
    let op_base = repo_base.operation().clone();

    // Lineage A: rewrite base with tree_a content.
    let tree_a = create_tree(repo, &[(path_a, "content from A\n")]);
    brevity::fork_agent_oplog(
        &repo_path_buf,
        "agentA-t10",
        repo_base.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();
    let loader_a =
        brevity::agent_repo_loader(repo_base.loader(), &repo_path_buf, "agentA-t10").unwrap();
    let repo_a = loader_a.load_at_head().block_on().unwrap();
    let base_a = repo_a.store().get_commit(base.id()).unwrap();
    let mut tx = repo_a.start_transaction();
    let _sibling_a = tx
        .repo_mut()
        .rewrite_commit(&base_a)
        .set_description("")
        .set_tree(tree_a)
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let op_a = tx
        .commit("rewrite base → content-A")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    // Lineage B: rewrite base with tree_b content (different).
    let tree_b = create_tree(repo, &[(path_b, "content from B\n")]);
    brevity::fork_agent_oplog(
        &repo_path_buf,
        "agentB-t10",
        repo_base.op_heads_store().as_ref(),
    )
    .block_on()
    .unwrap();
    let loader_b =
        brevity::agent_repo_loader(repo_base.loader(), &repo_path_buf, "agentB-t10").unwrap();
    let repo_b = loader_b.load_at_head().block_on().unwrap();
    let base_b = repo_b.store().get_commit(base.id()).unwrap();
    let mut tx = repo_b.start_transaction();
    let _sibling_b = tx
        .repo_mut()
        .rewrite_commit(&base_b)
        .set_description("")
        .set_tree(tree_b)
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let op_b = tx
        .commit("rewrite base → content-B")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    // Reconcile.
    let ops = vec![op_base, op_a, op_b];
    let merged = repo
        .loader()
        .merge_operations(ops.clone(), Some("reconcile T10"))
        .block_on()
        .unwrap();

    let reloaded = repo.loader().load_at(&merged).block_on().unwrap();
    let by_change = visible_commits_by_change(&reloaded);
    let wc_visible = by_change.get(&wc_change).map(|v| v.len()).unwrap_or(0);
    eprintln!("T10: wc_visible={wc_visible}");

    // FAIL-OPEN: both non-empty non-identical siblings remain visible.
    assert_eq!(
        wc_visible, 2,
        "T10: non-empty non-identical siblings must both remain visible; got {wc_visible}"
    );

    eprintln!("T10: both non-empty different-tree siblings remain — fail-open confirmed");
}

/// T11 BOTH-EMPTY TWO-WRITER PAIR — NO PREDECESSOR EDGES (run-12 / delta-8 shape).
///
/// Models the production two-writer pattern: wave pre-create commits an empty
/// placeholder under change id C (lineage A), loop task-describe independently
/// rewrites the same original commit also producing an empty result (lineage B).
/// The two results E_a and E_b share the same change_id, but neither is a
/// transitive predecessor of the other — they only share the common predecessor R.
///
/// The critical v3.3 invariant tested here: `any_merge_authored` is false for both
/// (they were NOT authored by pairwise-merge rebase, they came from two independent
/// agent op lineages), so the v3.2 gate would fail-open. v3.3 must recognize
/// `any_empty_non_wc` alone as sufficient to collapse to 1 visible, no authoring.
#[test]
fn t11_both_empty_no_predecessor_edges_two_writer() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_dir = test_repo.repo_path().to_path_buf();

    // 1. Create shared base commit R (empty — tree == root.tree so is_empty() = true).
    //    Use new_commit with the root commit's tree so R has no content change.
    //    Both agent lineages will rewrite R independently.
    let mut tx = repo.start_transaction();
    let root_id = repo.store().root_commit_id().clone();
    let root_commit = repo.store().get_commit(&root_id).unwrap();
    let r = tx
        .repo_mut()
        .new_commit(vec![root_id.clone()], root_commit.tree())
        .set_description("pre-create placeholder R")
        .write()
        .block_on()
        .unwrap();
    let target_change = r.change_id().clone();
    let repo_r = tx.commit("create R").block_on().unwrap();
    let op_r = repo_r.operation().clone();

    // 2. LINEAGE A (wave pre-create): rewrites R with same empty tree, different desc.
    //    Produces E_a: change_id=target_change, predecessor=R, tree == root.tree → empty.
    brevity::fork_agent_oplog(&repo_dir, "writer-a-t11", repo_r.op_heads_store().as_ref())
        .block_on()
        .unwrap();
    let loader_a = brevity::agent_repo_loader(repo_r.loader(), &repo_dir, "writer-a-t11").unwrap();
    let repo_a = loader_a.load_at_head().block_on().unwrap();
    let r_in_a = repo_a.store().get_commit(r.id()).unwrap();
    let mut tx = repo_a.start_transaction();
    let e_a = tx
        .repo_mut()
        .rewrite_commit(&r_in_a)
        .set_description("slice[3] beta-placeholder")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let op_a = tx
        .commit("pre-create empty")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    // 3. LINEAGE B (loop task-describe): rewrites R with same empty tree, different
    //    desc — independently, no knowledge of E_a. Produces E_b: same change_id,
    //    predecessor=R, no predecessor edge to E_a.
    brevity::fork_agent_oplog(&repo_dir, "writer-b-t11", repo_r.op_heads_store().as_ref())
        .block_on()
        .unwrap();
    let loader_b = brevity::agent_repo_loader(repo_r.loader(), &repo_dir, "writer-b-t11").unwrap();
    let repo_b = loader_b.load_at_head().block_on().unwrap();
    let r_in_b = repo_b.store().get_commit(r.id()).unwrap();
    let mut tx = repo_b.start_transaction();
    let e_b = tx
        .repo_mut()
        .rewrite_commit(&r_in_b)
        .set_description("[Slice 3] beta-loop-describe")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let op_b = tx
        .commit("loop-describe empty")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    // Verify both E_a and E_b are empty.
    {
        let check_a = loader_a.load_at(&op_a).block_on().unwrap();
        let check_b = loader_b.load_at(&op_b).block_on().unwrap();
        assert!(
            check_a
                .store()
                .get_commit(e_a.id())
                .unwrap()
                .is_empty(&*check_a)
                .block_on()
                .unwrap_or(false),
            "T11 setup: E_a must be empty"
        );
        assert!(
            check_b
                .store()
                .get_commit(e_b.id())
                .unwrap()
                .is_empty(&*check_b)
                .block_on()
                .unwrap_or(false),
            "T11 setup: E_b must be empty"
        );
    }

    eprintln!(
        "T11: both-empty pair E_a={} E_b={} change={}",
        &e_a.id().hex()[..8],
        &e_b.id().hex()[..8],
        &target_change.reverse_hex()[..8]
    );

    // A generic reconciliation must preserve the honest description divergence.
    let ops = vec![op_r, op_a, op_b];
    let generic_merged = repo
        .loader()
        .merge_operations(ops.clone(), Some("generic reconcile T11"))
        .block_on()
        .unwrap();
    let generic_reloaded = repo.loader().load_at(&generic_merged).block_on().unwrap();
    let generic_by_change = visible_commits_by_change(&generic_reloaded);
    let generic_visible = generic_by_change
        .get(&target_change)
        .map(|v| v.len())
        .unwrap_or(0);
    assert_eq!(
        generic_visible, 2,
        "T11: generic reconciliation must preserve honest empty-commit divergence"
    );

    // The isolated-agent policy may collapse the known two-writer placeholder
    // shape. It removes the lex-smaller and keeps exactly 1 visible, no authoring.
    let merged = repo
        .loader()
        .merge_agent_operations(ops.clone(), Some("agent reconcile T11"))
        .block_on()
        .unwrap();

    // Only honest rewrites allowed.
    assert_authored_commits_are_honest_rewrites(repo.loader(), &ops, &merged, "T11");

    let reloaded = repo.loader().load_at(&merged).block_on().unwrap();
    let by_change = visible_commits_by_change(&reloaded);
    let visible = by_change.get(&target_change).map(|v| v.len()).unwrap_or(0);
    eprintln!("T11: target_change visible={visible}");

    assert_eq!(
        visible, 1,
        "T11: both-empty siblings must collapse to exactly 1 visible; got {visible}"
    );

    // Survivor must be the lex-greater of E_a and E_b (no wc bias here).
    let surviving_id = &by_change[&target_change][0];
    let expected_survivor = if e_a.id().hex() > e_b.id().hex() {
        e_a.id()
    } else {
        e_b.id()
    };
    assert_eq!(
        surviving_id,
        expected_survivor,
        "T11: lex-greater sibling must survive; expected {} got {}",
        &expected_survivor.hex()[..8],
        &surviving_id.hex()[..8]
    );

    eprintln!(
        "T11: both-empty two-writer collapse confirmed — survivor {}",
        &surviving_id.hex()[..8]
    );
}

/// T12 BOTH-EMPTY PAIR — WC-REFERENCED MEMBER SURVIVES.
///
/// Same structural shape as T11 (two independent lineages both rewrite a common
/// base R → E_a and E_b, both empty, neither a predecessor of the other).
/// Additionally, E_b is registered as the wc commit. The wc-referenced empty
/// sibling must survive; the non-wc empty sibling is removed. No authoring.
#[test]
fn t12_both_empty_wc_referenced_member_survives() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_dir = test_repo.repo_path().to_path_buf();

    // 1. Create shared base commit R (empty — uses root commit tree so is_empty() = true).
    let mut tx = repo.start_transaction();
    let root_id_t12 = repo.store().root_commit_id().clone();
    let root_commit_t12 = repo.store().get_commit(&root_id_t12).unwrap();
    let r = tx
        .repo_mut()
        .new_commit(vec![root_id_t12.clone()], root_commit_t12.tree())
        .set_description("base R for T12")
        .write()
        .block_on()
        .unwrap();
    let target_change = r.change_id().clone();
    let repo_r = tx.commit("create R T12").block_on().unwrap();
    let op_r = repo_r.operation().clone();

    // 2. Lineage A: rewrites R → E_a (empty, predecessor = R).
    brevity::fork_agent_oplog(&repo_dir, "writer-a-t12", repo_r.op_heads_store().as_ref())
        .block_on()
        .unwrap();
    let loader_a = brevity::agent_repo_loader(repo_r.loader(), &repo_dir, "writer-a-t12").unwrap();
    let repo_a = loader_a.load_at_head().block_on().unwrap();
    let r_in_a = repo_a.store().get_commit(r.id()).unwrap();
    let mut tx = repo_a.start_transaction();
    let e_a = tx
        .repo_mut()
        .rewrite_commit(&r_in_a)
        .set_description("slice placeholder A-t12")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let op_a = tx
        .commit("lineage-A empty T12")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    // 3. Lineage B: independently rewrites R → E_b (empty, predecessor = R, no
    //    edge to E_a). Forks from op_r, not op_a.
    brevity::fork_agent_oplog(&repo_dir, "writer-b-t12", repo_r.op_heads_store().as_ref())
        .block_on()
        .unwrap();
    let loader_b = brevity::agent_repo_loader(repo_r.loader(), &repo_dir, "writer-b-t12").unwrap();
    let repo_b = loader_b.load_at_head().block_on().unwrap();
    let r_in_b = repo_b.store().get_commit(r.id()).unwrap();
    let mut tx = repo_b.start_transaction();
    let e_b = tx
        .repo_mut()
        .rewrite_commit(&r_in_b)
        .set_description("[Slice] placeholder B-t12")
        .write()
        .block_on()
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();
    let op_b = tx
        .commit("lineage-B empty T12")
        .block_on()
        .unwrap()
        .operation()
        .clone();

    // Verify both are empty.
    {
        let check_a = loader_a.load_at(&op_a).block_on().unwrap();
        let check_b = loader_b.load_at(&op_b).block_on().unwrap();
        assert!(
            check_a
                .store()
                .get_commit(e_a.id())
                .unwrap()
                .is_empty(&*check_a)
                .block_on()
                .unwrap_or(false),
            "T12 setup: E_a must be empty"
        );
        assert!(
            check_b
                .store()
                .get_commit(e_b.id())
                .unwrap()
                .is_empty(&*check_b)
                .block_on()
                .unwrap_or(false),
            "T12 setup: E_b must be empty"
        );
    }

    // 4. Build a merged view that sees both E_a and E_b as heads, with E_b
    //    registered as the wc commit. Then call dedup_evolved_heads directly.
    //    We use a fresh transaction on repo_b's snapshot (which has E_b as head).
    let repo_b_snap = loader_b.load_at(&op_b).block_on().unwrap();
    let mut tx = repo_b_snap.start_transaction();
    // E_a was hidden by the lineage-B rewrite in its own oplog; force it back
    // into the view so we see both siblings.
    let e_a_commit = tx.repo_mut().store().get_commit(e_a.id()).unwrap();
    tx.repo_mut().add_head(&e_a_commit).block_on().unwrap();
    // Register E_b as the wc commit.
    tx.repo_mut()
        .set_wc_commit(
            jj_lib::ref_name::WorkspaceNameBuf::from("ws-t12"),
            e_b.id().clone(),
        )
        .unwrap();
    tx.repo_mut().rebase_descendants().block_on().unwrap();

    // Pre-dedup sanity: both must appear as view heads (or ancestors).
    let pre_heads: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
    let pre_change_count = pre_heads
        .iter()
        .filter(|id| {
            tx.repo()
                .store()
                .get_commit(id)
                .ok()
                .filter(|c| *c.change_id() == target_change)
                .is_some()
        })
        .count();
    assert!(
        pre_change_count >= 2,
        "T12 setup: both must be view heads before dedup; got {pre_change_count}"
    );

    // Invoke dedup: E_b is wc-referenced → it must survive; E_a must be removed.
    let merged_ops = [op_r.clone(), op_a.clone(), op_b.clone()];
    tx.repo_mut()
        .dedup_agent_evolved_heads(&merged_ops, None)
        .block_on()
        .unwrap();
    let rebased = tx.repo_mut().rebase_descendants().block_on().unwrap();
    assert_eq!(rebased, 0, "T12: dedup must not author commits");

    // Post-dedup: exactly 1 visible with target_change, must be E_b.
    let post_heads: Vec<CommitId> = tx.repo().view().heads().iter().cloned().collect();
    let survivors: Vec<CommitId> = post_heads
        .iter()
        .filter(|id| {
            tx.repo()
                .store()
                .get_commit(id)
                .ok()
                .filter(|c| *c.change_id() == target_change)
                .is_some()
        })
        .cloned()
        .collect();

    assert_eq!(
        survivors.len(),
        1,
        "T12: both-empty pair with wc-ref must collapse to 1 visible; got {}",
        survivors.len()
    );
    assert_eq!(
        &survivors[0],
        e_b.id(),
        "T12: wc-referenced E_b must survive; got {}",
        &survivors[0].hex()[..8]
    );

    // wc must still point to E_b.
    let wc_id = tx
        .repo()
        .view()
        .get_wc_commit_id(jj_lib::ref_name::WorkspaceName::new("ws-t12"));
    assert_eq!(wc_id, Some(e_b.id()), "T12: wc must still point to E_b");

    eprintln!("T12: wc-referenced empty sibling survives — confirmed");
}
