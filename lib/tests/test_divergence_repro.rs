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
use jj_lib::op_heads_store::OpHeadsStore as _;
use jj_lib::repo::ReadonlyRepo;
use jj_lib::repo::Repo as _;
use pollster::FutureExt as _;
use testutils::TestRepo;
use testutils::create_random_commit;

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

/// v3 no-authoring invariant helper: after `merge_operations`, the set of all
/// visible commit ids in the merged result must be a subset of the union of
/// the visible commit id sets of all the input ops.
///
/// A new commit id in the merged result that was not in ANY input op means
/// `merge_operations` authored a commit — a v3 invariant violation.
fn assert_no_new_commits_authored(
    loader: &jj_lib::repo::RepoLoader,
    input_ops: &[jj_lib::operation::Operation],
    merged_op: &jj_lib::operation::Operation,
    label: &str,
) {
    // Collect all commit ids visible in each input op.
    let mut all_input_commits: HashSet<CommitId> = HashSet::new();
    for op in input_ops {
        let repo = loader.load_at(op).unwrap();
        let heads: Vec<CommitId> = repo.view().heads().iter().cloned().collect();
        let mut stack = heads;
        let mut visited: HashSet<CommitId> = HashSet::new();
        while let Some(id) = stack.pop() {
            if !visited.insert(id.clone()) {
                continue;
            }
            all_input_commits.insert(id.clone());
            if let Ok(commit) = repo.store().get_commit(&id) {
                for pid in commit.parent_ids() {
                    stack.push(pid.clone());
                }
            }
        }
    }

    // Walk the merged result and find any commit not in the input set.
    let merged_repo = loader.load_at(merged_op).unwrap();
    let heads: Vec<CommitId> = merged_repo.view().heads().iter().cloned().collect();
    let mut stack = heads;
    let mut visited: HashSet<CommitId> = HashSet::new();
    while let Some(id) = stack.pop() {
        if !visited.insert(id.clone()) {
            continue;
        }
        assert!(
            all_input_commits.contains(&id),
            "[{label}] v3 no-authoring violated: merged result contains commit {} \
             that was not visible in ANY input op",
            &id.hex()[..12.min(id.hex().len())]
        );
        if let Ok(commit) = merged_repo.store().get_commit(&id) {
            for pid in commit.parent_ids() {
                stack.push(pid.clone());
            }
        }
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
        .unwrap();
    let change_id = c1.change_id().clone();
    let shared_after_c1 = tx.commit("new empty commit").unwrap();

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
    let agent_repo = agent_loader.load_at_head().unwrap();

    // describe #1
    let mut tx = agent_repo.start_transaction();
    let g2 = tx
        .repo_mut()
        .rewrite_commit(&c1)
        .set_description("slice[0] intermediate")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let agent_repo = tx.commit("describe commit C1").unwrap();

    // describe #2
    let mut tx = agent_repo.start_transaction();
    let g3 = tx
        .repo_mut()
        .rewrite_commit(&g2)
        .set_description("slice[0] intermediate 2")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let agent_repo = tx.commit("describe commit G2").unwrap();

    // squash-equivalent rewrite #3
    let mut tx = agent_repo.start_transaction();
    let _g4 = tx
        .repo_mut()
        .rewrite_commit(&g3)
        .set_description("[Slice 0] final")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    tx.commit("squash commits into G3").unwrap();

    // 5. Merge-back: reconcile divergent operations on the SHARED store.
    brevity::merge_agent_oplog(shared_after_c1.loader(), &repo_path, "agent-0")
        .block_on()
        .unwrap();

    // 6. Load shared at head; assert exactly ONE visible commit for C1.
    let reloaded = repo.loader().load_at_head().unwrap();
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
        .unwrap();
    let change_id = c1.change_id().clone();
    let shared_after_c1 = tx.commit("new empty commit").unwrap();
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
    let agent_repo = agent_loader.load_at_head().unwrap();
    let mut tx = agent_repo.start_transaction();
    let g2 = tx
        .repo_mut()
        .rewrite_commit(&c1)
        .set_description("slice[0] intermediate")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let agent_repo_g2 = tx.commit("describe commit C1 -> G2").unwrap();
    let op_g2 = agent_repo_g2.operation().clone();

    let agent_repo = agent_repo_g2;
    let mut tx = agent_repo.start_transaction();
    let g3 = tx
        .repo_mut()
        .rewrite_commit(&g2)
        .set_description("slice[0] intermediate 2")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let agent_repo = tx.commit("describe commit G2 -> G3").unwrap();

    let mut tx = agent_repo.start_transaction();
    let _g4 = tx
        .repo_mut()
        .rewrite_commit(&g3)
        .set_description("[Slice 0] final")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let agent_repo_g4 = tx.commit("squash commits into G3 -> G4").unwrap();
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
            .unwrap();

        // v3 no-authoring invariant.
        assert_no_new_commits_authored(
            repo.loader(),
            &ops,
            &merged,
            &format!("SCENARIO B order {order:?}"),
        );

        let reloaded = repo.loader().load_at(&merged).unwrap();
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
        .unwrap();
    let change_id = c1.change_id().clone();
    let shared_after_c1 = tx.commit("new empty commit").unwrap();
    let op_g1 = shared_after_c1.operation().clone();
    eprintln!(
        "G1 op={} C1 commit={}",
        op_g1.id().hex()[..8].to_string(),
        c1.id().hex()[..8].to_string()
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
    let agent_repo = agent_loader.load_at_head().unwrap();

    let mut tx = agent_repo.start_transaction();
    let g2 = tx
        .repo_mut()
        .rewrite_commit(&c1)
        .set_description("slice[0] intermediate")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let agent_repo_g2 = tx.commit("describe commit C1 -> G2").unwrap();
    let op_g2 = agent_repo_g2.operation().clone();
    eprintln!(
        "G2 op={} G2 commit={}",
        op_g2.id().hex()[..8].to_string(),
        g2.id().hex()[..8].to_string()
    );

    let agent_repo = agent_repo_g2;
    let mut tx = agent_repo.start_transaction();
    let g3 = tx
        .repo_mut()
        .rewrite_commit(&g2)
        .set_description("slice[0] intermediate 2")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let agent_repo = tx.commit("describe commit G2 -> G3").unwrap();
    eprintln!("G3 commit={}", g3.id().hex()[..8].to_string());

    let mut tx = agent_repo.start_transaction();
    let g4 = tx
        .repo_mut()
        .rewrite_commit(&g3)
        .set_description("[Slice 0] final")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let agent_repo_g4 = tx.commit("squash commits into G3 -> G4").unwrap();
    let op_g4 = agent_repo_g4.operation().clone();
    eprintln!(
        "G4 op={} G4 commit={}",
        op_g4.id().hex()[..8].to_string(),
        g4.id().hex()[..8].to_string()
    );

    // Feed [G1, G4, G2] — intermediate G2 last.
    let ops = vec![op_g1, op_g4, op_g2];
    let merged = repo
        .loader()
        .merge_operations(ops.clone(), Some("reconcile divergent operations"))
        .unwrap();

    // v3 no-authoring invariant.
    assert_no_new_commits_authored(repo.loader(), &ops, &merged, "MINIMAL [G1,G4,G2]");

    let reloaded = repo.loader().load_at(&merged).unwrap();
    dump_chain(&reloaded, "MINIMAL [G1,G4,G2]");
    let by_change = visible_commits_by_change(&reloaded);
    let visible = by_change.get(&change_id).map(|v| v.len()).unwrap_or(0);
    assert_eq!(visible, 1, "got {visible} visible commits (DIVERGENT)");
}

/// SCENARIO C (PRODUCTION SHAPE): the slice change has TWO generations, and
/// the SAME workspace working-copy commit (one change id) sits on each
/// generation in two different reconcile lineages.
///
/// v3 behavior: the stale slice gen (C1) is NOT a view head — it is a
/// NON-HEAD parent of the stale wc commit W1. The stale wc commit W1 is a
/// non-stale visible descendant of C1 (it has a different change_id). So v3
/// fails open and leaves the slice divergence intact. This is intentional —
/// dedup is idempotent and a later reconcile (after the stale wc commit is
/// retired) can finish the job.
///
/// The CRITICAL invariant is that NO new commits are authored by reconcile.
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
        .unwrap();
    let slice_change = c1.change_id().clone();
    let shared_after_c1 = tx.commit("new empty commit").unwrap();

    // 2. Create the workspace wc commit W on top of the ORIGINAL gen, on the
    //    SHARED store, so both lineages fork from a state that already has W.
    let mut tx = shared_after_c1.start_transaction();
    let w = tx
        .repo_mut()
        .new_commit(vec![c1.id().clone()], c1.tree())
        .set_description("")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let shared_after_w = tx
        .commit("create initial working-copy commit in workspace loop-0")
        .unwrap();
    let _wc_change = w.change_id().clone();
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
    let repo_a = loader_a.load_at_head().unwrap();
    let mut tx = repo_a.start_transaction();
    let w1 = tx
        .repo_mut()
        .rewrite_commit(&w)
        .set_description("")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let op_lineage_a = tx
        .commit("snapshot working copy")
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
    let repo_b = loader_b.load_at_head().unwrap();

    // describe C1 -> G2, reparent wc W onto G2
    let mut tx = repo_b.start_transaction();
    let g2 = tx
        .repo_mut()
        .rewrite_commit(&c1)
        .set_description("slice[0] gamma intermediate")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let repo_b = tx.commit("describe commit C1 -> G2").unwrap();

    // describe G2 -> G3
    let g2 = repo_b.store().get_commit(g2.id()).unwrap();
    let mut tx = repo_b.start_transaction();
    let g3 = tx
        .repo_mut()
        .rewrite_commit(&g2)
        .set_description("slice[0] gamma intermediate 2")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let repo_b = tx.commit("describe commit G2 -> G3").unwrap();

    // squash G3 -> G4
    let g3 = repo_b.store().get_commit(g3.id()).unwrap();
    let mut tx = repo_b.start_transaction();
    let g4 = tx
        .repo_mut()
        .rewrite_commit(&g3)
        .set_description("[Slice 0] gamma final")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let op_lineage_b = tx
        .commit("squash commits into G3 -> G4")
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
        .unwrap();

    // v3 no-authoring invariant: reconcile must NEVER author new commits.
    assert_no_new_commits_authored(repo.loader(), &ops, &merged, "SCENARIO C");

    let reloaded = repo.loader().load_at(&merged).unwrap();
    dump_chain(&reloaded, "SCENARIO C (shared wc change pins both gens)");

    let by_change = visible_commits_by_change(&reloaded);
    let slice_visible = by_change.get(&slice_change).map(|v| v.len()).unwrap_or(0);
    eprintln!("slice change visible generations: {slice_visible}");

    // v3 fail-open: the stale C1 gen is pinned by a non-stale wc commit (W1).
    // Dedup cannot remove C1 without removing the stale wc head first.
    // The divergence persists; the no-authoring invariant is what matters here.
    // (A future reconcile after W1 is retired can finish collapsing this.)
    eprintln!(
        "SCENARIO C: slice has {slice_visible} visible generation(s) — v3 fails open on \
         wc-pinned divergence; no-authoring is the key invariant"
    );
    // No assertion on slice_visible here: fail-open means >= 1 is acceptable.
    // The no-authoring invariant above is the binding assertion.
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
        .unwrap();
    let slice_change = c1.change_id().clone();
    let repo1 = tx.commit("new empty commit").unwrap();
    let op_c1 = repo1.operation().clone();

    // Rewrite C1 -> G4 (records the predecessor edge) and capture the op.
    let mut tx = repo1.start_transaction();
    let g4 = tx
        .repo_mut()
        .rewrite_commit(&c1)
        .set_description("[Slice 0] final")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let repo_g4 = tx.commit("squash commit C1 -> G4").unwrap();
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
        .unwrap();
    // wc child on the NEW gen G4 -> keeps G4 visible-but-not-head.
    let _w2 = mut_repo
        .new_commit(vec![g4.id().clone()], g4.tree())
        .set_description("wc commit (workspace loop-new)")
        .write()
        .unwrap();
    mut_repo.rebase_descendants().unwrap();

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
        .unwrap();
    let rebased = tx.repo_mut().rebase_descendants().unwrap();

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
        .unwrap();
    let change_id = c_gen0.change_id().clone();
    let shared_base = tx.commit("initial: c-gen0").unwrap();
    let op_gen0 = shared_base.operation().clone();

    // Agent forks and produces C-gen0 -> C-gen1 -> C-gen2.
    brevity::fork_agent_oplog(&repo_path, "agent-a", shared_base.op_heads_store().as_ref())
        .block_on()
        .unwrap();
    let agent_loader =
        brevity::agent_repo_loader(shared_base.loader(), &repo_path, "agent-a").unwrap();
    let agent_repo = agent_loader.load_at_head().unwrap();

    let mut tx = agent_repo.start_transaction();
    let c_gen1 = tx
        .repo_mut()
        .rewrite_commit(&c_gen0)
        .set_description("c-gen1")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let agent_repo = tx.commit("rewrite gen0 -> gen1").unwrap();
    let op_gen1 = agent_repo.operation().clone();

    let c_gen1 = agent_repo.store().get_commit(c_gen1.id()).unwrap();
    let mut tx = agent_repo.start_transaction();
    let c_gen2 = tx
        .repo_mut()
        .rewrite_commit(&c_gen1)
        .set_description("c-gen2 final")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let agent_repo = tx.commit("rewrite gen1 -> gen2").unwrap();
    let op_gen2 = agent_repo.operation().clone();

    // M1 = merge of [op_gen0, op_gen2, op_gen1]: gen0 and gen1 are stale heads.
    // v3 must: remove gen0 and gen1 from heads, leave gen2. No new commits.
    let m1_ops: Vec<Operation> = vec![op_gen0.clone(), op_gen2.clone(), op_gen1.clone()];
    let m1 = repo
        .loader()
        .merge_operations(m1_ops.clone(), Some("reconcile R1"))
        .unwrap();

    assert_no_new_commits_authored(repo.loader(), &m1_ops, &m1, "T1 R1");
    {
        let r1_repo = repo.loader().load_at(&m1).unwrap();
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
        .unwrap();

    assert_no_new_commits_authored(repo.loader(), &m2_ops, &m2, "T1 R2");
    {
        let r2_repo = repo.loader().load_at(&m2).unwrap();
        dump_chain(&r2_repo, "T1 R2 (M1 + L2/gen0)");
        let by_change = visible_commits_by_change(&r2_repo);
        let visible = by_change.get(&change_id).map(|v| v.len()).unwrap_or(0);
        assert_eq!(
            visible, 1,
            "T1 R2: expected 1 visible after M2, got {visible} (sibling divergence)"
        );
    }

    let _ = c_gen2;
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
        .unwrap();
    let p_change = p_old.change_id().clone();
    let c_old = tx
        .repo_mut()
        .new_commit(vec![p_old.id().clone()], p_old.tree())
        .set_description("C-old")
        .write()
        .unwrap();
    let c_change = c_old.change_id().clone();
    tx.repo_mut().rebase_descendants().unwrap();
    let shared_base = tx.commit("initial: P-old + C-old").unwrap();
    let op_lineage_a = shared_base.operation().clone();

    // Lineage B: rewrites P-old -> P-new, then C-old -> C-new.
    brevity::fork_agent_oplog(&repo_path, "agent-b", shared_base.op_heads_store().as_ref())
        .block_on()
        .unwrap();
    let loader_b = brevity::agent_repo_loader(shared_base.loader(), &repo_path, "agent-b").unwrap();
    let repo_b = loader_b.load_at_head().unwrap();

    let mut tx = repo_b.start_transaction();
    let p_new = tx
        .repo_mut()
        .rewrite_commit(&p_old)
        .set_description("P-new")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let repo_b = tx.commit("rewrite P-old -> P-new").unwrap();

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
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let repo_b = tx.commit("rewrite C-rebased -> C-new").unwrap();
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
        .unwrap();

    assert_no_new_commits_authored(repo.loader(), &ops, &merged, "T2");

    let merged_repo = repo.loader().load_at(&merged).unwrap();
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

    let _ = p_new;
}

/// T3 HEAD-INVERSION + DEEP HARVEST (secondary mechanism): the edge-bearing
/// rewrite op sits at or below the CCA of the merged ops. Phase-1 harvest
/// (CCA-bounded) misses the edge; phase-2 (unbounded) must find it.
///
/// Shape:
///   - Shared history: X-orig written, then X-new written (rewrite X-orig->X-new,
///     records the predecessor edge). This forms the "deep" op that is the CCA.
///   - Two lineages fork from X-new:
///       L1: touches something else (no X-related edges).
///       L2: re-introduces X-orig as a bare head (stray op).
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
        .unwrap();
    let x_change = x_orig.change_id().clone();
    let repo_after_x_orig = tx.commit("X-orig").unwrap();

    let mut tx = repo_after_x_orig.start_transaction();
    let x_new = tx
        .repo_mut()
        .rewrite_commit(&x_orig)
        .set_description("X-new")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    // This op records the X-orig -> X-new predecessor edge.
    let shared_after_x_new = tx.commit("rewrite X-orig -> X-new").unwrap();
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
    let repo_l1 = loader_l1.load_at_head().unwrap();
    let mut tx = repo_l1.start_transaction();
    let _other = create_random_commit(tx.repo_mut())
        .set_description("other change L1")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let op_l1 = tx
        .commit("L1: add other change")
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
    let repo_l2 = loader_l2.load_at_head().unwrap();
    // Re-add X-orig as a head by using add_head. We do this via a transaction
    // that directly manipulates the view.
    let mut tx = repo_l2.start_transaction();
    // Get the actual x_orig commit from the store.
    let x_orig_from_store = tx.repo_mut().store().get_commit(x_orig.id()).unwrap();
    tx.repo_mut().add_head(&x_orig_from_store).unwrap();
    let op_l2 = tx
        .commit("L2: re-add X-orig as stray head")
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
        .unwrap();

    assert_no_new_commits_authored(repo.loader(), &ops, &merged, "T3");

    let merged_repo = repo.loader().load_at(&merged).unwrap();
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
        .unwrap();
    let a_change = a_old.change_id().clone();
    let repo_after_a_old = tx.commit("A-old").unwrap();
    let op_a_old = repo_after_a_old.operation().clone();

    let mut tx = repo_after_a_old.start_transaction();
    let _a_new = tx
        .repo_mut()
        .rewrite_commit(&a_old)
        .set_description("A-new")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let repo_after_a_new = tx.commit("rewrite A-old -> A-new").unwrap();
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
        .unwrap();
    mut_repo.rebase_descendants().unwrap();

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
        .unwrap();
    let rebased = tx.repo_mut().rebase_descendants().unwrap();

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
        .unwrap();
    let b_change = b_old.change_id().clone();
    let repo_after_b_old = tx.commit("B-old").unwrap();
    let op_b_old = repo_after_b_old.operation().clone();

    let mut tx = repo_after_b_old.start_transaction();
    let b_new = tx
        .repo_mut()
        .rewrite_commit(&b_old)
        .set_description("B-new")
        .write()
        .unwrap();
    tx.repo_mut().rebase_descendants().unwrap();
    let repo_after_b_new = tx.commit("rewrite B-old -> B-new").unwrap();
    let op_b_new = repo_after_b_new.operation().clone();

    // Hand-build: B-old is BOTH a view head AND a wc commit for workspace "main".
    // B-new is also a view head. Both are divergent.
    let mut tx = repo_after_b_new.start_transaction();
    let mut_repo: &mut MutableRepo = tx.repo_mut();

    // Ensure B-old is in the view heads (it may have been removed by the
    // rewrite+rebase above; we explicitly re-add it to simulate a stale op
    // that brought it back).
    let b_old_commit = mut_repo.store().get_commit(b_old.id()).unwrap();
    mut_repo.add_head(&b_old_commit).unwrap();

    // Set B-old as the workspace wc commit so the protection guard fires.
    mut_repo
        .set_wc_commit(
            jj_lib::ref_name::WorkspaceNameBuf::from("test-workspace"),
            b_old.id().clone(),
        )
        .unwrap();
    mut_repo.rebase_descendants().unwrap();

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
        .unwrap();
    let rebased = tx.repo_mut().rebase_descendants().unwrap();

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
    let _ = b_new;
}
