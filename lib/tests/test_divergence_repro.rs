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
//! These tests build the exact mechanism with the `brevity` module primitives
//! (the same code Hox calls) and assert: exactly ONE visible commit per
//! change_id after reconcile.

use std::collections::HashMap;

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
            .merge_operations(ops, Some("reconcile divergent operations"))
            .unwrap();
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
    let merged = repo
        .loader()
        .merge_operations(
            vec![op_g1, op_g4, op_g2],
            Some("reconcile divergent operations"),
        )
        .unwrap();
    let reloaded = repo.loader().load_at(&merged).unwrap();
    dump_chain(&reloaded, "MINIMAL [G1,G4,G2]");
    let by_change = visible_commits_by_change(&reloaded);
    let visible = by_change.get(&change_id).map(|v| v.len()).unwrap_or(0);
    assert_eq!(visible, 1, "got {visible} visible commits (DIVERGENT)");
}
