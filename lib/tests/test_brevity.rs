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

use std::path::Path;

use jj_lib::brevity;
use jj_lib::object_id::ObjectId as _;
use jj_lib::op_heads_store::OpHeadsStore as _;
use pollster::FutureExt as _;
use testutils::TestRepo;
use testutils::write_random_commit;

fn list_dir(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_str().unwrap().to_owned())
        .filter(|name| name != "lock")
        .collect::<Vec<_>>()
}

#[test]
fn test_fork_agent_oplog() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path();

    let forked = brevity::fork_agent_oplog(repo_path, "agent-0", repo.op_heads_store().as_ref())
        .block_on()
        .unwrap();

    // Verify directory structure
    let agent_dir = repo_path
        .parent()
        .unwrap()
        .join("agent-oplogs")
        .join("agent-0");
    assert!(agent_dir.exists());
    assert!(agent_dir.join("op_heads").join("heads").is_dir());
    assert_eq!(
        std::fs::read_to_string(agent_dir.join("op_heads").join("type")).unwrap(),
        "forked_op_heads_store"
    );

    // Verify forked heads match shared heads
    let shared_heads: Vec<String> = repo
        .op_heads_store()
        .get_op_heads()
        .block_on()
        .unwrap()
        .iter()
        .map(|id| id.hex())
        .collect();
    let mut forked_heads: Vec<String> = forked
        .get_op_heads()
        .block_on()
        .unwrap()
        .iter()
        .map(|id| id.hex())
        .collect();
    forked_heads.sort();
    let mut shared_sorted = shared_heads.clone();
    shared_sorted.sort();
    assert_eq!(forked_heads, shared_sorted);
}

#[test]
fn test_fork_already_exists() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path();

    // First fork succeeds
    brevity::fork_agent_oplog(repo_path, "agent-0", repo.op_heads_store().as_ref())
        .block_on()
        .unwrap();

    // Second fork with same name fails
    let result =
        brevity::fork_agent_oplog(repo_path, "agent-0", repo.op_heads_store().as_ref()).block_on();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("already exists"));
}

#[test]
fn test_agent_repo_loader() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path();

    // Fork the oplog
    brevity::fork_agent_oplog(repo_path, "agent-0", repo.op_heads_store().as_ref())
        .block_on()
        .unwrap();

    // Get an agent repo loader
    let agent_loader = brevity::agent_repo_loader(repo.loader(), repo_path, "agent-0").unwrap();
    let agent_repo = agent_loader.load_at_head().block_on().unwrap();

    // Commit via the agent repo
    let mut tx = agent_repo.start_transaction();
    write_random_commit(tx.repo_mut());
    tx.commit("agent transaction").block_on().unwrap();

    // Shared store should be unchanged
    let shared_heads_dir = repo_path.join("op_heads").join("heads");
    let shared_heads = list_dir(&shared_heads_dir);
    assert_eq!(shared_heads.len(), 1);
    assert_eq!(shared_heads[0], repo.op_id().hex());

    // Agent store should have advanced
    let agent_heads_dir = repo_path
        .parent()
        .unwrap()
        .join("agent-oplogs")
        .join("agent-0")
        .join("op_heads")
        .join("heads");
    let agent_heads = list_dir(&agent_heads_dir);
    assert_eq!(agent_heads.len(), 1);
    assert_ne!(agent_heads[0], repo.op_id().hex());
}

#[test]
fn test_merge_agent_oplog() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path();

    // Fork the oplog
    brevity::fork_agent_oplog(repo_path, "agent-0", repo.op_heads_store().as_ref())
        .block_on()
        .unwrap();

    // Agent commits a change
    let agent_loader = brevity::agent_repo_loader(repo.loader(), repo_path, "agent-0").unwrap();
    let agent_repo = agent_loader.load_at_head().block_on().unwrap();
    let mut tx = agent_repo.start_transaction();
    let agent_commit = write_random_commit(tx.repo_mut());
    tx.commit("agent work").block_on().unwrap();

    // Merge the agent's oplog back
    let merged_op = brevity::merge_agent_oplog(repo.loader(), repo_path, "agent-0")
        .block_on()
        .unwrap();

    // The merged operation should be new (not the original shared head)
    assert_ne!(merged_op.id(), repo.op_id());

    // Reload the shared repo and verify the agent's commit is visible
    let reloaded = repo.loader().load_at_head().block_on().unwrap();
    assert!(reloaded.view().heads().contains(agent_commit.id()));
}

#[test]
fn test_multiple_agents_isolated() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path();

    // Fork two agents
    brevity::fork_agent_oplog(repo_path, "agent-0", repo.op_heads_store().as_ref())
        .block_on()
        .unwrap();
    brevity::fork_agent_oplog(repo_path, "agent-1", repo.op_heads_store().as_ref())
        .block_on()
        .unwrap();

    // Agent 0 commits
    let loader0 = brevity::agent_repo_loader(repo.loader(), repo_path, "agent-0").unwrap();
    let repo0 = loader0.load_at_head().block_on().unwrap();
    let mut tx0 = repo0.start_transaction();
    let commit0 = write_random_commit(tx0.repo_mut());
    tx0.commit("agent-0 work").block_on().unwrap();

    // Agent 1 commits
    let loader1 = brevity::agent_repo_loader(repo.loader(), repo_path, "agent-1").unwrap();
    let repo1 = loader1.load_at_head().block_on().unwrap();
    let mut tx1 = repo1.start_transaction();
    let commit1 = write_random_commit(tx1.repo_mut());
    tx1.commit("agent-1 work").block_on().unwrap();

    // Neither agent sees the other's commit
    let repo0_view = loader0.load_at_head().block_on().unwrap();
    let repo1_view = loader1.load_at_head().block_on().unwrap();
    assert!(repo0_view.view().heads().contains(commit0.id()));
    assert!(!repo0_view.view().heads().contains(commit1.id()));
    assert!(repo1_view.view().heads().contains(commit1.id()));
    assert!(!repo1_view.view().heads().contains(commit0.id()));

    // Merge both back
    brevity::merge_agent_oplog(repo.loader(), repo_path, "agent-0")
        .block_on()
        .unwrap();
    brevity::merge_agent_oplog(repo.loader(), repo_path, "agent-1")
        .block_on()
        .unwrap();

    // Both commits should be visible in the shared repo
    let merged = repo.loader().load_at_head().block_on().unwrap();
    assert!(merged.view().heads().contains(commit0.id()));
    assert!(merged.view().heads().contains(commit1.id()));
}

#[test]
fn test_list_agent_oplogs() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path();

    // Initially empty
    let agents = brevity::list_agent_oplogs(repo_path).unwrap();
    assert!(agents.is_empty());

    // Fork three agents
    for name in &["agent-2", "agent-0", "agent-1"] {
        brevity::fork_agent_oplog(repo_path, name, repo.op_heads_store().as_ref())
            .block_on()
            .unwrap();
    }

    let agents = brevity::list_agent_oplogs(repo_path).unwrap();
    assert_eq!(agents, vec!["agent-0", "agent-1", "agent-2"]);
}

#[test]
fn test_cleanup_agent_oplog() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path();

    brevity::fork_agent_oplog(repo_path, "agent-0", repo.op_heads_store().as_ref())
        .block_on()
        .unwrap();

    // Agent directory exists
    let agent_dir = repo_path
        .parent()
        .unwrap()
        .join("agent-oplogs")
        .join("agent-0");
    assert!(agent_dir.exists());

    // Cleanup
    brevity::cleanup_agent_oplog(repo_path, "agent-0").unwrap();
    assert!(!agent_dir.exists());

    // Cleaning up again should error
    let result = brevity::cleanup_agent_oplog(repo_path, "agent-0");
    assert!(result.is_err());
}

#[test]
fn test_agent_name_validation() {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path();

    // Empty name
    let result =
        brevity::fork_agent_oplog(repo_path, "", repo.op_heads_store().as_ref()).block_on();
    assert!(result.is_err());

    // Path traversal
    let result =
        brevity::fork_agent_oplog(repo_path, "../etc", repo.op_heads_store().as_ref()).block_on();
    assert!(result.is_err());

    // Slash
    let result = brevity::fork_agent_oplog(repo_path, "agent/bad", repo.op_heads_store().as_ref())
        .block_on();
    assert!(result.is_err());

    // Starting with underscore
    let result =
        brevity::fork_agent_oplog(repo_path, "_bad", repo.op_heads_store().as_ref()).block_on();
    assert!(result.is_err());

    // Valid names
    let result =
        brevity::fork_agent_oplog(repo_path, "agent-0", repo.op_heads_store().as_ref()).block_on();
    assert!(result.is_ok());

    let result =
        brevity::fork_agent_oplog(repo_path, "Agent_1", repo.op_heads_store().as_ref()).block_on();
    assert!(result.is_ok());
}
