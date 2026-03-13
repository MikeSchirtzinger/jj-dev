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
use std::sync::Arc;

use jj_lib::forked_op_heads_store::ForkedOpHeadsStore;
use jj_lib::object_id::ObjectId as _;
use jj_lib::op_heads_store::OpHeadsStore as _;
use jj_lib::op_store::OperationId;
use jj_lib::repo::RepoLoader;
use jj_lib::repo::StoreFactories;
use jj_lib::simple_op_heads_store::SimpleOpHeadsStore;
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
fn test_forked_op_heads_fork_from() {
    let parent_dir = testutils::new_temp_dir();
    let agent_dir = testutils::new_temp_dir();

    // Create a parent store with one head.
    let parent = SimpleOpHeadsStore::init(parent_dir.path()).unwrap();
    let id1 = OperationId::from_hex("aabbccdd");
    parent.update_op_heads(&[], &id1).block_on().unwrap();

    let forked = ForkedOpHeadsStore::fork_from(&parent, id1.clone(), agent_dir.path()).unwrap();

    let heads: Vec<String> = forked
        .get_op_heads()
        .block_on()
        .unwrap()
        .iter()
        .map(|id| id.hex())
        .collect();
    assert!(heads.contains(&id1.hex()));
    assert_eq!(forked.fork_base(), &id1);
}

#[test]
fn test_forked_op_heads_update() {
    let parent_dir = testutils::new_temp_dir();
    let agent_dir = testutils::new_temp_dir();

    let parent = SimpleOpHeadsStore::init(parent_dir.path()).unwrap();
    let id1 = OperationId::from_hex("aabbccdd");
    parent.update_op_heads(&[], &id1).block_on().unwrap();

    let forked = ForkedOpHeadsStore::fork_from(&parent, id1.clone(), agent_dir.path()).unwrap();

    let id2 = OperationId::from_hex("11223344");
    forked
        .update_op_heads(std::slice::from_ref(&id1), &id2)
        .block_on()
        .unwrap();

    let heads: Vec<String> = forked
        .get_op_heads()
        .block_on()
        .unwrap()
        .iter()
        .map(|id| id.hex())
        .collect();
    assert_eq!(heads, vec![id2.hex()]);
}

#[test]
fn test_forked_op_heads_lock() {
    let parent_dir = testutils::new_temp_dir();
    let agent_dir = testutils::new_temp_dir();

    let parent = SimpleOpHeadsStore::init(parent_dir.path()).unwrap();
    let id1 = OperationId::from_hex("aabbccdd");
    parent.update_op_heads(&[], &id1).block_on().unwrap();

    let forked = ForkedOpHeadsStore::fork_from(&parent, id1.clone(), agent_dir.path()).unwrap();
    let _lock = forked.lock().block_on().unwrap();
}

#[test]
fn test_forked_op_heads_name() {
    assert_eq!(ForkedOpHeadsStore::name(), "forked_op_heads_store");
}

#[test]
fn test_forked_op_heads_load_roundtrip() {
    let parent_dir = testutils::new_temp_dir();
    let agent_dir = testutils::new_temp_dir();

    let parent = SimpleOpHeadsStore::init(parent_dir.path()).unwrap();
    let id1 = OperationId::from_hex("aabbccdd");
    parent.update_op_heads(&[], &id1).block_on().unwrap();

    let _forked = ForkedOpHeadsStore::fork_from(&parent, id1.clone(), agent_dir.path()).unwrap();

    // Reload from disk
    let reloaded = ForkedOpHeadsStore::load(agent_dir.path()).unwrap();
    assert_eq!(reloaded.fork_base(), &id1);

    let heads: Vec<String> = reloaded
        .get_op_heads()
        .block_on()
        .unwrap()
        .iter()
        .map(|id| id.hex())
        .collect();
    assert!(heads.contains(&id1.hex()));
}

#[test]
fn test_forked_store_registered_in_factories() {
    let factories = StoreFactories::default();

    // Write a type file and try loading
    let parent_dir = testutils::new_temp_dir();
    let agent_dir = testutils::new_temp_dir();

    // Create a parent, fork from it to get a valid forked store on disk
    let parent = SimpleOpHeadsStore::init(parent_dir.path()).unwrap();
    let id1 = OperationId::from_hex("aabbccdd");
    parent.update_op_heads(&[], &id1).block_on().unwrap();
    let _forked = ForkedOpHeadsStore::fork_from(&parent, id1.clone(), agent_dir.path()).unwrap();

    // Write the type file
    std::fs::write(agent_dir.path().join("type"), "forked_op_heads_store").unwrap();

    let settings = testutils::user_settings();
    let loaded = factories.load_op_heads_store(&settings, agent_dir.path());
    assert!(loaded.is_ok());
    let store = loaded.unwrap();
    assert_eq!(store.name(), "forked_op_heads_store");
}

#[test]
fn test_forked_op_heads_independent_of_shared() {
    // Create a real test repo, fork the op heads, commit via the forked store,
    // and verify the shared store is unaffected.
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let repo_path = test_repo.repo_path();

    // Get the current shared op head
    let shared_op_heads_dir = repo_path.join("op_heads").join("heads");
    let shared_heads_before = list_dir(&shared_op_heads_dir);
    assert_eq!(shared_heads_before.len(), 1);

    // Get the current shared head IDs
    let shared_head_ids: Vec<OperationId> =
        repo.op_heads_store().get_op_heads().block_on().unwrap();
    let fork_op_id = shared_head_ids.first().unwrap().clone();

    // Create a forked store using fork_from
    let forked_dir = test_repo.env.root().join("agent-0-op-heads");
    let forked_store =
        ForkedOpHeadsStore::fork_from(repo.op_heads_store().as_ref(), fork_op_id, &forked_dir)
            .unwrap();

    // Verify forked store has the same heads
    let forked_heads: Vec<String> = forked_store
        .get_op_heads()
        .block_on()
        .unwrap()
        .iter()
        .map(|id| id.hex())
        .collect();
    assert_eq!(forked_heads, shared_heads_before);

    // Create a RepoLoader with the forked store and load a repo from it
    let forked_loader = RepoLoader::new(
        repo.loader().settings().clone(),
        repo.loader().store().clone(),
        repo.loader().op_store().clone(),
        Arc::new(forked_store),
        repo.loader().index_store().clone(),
        repo.loader().submodule_store().clone(),
    );
    let forked_repo = forked_loader.load_at_head().unwrap();

    // Commit a transaction via the forked repo
    let mut tx = forked_repo.start_transaction();
    write_random_commit(tx.repo_mut());
    let forked_committed = tx.commit("forked transaction").unwrap();
    let forked_op_id = forked_committed.operation().id().hex();

    // The forked store should have the new op head
    let forked_heads_after = list_dir(&forked_dir.join("heads"));
    assert_eq!(forked_heads_after.len(), 1);
    assert_eq!(forked_heads_after[0], forked_op_id);

    // The shared store should be UNCHANGED
    let shared_heads_after = list_dir(&shared_op_heads_dir);
    assert_eq!(shared_heads_after, shared_heads_before);
}
