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

//! Per-agent forked op_heads store for parallel multi-agent environments.
//!
//! [`ForkedOpHeadsStore`] gives each agent workspace its own isolated view of
//! the operation log head pointer. Agent writes advance only the agent's
//! private head; the shared (parent) store is never touched. The orchestrator
//! merges agent heads back into the main store at task completion.
//!
//! ## Lifecycle
//!
//! ```text
//! Orchestrator                           Agent Workspace
//!      |                                       |
//!      |-- ForkedOpHeadsStore::fork_from() --->|
//!      |   (copies main head into agent dir)   |
//!      |                                       |-- all writes -> agent's store
//!      |                                       |-- reads -> agent's store
//!      |                                       |
//!      |<-- task completion signal ------------|
//!      |-- merge_agent_heads_into_main() ----->| (orchestrator controls this)
//! ```

#![expect(missing_docs)]

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::path::Path;
use std::path::PathBuf;

use async_trait::async_trait;
use pollster::FutureExt as _;

use crate::hex_util;
use crate::object_id::ObjectId as _;
use crate::op_heads_store::OpHeadsStore;
use crate::op_heads_store::OpHeadsStoreError;
use crate::op_heads_store::OpHeadsStoreLock;
use crate::op_store::OperationId;
use crate::simple_op_heads_store::SimpleOpHeadsStore;

/// An [`OpHeadsStore`] that is forked from a parent store at a specific op.
///
/// Reads and writes go to the agent's private store directory. The parent
/// store is never written by this type — agent isolation is enforced at the
/// store boundary.
///
/// The `fork_base` file records the op ID at which the fork was created,
/// enabling the orchestrator to compute correct merge ancestry when
/// re-integrating the agent's work.
pub struct ForkedOpHeadsStore {
    /// The agent's private store (reads and writes).
    private: SimpleOpHeadsStore,
    /// Absolute path to the agent's op_heads directory (for debug output).
    dir: PathBuf,
    /// The op ID at which this store was forked from the parent.
    fork_base: OperationId,
}

impl Debug for ForkedOpHeadsStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForkedOpHeadsStore")
            .field("dir", &self.dir)
            .field("fork_base", &self.fork_base.hex())
            .finish()
    }
}

impl ForkedOpHeadsStore {
    pub fn name() -> &'static str {
        "forked_op_heads_store"
    }

    /// Fork from a parent store at the given operation.
    ///
    /// Copies the current op heads from `parent` into a new private store at
    /// `agent_op_heads_dir`. The `fork_op_id` is recorded in a `fork_base`
    /// file so that `merge_back` can compute correct merge ancestry.
    ///
    /// # Errors
    /// Returns an error if `agent_op_heads_dir` already exists or if any
    /// filesystem operation fails.
    pub fn fork_from(
        parent: &dyn OpHeadsStore,
        fork_op_id: OperationId,
        agent_op_heads_dir: &Path,
    ) -> Result<Self, OpHeadsStoreError> {
        // Ensure the agent dir exists before initializing the private store.
        std::fs::create_dir_all(agent_op_heads_dir).map_err(|e| OpHeadsStoreError::Write {
            new_op_id: fork_op_id.clone(),
            source: e.into(),
        })?;

        // Initialize the private store — this creates `agent_op_heads_dir/heads/`.
        let private = SimpleOpHeadsStore::init(agent_op_heads_dir)
            .map_err(|e| OpHeadsStoreError::Read(e.into()))?;

        // Copy current parent heads into the private store.
        let parent_heads = parent.get_op_heads().block_on()?;
        for head_id in &parent_heads {
            private.update_op_heads(&[], head_id).block_on()?;
        }

        // Write the fork_base file.
        let fork_base_path = agent_op_heads_dir.join("fork_base");
        std::fs::write(&fork_base_path, fork_op_id.hex()).map_err(|e| {
            OpHeadsStoreError::Write {
                new_op_id: fork_op_id.clone(),
                source: e.into(),
            }
        })?;

        Ok(Self {
            private,
            dir: agent_op_heads_dir.to_path_buf(),
            fork_base: fork_op_id,
        })
    }

    /// Load an existing forked store from disk.
    ///
    /// Use this when an agent process restarts and needs to reload its state.
    pub fn load(agent_op_heads_dir: &Path) -> Result<Self, OpHeadsStoreError> {
        let private = SimpleOpHeadsStore::load(agent_op_heads_dir);

        let fork_base_path = agent_op_heads_dir.join("fork_base");
        let fork_base_hex = std::fs::read_to_string(&fork_base_path)
            .map_err(|e| OpHeadsStoreError::Read(e.into()))?;
        let fork_base_bytes = hex_util::decode_hex(fork_base_hex.trim()).ok_or_else(|| {
            OpHeadsStoreError::Read(format!("invalid fork_base hex in {fork_base_path:?}").into())
        })?;
        let fork_base = OperationId::new(fork_base_bytes);

        Ok(Self {
            private,
            dir: agent_op_heads_dir.to_path_buf(),
            fork_base,
        })
    }

    /// Returns the op_id at which this store was forked from the parent.
    ///
    /// The orchestrator uses this as the merge base when re-integrating the
    /// agent's operations into the main store.
    pub fn fork_base(&self) -> &OperationId {
        &self.fork_base
    }

    /// Returns the path to this agent's op_heads directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Returns the agent's current op heads (may differ from the parent store).
    pub async fn agent_op_heads(&self) -> Result<Vec<OperationId>, OpHeadsStoreError> {
        self.private.get_op_heads().await
    }
}

#[async_trait]
impl OpHeadsStore for ForkedOpHeadsStore {
    fn name(&self) -> &str {
        Self::name()
    }

    /// Update op heads in the agent's private store only.
    ///
    /// The parent store is never touched — full agent isolation guaranteed.
    async fn update_op_heads(
        &self,
        old_ids: &[OperationId],
        new_id: &OperationId,
    ) -> Result<(), OpHeadsStoreError> {
        self.private.update_op_heads(old_ids, new_id).await
    }

    async fn get_op_heads(&self) -> Result<Vec<OperationId>, OpHeadsStoreError> {
        self.private.get_op_heads().await
    }

    async fn lock(&self) -> Result<Box<dyn OpHeadsStoreLock + '_>, OpHeadsStoreError> {
        self.private.lock().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_op_id(hex: &str) -> OperationId {
        let bytes = hex_util::decode_hex(hex).expect("valid hex");
        OperationId::new(bytes)
    }

    #[test]
    fn test_fork_copies_parent_heads() {
        let parent_dir = tempfile::tempdir().unwrap();
        let agent_dir = tempfile::tempdir().unwrap();

        // Create a parent store with one head.
        let parent = SimpleOpHeadsStore::init(parent_dir.path()).unwrap();
        let op_id = make_op_id("aabbccdd00000000000000000000000000000000000000000000000000000000");
        parent.update_op_heads(&[], &op_id).block_on().unwrap();

        // Fork from the parent.
        let forked =
            ForkedOpHeadsStore::fork_from(&parent, op_id.clone(), agent_dir.path()).unwrap();

        // The forked store should have the same heads as the parent.
        let agent_heads = forked.agent_op_heads().block_on().unwrap();
        assert_eq!(agent_heads.len(), 1);
        assert_eq!(agent_heads[0], op_id);

        // The fork_base should match the provided op_id.
        assert_eq!(forked.fork_base(), &op_id);
    }

    #[test]
    fn test_agent_writes_do_not_affect_parent() {
        let parent_dir = tempfile::tempdir().unwrap();
        let agent_dir = tempfile::tempdir().unwrap();

        let parent = SimpleOpHeadsStore::init(parent_dir.path()).unwrap();
        let base_id =
            make_op_id("aabbccdd00000000000000000000000000000000000000000000000000000000");
        parent.update_op_heads(&[], &base_id).block_on().unwrap();

        let forked =
            ForkedOpHeadsStore::fork_from(&parent, base_id.clone(), agent_dir.path()).unwrap();

        // Agent writes a new op.
        let new_id = make_op_id("1122334400000000000000000000000000000000000000000000000000000000");
        forked
            .update_op_heads(std::slice::from_ref(&base_id), &new_id)
            .block_on()
            .unwrap();

        // Parent store should still have only base_id.
        let parent_heads = parent.get_op_heads().block_on().unwrap();
        assert_eq!(parent_heads, vec![base_id]);

        // Agent store should have only new_id.
        let agent_heads = forked.get_op_heads().block_on().unwrap();
        assert_eq!(agent_heads, vec![new_id]);
    }

    #[test]
    fn test_load_roundtrip() {
        let agent_dir = tempfile::tempdir().unwrap();
        let parent_dir = tempfile::tempdir().unwrap();

        let parent = SimpleOpHeadsStore::init(parent_dir.path()).unwrap();
        let op_id = make_op_id("deadbeef00000000000000000000000000000000000000000000000000000000");
        parent.update_op_heads(&[], &op_id).block_on().unwrap();

        // Create fork.
        let _forked =
            ForkedOpHeadsStore::fork_from(&parent, op_id.clone(), agent_dir.path()).unwrap();

        // Reload from disk.
        let reloaded = ForkedOpHeadsStore::load(agent_dir.path()).unwrap();
        assert_eq!(reloaded.fork_base(), &op_id);

        let heads = reloaded.get_op_heads().block_on().unwrap();
        assert_eq!(heads.len(), 1);
        assert_eq!(heads[0], op_id);
    }
}
