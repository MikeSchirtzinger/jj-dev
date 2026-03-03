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

//! Brevity Ventures agent oplog lifecycle management.
//!
//! Provides per-agent operation log isolation using [`ForkedOpHeadsStore`].
//! Agents get private op-head pointers so they never contend on the shared
//! oplog, while operations themselves remain in the shared content-addressed
//! `op_store`.
//!
//! ## Disk layout
//!
//! ```text
//! .jj/
//!   repo/
//!     op_store/              (shared, immutable)
//!     op_heads/              (orchestrator's view)
//!   agent-oplogs/            (per-agent isolation)
//!     agent-0/
//!       op_heads/
//!         type  → "forked_op_heads_store"
//!         heads/{op-hex}
//! ```

#![expect(missing_docs)]

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;

use crate::forked_op_heads_store::ForkedOpHeadsStore;
use crate::forked_op_heads_store::ForkedOpHeadsStoreInitError;
use crate::op_heads_store::OpHeadsStore;
use crate::op_heads_store::OpHeadsStoreError;
use crate::op_store::OpStoreError;
use crate::op_store::OperationId;
use crate::operation::Operation;
use crate::repo::RepoLoader;
use crate::repo::RepoLoaderError;

#[derive(Debug, Error)]
pub enum BrevityError {
    #[error("Agent oplog already exists: {0}")]
    AlreadyExists(String),
    #[error("Agent oplog not found: {0}")]
    NotFound(String),
    #[error("Invalid agent name: {0}")]
    InvalidName(String),
    #[error("Failed to fork agent oplog")]
    Fork(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("IO error")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Init(#[from] ForkedOpHeadsStoreInitError),
    #[error(transparent)]
    OpHeadsStore(#[from] OpHeadsStoreError),
    #[error(transparent)]
    RepoLoader(#[from] RepoLoaderError),
    #[error(transparent)]
    OpStore(#[from] OpStoreError),
}

/// Validate agent name: must be `[a-zA-Z0-9][a-zA-Z0-9_-]*` (no path
/// traversal).
fn validate_agent_name(name: &str) -> Result<(), BrevityError> {
    if name.is_empty() {
        return Err(BrevityError::InvalidName(
            "agent name must not be empty".into(),
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(BrevityError::InvalidName(format!(
            "agent name contains invalid characters: {name}"
        )));
    }
    if !name.chars().next().unwrap().is_ascii_alphanumeric() {
        return Err(BrevityError::InvalidName(format!(
            "agent name must start with alphanumeric character: {name}"
        )));
    }
    Ok(())
}

/// Root directory for all agent oplogs: `.jj/agent-oplogs/`.
///
/// `repo_path` is the `.jj/repo` directory.
fn agent_oplogs_dir(repo_path: &Path) -> PathBuf {
    repo_path
        .parent()
        .expect("repo_path should have a parent (.jj)")
        .join("agent-oplogs")
}

/// Path to a specific agent's `op_heads` directory.
fn agent_op_heads_path(repo_path: &Path, agent_name: &str) -> PathBuf {
    agent_oplogs_dir(repo_path)
        .join(agent_name)
        .join("op_heads")
}

/// Fork current op heads into a per-agent private store.
///
/// Creates `.jj/agent-oplogs/{agent_name}/op_heads/` with a copy of the
/// source store's current heads. The shared `op_store` (operations + views)
/// is NOT copied — it is shared and content-addressed.
pub async fn fork_agent_oplog(
    repo_path: &Path,
    agent_name: &str,
    source: &dyn OpHeadsStore,
) -> Result<ForkedOpHeadsStore, BrevityError> {
    validate_agent_name(agent_name)?;

    let agent_dir = agent_oplogs_dir(repo_path).join(agent_name);
    if agent_dir.exists() {
        return Err(BrevityError::AlreadyExists(agent_name.to_string()));
    }

    // Create parent directories
    fs::create_dir_all(agent_oplogs_dir(repo_path))?;
    fs::create_dir(&agent_dir)?;

    let op_heads_dir = agent_dir.join("op_heads");
    fs::create_dir(&op_heads_dir)?;

    // Get current heads from the source store
    let current_heads: Vec<OperationId> = source.get_op_heads().await?;

    // Initialize the forked store with the current heads
    let forked_store = ForkedOpHeadsStore::init_from(&op_heads_dir, &current_heads)?;

    // Write the type file for StoreFactories dispatch
    fs::write(op_heads_dir.join("type"), ForkedOpHeadsStore::name())?;

    Ok(forked_store)
}

/// Build a `RepoLoader` that uses an agent's forked op_heads_store.
///
/// Shares `store`, `op_store`, `index_store`, and `submodule_store` with the
/// base loader. Only the `op_heads_store` is substituted.
pub fn agent_repo_loader(
    base_loader: &RepoLoader,
    repo_path: &Path,
    agent_name: &str,
) -> Result<RepoLoader, BrevityError> {
    validate_agent_name(agent_name)?;

    let op_heads_dir = agent_op_heads_path(repo_path, agent_name);
    if !op_heads_dir.exists() {
        return Err(BrevityError::NotFound(agent_name.to_string()));
    }

    let forked_store = ForkedOpHeadsStore::load(&op_heads_dir);
    Ok(RepoLoader::new(
        base_loader.settings().clone(),
        base_loader.store().clone(),
        base_loader.op_store().clone(),
        Arc::new(forked_store),
        base_loader.index_store().clone(),
        base_loader.submodule_store().clone(),
    ))
}

/// Merge an agent's final op head(s) back into the shared store.
///
/// Uses `RepoLoader::merge_operations()` to create a merge operation, then
/// publishes it to the shared `op_heads_store`.
pub async fn merge_agent_oplog(
    base_loader: &RepoLoader,
    repo_path: &Path,
    agent_name: &str,
) -> Result<Operation, BrevityError> {
    validate_agent_name(agent_name)?;

    let op_heads_dir = agent_op_heads_path(repo_path, agent_name);
    if !op_heads_dir.exists() {
        return Err(BrevityError::NotFound(agent_name.to_string()));
    }

    // Load agent's forked op heads
    let forked_store = ForkedOpHeadsStore::load(&op_heads_dir);
    let agent_head_ids = forked_store.get_op_heads().await?;

    // Load shared store's current op heads
    let shared_store = base_loader.op_heads_store();
    let shared_head_ids = shared_store.get_op_heads().await?;

    // Load all operations (agent + shared)
    let mut all_ops = Vec::new();
    for id in &shared_head_ids {
        all_ops.push(base_loader.load_operation(id)?);
    }
    for id in &agent_head_ids {
        // Skip if already in shared (no-op fork with no agent work)
        if !shared_head_ids.contains(id) {
            all_ops.push(base_loader.load_operation(id)?);
        }
    }

    // If only shared heads (agent didn't diverge), just return the current head
    if all_ops.len() <= shared_head_ids.len() {
        return Ok(all_ops.into_iter().next().unwrap());
    }

    // Merge all operations
    let description = format!("merge agent {agent_name} oplog");
    let merged_op = base_loader.merge_operations(all_ops, Some(&description))?;

    // Publish to the shared op_heads_store
    let _lock = shared_store.lock().await?;
    shared_store
        .update_op_heads(&shared_head_ids, merged_op.id())
        .await?;

    Ok(merged_op)
}

/// List all agent oplog directories.
pub fn list_agent_oplogs(repo_path: &Path) -> Result<Vec<String>, BrevityError> {
    let dir = agent_oplogs_dir(repo_path);
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut agents = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && let Some(name) = entry.file_name().to_str()
        {
            agents.push(name.to_string());
        }
    }
    agents.sort();
    Ok(agents)
}

/// Remove an agent's oplog directory after successful merge.
pub fn cleanup_agent_oplog(repo_path: &Path, agent_name: &str) -> Result<(), BrevityError> {
    validate_agent_name(agent_name)?;

    let agent_dir = agent_oplogs_dir(repo_path).join(agent_name);
    if !agent_dir.exists() {
        return Err(BrevityError::NotFound(agent_name.to_string()));
    }

    fs::remove_dir_all(&agent_dir)?;
    Ok(())
}
