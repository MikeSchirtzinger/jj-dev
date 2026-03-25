// Copyright 2020 The Jujutsu Authors
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

use std::collections::HashMap;
use std::io;
use std::io::Read as _;
use std::iter;

use clap_complete::ArgValueCompleter;
use futures::TryStreamExt as _;
use futures::future::try_join_all;
use itertools::Itertools as _;
use jj_lib::object_id::ObjectId as _;
use jj_lib::repo::Repo as _;
use jj_lib::revset::RevsetStreamExt as _;
use tracing::instrument;

use crate::cli_util::CommandHelper;
use crate::cli_util::RevisionArg;
use crate::command_error::CommandError;
use crate::command_error::user_error;
use crate::complete;
use crate::description_util::ParsedBulkEditMessage;
use crate::description_util::add_trailers_with_template;
use crate::description_util::description_template;
use crate::description_util::edit_description;
use crate::description_util::edit_multiple_descriptions;
use crate::description_util::join_message_paragraphs;
use crate::description_util::parse_trailers_template;
use crate::text_util::complete_newline;
use crate::ui::Ui;

/// Update the change description or other metadata [default alias: desc]
///
/// Starts an editor to let you edit the description of changes. The editor
/// will be $EDITOR, or `nano` if that's not defined (`Notepad` on Windows).
#[derive(clap::Args, Clone, Debug)]
pub(crate) struct DescribeArgs {
    /// The revision(s) whose description to edit (default: @) [aliases: -r]
    #[arg(value_name = "REVSETS")]
    #[arg(add = ArgValueCompleter::new(complete::revset_expression_mutable))]
    revisions_pos: Vec<RevisionArg>,

    #[arg(short = 'r', hide = true, value_name = "REVSETS")]
    #[arg(add = ArgValueCompleter::new(complete::revset_expression_mutable))]
    revisions_opt: Vec<RevisionArg>,

    /// The change description to use (don't open editor)
    ///
    /// If multiple revisions are specified, the same description will be used
    /// for all of them.
    #[arg(
        long = "message",
        short,
        value_name = "MESSAGE",
        conflicts_with = "stdin"
    )]
    message_paragraphs: Option<Vec<String>>,

    /// Read the change description from stdin
    ///
    /// If multiple revisions are specified, the same description will be used
    /// for all of them.
    #[arg(long)]
    stdin: bool,

    /// Open an editor to edit the change description
    ///
    /// Forces an editor to open when using `--stdin` or `--message` to
    /// allow the message to be edited afterwards.
    #[arg(long)]
    editor: bool,

    /// Set Hox priority (critical, high, medium, low)
    #[arg(long, value_name = "PRIORITY")]
    set_priority: Option<String>,

    /// Set Hox status (open, in_progress, blocked, review, done, abandoned)
    #[arg(long, value_name = "STATUS")]
    set_status: Option<String>,

    /// Set Hox agent identifier
    #[arg(long, value_name = "AGENT")]
    set_agent: Option<String>,

    /// Set Hox orchestrator identifier
    #[arg(long, value_name = "ORCHESTRATOR")]
    set_orchestrator: Option<String>,

    /// Set message target (supports wildcards like O-A-*)
    #[arg(long, value_name = "TARGET")]
    set_msg_to: Option<String>,

    /// Set message type (mutation, info, align_request)
    #[arg(long, value_name = "TYPE")]
    set_msg_type: Option<String>,

    /// Set loop iteration number
    #[arg(long, value_name = "ITERATION")]
    set_loop_iteration: Option<u32>,

    /// Set max loop iterations
    #[arg(long, value_name = "MAX_ITERATIONS")]
    set_loop_max_iterations: Option<u32>,

    /// Write metadata to the operation store without advancing op heads.
    ///
    /// This is only valid with Hox metadata flags. The unpublished operation ID
    /// is printed so an orchestrator can explicitly integrate it later.
    #[arg(
        long,
        conflicts_with_all = ["editor", "stdin", "message_paragraphs"]
    )]
    metadata_only: bool,
}

#[instrument(skip_all)]
pub(crate) async fn cmd_describe(
    ui: &mut Ui,
    command: &CommandHelper,
    args: &DescribeArgs,
) -> Result<(), CommandError> {
    let mut workspace_command = command.workspace_helper(ui).await?;
    let target_expr = if !args.revisions_pos.is_empty() || !args.revisions_opt.is_empty() {
        workspace_command
            .parse_union_revsets(ui, &[&*args.revisions_pos, &*args.revisions_opt].concat())?
    } else {
        workspace_command.parse_revset(ui, &RevisionArg::AT)?
    }
    .resolve()?;
    workspace_command
        .check_rewritable_expr(&target_expr)
        .await?;
    let commits: Vec<_> = target_expr
        .evaluate(workspace_command.repo().as_ref())?
        .stream()
        .commits(workspace_command.repo().store()) // in reverse topological order
        .try_collect()
        .await?;
    if commits.is_empty() {
        writeln!(ui.status(), "No revisions to describe.")?;
        return Ok(());
    }
    let text_editor = workspace_command.text_editor()?;

    let mut tx = workspace_command.start_transaction();
    let tx_description = match commits.as_slice() {
        [] => unreachable!(),
        [commit] => format!("describe commit {}", commit.id().hex()),
        [first_commit, remaining_commits @ ..] => {
            format!(
                "describe commit {} and {} more",
                first_commit.id().hex(),
                remaining_commits.len()
            )
        }
    };

    let shared_description = if args.stdin {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        Some(complete_newline(buffer))
    } else {
        args.message_paragraphs
            .as_deref()
            .map(join_message_paragraphs)
    };

    let hox_priority = if let Some(priority) = &args.set_priority {
        let value = match priority.to_lowercase().as_str() {
            "critical" => 0,
            "high" => 1,
            "medium" => 2,
            "low" => 3,
            _ => {
                return Err(user_error(format!(
                    "Invalid priority: {priority}. Use: critical, high, medium, low"
                )));
            }
        };
        Some(value)
    } else {
        None
    };

    let hox_status = if let Some(status) = &args.set_status {
        let valid = [
            "open",
            "in_progress",
            "blocked",
            "review",
            "done",
            "abandoned",
        ];
        if !valid.contains(&status.as_str()) {
            return Err(user_error(format!(
                "Invalid status: {status}. Use: {}",
                valid.join(", ")
            )));
        }
        Some(status.clone())
    } else {
        None
    };

    let hox_msg_type = if let Some(msg_type) = &args.set_msg_type {
        let valid = ["mutation", "info", "align_request"];
        if !valid.contains(&msg_type.as_str()) {
            return Err(user_error(format!(
                "Invalid message type: {msg_type}. Use: {}",
                valid.join(", ")
            )));
        }
        Some(msg_type.clone())
    } else {
        None
    };

    let has_hox_changes = hox_priority.is_some()
        || hox_status.is_some()
        || args.set_agent.is_some()
        || args.set_orchestrator.is_some()
        || args.set_msg_to.is_some()
        || hox_msg_type.is_some()
        || args.set_loop_iteration.is_some()
        || args.set_loop_max_iterations.is_some();

    let mut commit_builders = commits
        .iter()
        .map(|commit| {
            let mut commit_builder = tx.repo_mut().rewrite_commit(commit).detach();
            if let Some(description) = &shared_description {
                commit_builder.set_description(description);
            }
            if let Some(priority) = hox_priority {
                commit_builder.set_priority(Some(priority));
            }
            if let Some(status) = &hox_status {
                commit_builder.set_status(Some(status.clone()));
            }
            if let Some(agent) = &args.set_agent {
                commit_builder.set_agent(Some(agent.clone()));
            }
            if let Some(orchestrator) = &args.set_orchestrator {
                commit_builder.set_orchestrator(Some(orchestrator.clone()));
            }
            if let Some(msg_to) = &args.set_msg_to {
                commit_builder.set_msg_to(Some(msg_to.clone()));
            }
            if let Some(msg_type) = &hox_msg_type {
                commit_builder.set_msg_type(Some(msg_type.clone()));
            }
            if let Some(iteration) = args.set_loop_iteration {
                commit_builder.set_iteration(Some(iteration));
            }
            if let Some(max_iterations) = args.set_loop_max_iterations {
                commit_builder.set_max_iterations(Some(max_iterations));
            }
            commit_builder
        })
        .collect_vec();

    let use_editor = !args.metadata_only && (args.editor || shared_description.is_none());

    if let Some(trailer_template) = parse_trailers_template(ui, &tx)? {
        for commit_builder in &mut commit_builders {
            // The first trailer would become the first line of the description.
            // Also, a commit with no description is treated in a special way in jujutsu: it
            // can be discarded as soon as it's no longer the working copy. Adding a
            // trailer to an empty description would break that logic.
            if use_editor || !commit_builder.description().is_empty() {
                let temp_commit = commit_builder.write_hidden().await?;
                let new_description = add_trailers_with_template(&trailer_template, &temp_commit)?;
                commit_builder.set_description(new_description);
            }
        }
    }

    if use_editor {
        let temp_commits: Vec<_> = try_join_all(
            iter::zip(&commits, &commit_builders)
                // Edit descriptions in topological order
                .rev()
                .map(async |(commit, commit_builder)| {
                    commit_builder
                        .write_hidden()
                        .await
                        .map(|temp_commit| (commit.id(), temp_commit))
                }),
        )
        .await?;

        if let [(_, temp_commit)] = &*temp_commits {
            let intro = "";
            let template = description_template(ui, &tx, intro, temp_commit)?;
            let description = edit_description(&text_editor, &template)?;
            commit_builders[0].set_description(description);
        } else {
            let ParsedBulkEditMessage {
                descriptions,
                missing,
                duplicates,
                unexpected,
            } = edit_multiple_descriptions(ui, &text_editor, &tx, &temp_commits)?;
            if !missing.is_empty() {
                return Err(user_error(format!(
                    "The description for the following commits were not found in the edited \
                     message: {}",
                    missing.join(", ")
                )));
            }
            if !duplicates.is_empty() {
                return Err(user_error(format!(
                    "The following commits were found in the edited message multiple times: {}",
                    duplicates.join(", ")
                )));
            }
            if !unexpected.is_empty() {
                return Err(user_error(format!(
                    "The following commits were not being edited, but were found in the edited \
                     message: {}",
                    unexpected.join(", ")
                )));
            }

            for (commit, commit_builder) in iter::zip(&commits, &mut commit_builders) {
                let description = descriptions.get(commit.id()).unwrap();
                commit_builder.set_description(description);
            }
        }
    }

    // Filter out unchanged commits to avoid rebasing descendants in
    // `transform_descendants` below unnecessarily.
    let commit_builders: HashMap<_, _> = iter::zip(&commits, commit_builders)
        .filter(|(old_commit, commit_builder)| {
            old_commit.description() != commit_builder.description() || has_hox_changes
        })
        .map(|(old_commit, commit_builder)| (old_commit.id(), commit_builder))
        .collect();

    let mut num_described = 0;
    let mut num_reparented = 0;
    // Even though `MutableRepo::rewrite_commit` and
    // `MutableRepo::rebase_descendants` can handle rewriting of a commit even
    // if it is a descendant of another commit being rewritten, using
    // `MutableRepo::transform_descendants` prevents us from rewriting the same
    // commit multiple times, and adding additional entries in the predecessor
    // chain.
    tx.repo_mut()
        .transform_descendants(
            commit_builders.keys().map(|&id| id.clone()).collect(),
            async |rewriter| {
                let old_commit_id = rewriter.old_commit().id().clone();
                let commit_builder = rewriter.reparent();
                if let Some(temp_builder) = commit_builders.get(&old_commit_id) {
                    commit_builder
                        .set_description(temp_builder.description())
                        .set_priority(temp_builder.priority())
                        .set_status(temp_builder.status().map(str::to_owned))
                        .set_agent(temp_builder.agent().map(str::to_owned))
                        .set_orchestrator(temp_builder.orchestrator().map(str::to_owned))
                        .set_msg_to(temp_builder.msg_to().map(str::to_owned))
                        .set_msg_type(temp_builder.msg_type().map(str::to_owned))
                        .set_iteration(temp_builder.iteration())
                        .set_max_iterations(temp_builder.max_iterations())
                        .write()
                        .await?;
                    num_described += 1;
                } else {
                    commit_builder.write().await?;
                    num_reparented += 1;
                }
                Ok(())
            },
        )
        .await?;
    if num_described > 1 {
        writeln!(ui.status(), "Updated {num_described} commits")?;
    }
    if num_reparented > 0 {
        writeln!(ui.status(), "Rebased {num_reparented} descendant commits")?;
    }
    if args.metadata_only {
        if !has_hox_changes {
            return Err(user_error(
                "--metadata-only requires at least one Hox metadata flag \
                 (--set-status, --set-priority, --set-agent, etc.)",
            ));
        }
        // Commit rewrites leave pending descendant rebases. Rebase them before
        // writing the unpublished operation so Transaction::write() remains
        // valid without publishing this operation to other workspaces.
        let mut inner_tx = tx.into_inner();
        inner_tx.repo_mut().rebase_descendants().await?;
        let unpublished = inner_tx.write(tx_description).await?;
        let op_id = unpublished.operation().id().hex();
        unpublished.leave_unpublished();
        writeln!(ui.status(), "Metadata-only op: {op_id}")?;
    } else {
        tx.finish(ui, tx_description).await?;
    }
    Ok(())
}
