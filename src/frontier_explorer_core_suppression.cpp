/*
Copyright 2026 Mert Güler

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "frontier_exploration_ros2/frontier_explorer_core.hpp"

#include <cmath>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace frontier_exploration_ros2
{

bool attributed_watchdog_stop_requests_frontier_block(
  const std::string & payload,
  int stop_threshold)
{
  if (stop_threshold <= 0) {
    return false;
  }
  const std::regex event_pattern(
    "\"event\"\\s*:\\s*\"(collision_stop|depth_guard_stop)\"");
  if (!std::regex_search(payload, event_pattern)) {
    return false;
  }
  const std::regex count_pattern(
    "\"(recent_safety_stops|recent_collision_stops)\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  int count = 1;
  if (std::regex_search(payload, match, count_pattern) && match.size() >= 3) {
    try {
      count = std::stoi(match[2].str());
    } catch (const std::exception &) {
      count = 1;
    }
  }
  return count >= stop_threshold;
}

bool FrontierExplorerCore::suppression_enabled() const
{
  return params.frontier_suppression_enabled;
}

bool FrontierExplorerCore::suppression_runtime_active(int64_t now_ns) const
{
  return (
    suppression_enabled() &&
    (!frontier_suppression_activation_ns_.has_value() || now_ns >= *frontier_suppression_activation_ns_));
}

bool FrontierExplorerCore::should_return_to_start_when_all_frontiers_suppressed() const
{
  return params.all_frontiers_suppressed_behavior == "return_to_start";
}

bool FrontierExplorerCore::should_complete_when_all_frontiers_suppressed() const
{
  return params.all_frontiers_suppressed_behavior == "complete";
}

FrontierSuppression * FrontierExplorerCore::ensure_frontier_suppression()
{
  if (!suppression_enabled()) {
    return nullptr;
  }
  if (!frontier_suppression_) {
    frontier_suppression_ = std::make_unique<FrontierSuppression>(FrontierSuppressionConfig{
      params.frontier_suppression_attempt_threshold,
      params.frontier_suppression_base_size_m,
      params.frontier_suppression_expansion_size_m,
      params.frontier_suppression_timeout_s,
      params.frontier_suppression_no_progress_timeout_s,
      params.frontier_suppression_progress_epsilon_m,
      params.frontier_suppression_max_attempt_records,
      params.frontier_suppression_max_regions,
      params.frontier_visit_tolerance,
    });
  }
  return frontier_suppression_.get();
}

FrontierSequence FrontierExplorerCore::filter_frontiers_for_suppression(
  const FrontierSequence & frontiers)
{
  const int64_t now_ns = callbacks.now_ns();
  if (!suppression_runtime_active(now_ns)) {
    return frontiers;
  }
  FrontierSuppression * suppression = ensure_frontier_suppression();
  if (!suppression) {
    return frontiers;
  }
  suppression->prune_expired(now_ns);
  return suppression->filter_frontiers(frontiers);
}

void FrontierExplorerCore::record_failed_frontier_attempt(
  const std::optional<FrontierLike> & frontier)
{
  const int64_t now_ns = callbacks.now_ns();
  if (!suppression_runtime_active(now_ns) || !frontier.has_value()) {
    return;
  }
  FrontierSuppression * suppression = ensure_frontier_suppression();
  if (suppression) {
    suppression->record_failed_attempt(
      *frontier,
      now_ns,
      callbacks.log_warn);
  }
}

void FrontierExplorerCore::suppress_failed_frontier_goal(
  const std::optional<FrontierLike> & frontier,
  const FrontierSequence & frontier_sequence,
  const std::string & reason,
  bool bypass_startup_grace)
{
  if (frontier.has_value()) {
    suppress_blocked_frontier_region(*frontier, reason, bypass_startup_grace);
    return;
  }

  if (!frontier_sequence.empty()) {
    suppress_blocked_frontier_region(
      frontier_sequence.front(), reason, bypass_startup_grace);
  }
}

void FrontierExplorerCore::suppress_blocked_frontier_region(
  const FrontierLike & frontier,
  const std::string & reason,
  bool bypass_startup_grace)
{
  const int64_t now_ns = callbacks.now_ns();
  if (
    !suppression_enabled() ||
    (!bypass_startup_grace && !suppression_runtime_active(now_ns)))
  {
    return;
  }
  FrontierSuppression * suppression = ensure_frontier_suppression();
  if (!suppression) {
    return;
  }
  suppression->suppress_region(
    frontier,
    now_ns,
    callbacks.log_warn);
  callbacks.log_info(
    "Temporarily suppressing blocked frontier region: " + reason +
    "; " + describe_frontier(frontier));
}

void FrontierExplorerCore::clear_active_goal_progress_state()
{
  if (frontier_suppression_) {
    frontier_suppression_->clear_goal_progress_tracking();
  }
  clear_active_goal_euclidean_progress_state();
}

void FrontierExplorerCore::clear_active_goal_euclidean_progress_state()
{
  active_goal_best_euclidean_distance.reset();
  active_goal_last_euclidean_progress_ns.reset();
}

void FrontierExplorerCore::start_active_goal_progress_tracking()
{
  const int64_t now_ns = callbacks.now_ns();
  start_active_goal_euclidean_progress_tracking(now_ns);
  FrontierSuppression * suppression = ensure_frontier_suppression();
  if (suppression_runtime_active(now_ns) && suppression && active_goal_kind == "frontier") {
    suppression->start_goal_progress_tracking(current_dispatch_id, now_ns);
    return;
  }
  clear_active_goal_progress_state();
}

void FrontierExplorerCore::start_active_goal_euclidean_progress_tracking(int64_t now_ns)
{
  clear_active_goal_euclidean_progress_state();
  note_active_goal_euclidean_progress(now_ns);
}

void FrontierExplorerCore::note_active_goal_progress(double distance_remaining)
{
  if (frontier_suppression_) {
    frontier_suppression_->note_goal_progress(
      current_dispatch_id,
      distance_remaining,
      callbacks.now_ns());
  }
}

void FrontierExplorerCore::note_active_goal_euclidean_progress(int64_t now_ns)
{
  const auto target_point = active_goal_target_point();
  if (!target_point.has_value()) {
    return;
  }

  const auto current_pose = callbacks.get_current_pose();
  if (!current_pose.has_value()) {
    return;
  }

  const double distance_to_goal = std::hypot(
    target_point->first - current_pose->position.x,
    target_point->second - current_pose->position.y);
  if (!active_goal_best_euclidean_distance.has_value()) {
    active_goal_best_euclidean_distance = distance_to_goal;
    active_goal_last_euclidean_progress_ns = now_ns;
    return;
  }

  if (
    *active_goal_best_euclidean_distance - distance_to_goal >=
    params.frontier_suppression_progress_epsilon_m)
  {
    active_goal_best_euclidean_distance = distance_to_goal;
    active_goal_last_euclidean_progress_ns = now_ns;
  }
}

bool FrontierExplorerCore::active_goal_euclidean_progress_recent(int64_t now_ns) const
{
  if (!active_goal_last_euclidean_progress_ns.has_value()) {
    return false;
  }

  const int64_t timeout_ns =
    static_cast<int64_t>(params.frontier_suppression_no_progress_timeout_s * 1e9);
  return now_ns - *active_goal_last_euclidean_progress_ns < timeout_ns;
}

bool FrontierExplorerCore::evaluate_active_goal_progress_timeout()
{
  const int64_t now_ns = callbacks.now_ns();
  if (!suppression_runtime_active(now_ns)) {
    return false;
  }
  FrontierSuppression * suppression = ensure_frontier_suppression();
  if (!suppression ||
    active_goal_kind != "frontier" ||
    goal_state != GoalLifecycleState::ACTIVE ||
    !goal_handle ||
    cancel_request_in_progress)
  {
    return false;
  }
  suppression->prune_expired(now_ns);
  if (!suppression->is_tracking_dispatch(current_dispatch_id)) {
    suppression->start_goal_progress_tracking(current_dispatch_id, now_ns);
    return false;
  }
  note_active_goal_euclidean_progress(now_ns);
  if (active_goal_euclidean_progress_recent(now_ns)) {
    return false;
  }
  if (!suppression->mark_timeout_cancel_if_needed(current_dispatch_id, now_ns)) {
    return false;
  }
  request_active_goal_cancel(
    "Canceling frontier goal after no meaningful progress was observed within suppression timeout.");
  return true;
}

void FrontierExplorerCore::handle_all_frontiers_suppressed(
  const geometry_msgs::msg::Pose & current_pose)
{
  if (should_complete_when_all_frontiers_suppressed()) {
    if (!return_to_start_completed) {
      callbacks.log_info("All currently detected frontiers are suppressed; completing exploration");
      handle_exploration_complete(current_pose);
    }
    return;
  }

  if (
    !should_return_to_start_when_all_frontiers_suppressed() ||
    !start_pose.has_value() ||
    suppressed_return_to_start_started)
  {
    return;
  }

  if (is_pose_within_xy_tolerance(current_pose, start_pose->pose)) {
    callbacks.log_info("All frontiers remain suppressed; staying at the start pose while waiting");
    suppressed_return_to_start_started = true;
    return;
  }

  suppressed_return_to_start_started = true;
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2)
      << "All frontiers are temporarily suppressed, returning to start pose while waiting: ("
      << start_pose->pose.position.x << ", "
      << start_pose->pose.position.y << ")";

  if (!send_pose_goal(
      *start_pose,
      "suppressed_return_to_start",
      std::nullopt,
      {},
      oss.str()))
  {
    suppressed_return_to_start_started = false;
  }
}

void FrontierExplorerCore::consider_cancel_suppressed_return_to_start()
{
  if (
    !goal_in_progress ||
    active_goal_kind != "suppressed_return_to_start" ||
    !map.has_value() ||
    !costmap.has_value())
  {
    return;
  }

  const auto current_pose = callbacks.get_current_pose();
  if (!current_pose.has_value()) {
    return;
  }

  FrontierSnapshot snapshot;
  try {
    snapshot = get_frontier_snapshot(*current_pose, params.frontier_candidate_min_goal_distance_m);
  } catch (const std::out_of_range &) {
    return;
  }

  FrontierSequence filtered_frontiers = filter_frontiers_for_suppression(snapshot.frontiers);
  if (filtered_frontiers.empty()) {
    publish_frontier_markers(filtered_frontiers);
    return;
  }

  const auto selection = select_frontier(filtered_frontiers, *current_pose);
  publish_frontier_markers(filtered_frontiers);
  if (!selection.frontier.has_value()) {
    return;
  }

  const auto frontier_sequence = select_frontier_sequence(
    filtered_frontiers,
    *current_pose,
    selection.frontier);
  if (frontier_sequence.empty()) {
    return;
  }

  pending_frontier_sequence = frontier_sequence;
  pending_frontier_selection_mode = selection.mode;
  pending_frontier_dispatch_context = "suppression_cleared";
  suppressed_return_to_start_started = false;
  callbacks.log_info(
    "Frontiers are available again; preempting temporary return-to-start goal");
  dispatch_pending_frontier_goal(*current_pose);
}

}  // namespace frontier_exploration_ros2
