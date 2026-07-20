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

#include "frontier_explorer_core_detail.hpp"
#include "frontier_exploration_ros2/mrtsp_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace frontier_exploration_ros2
{

FrontierSequence FrontierExplorerCore::build_mrtsp_frontier_sequence(
  const FrontierSequence & frontiers,
  const geometry_msgs::msg::Pose & current_pose) const
{
  const FrontierSequence bounded_frontiers = filter_frontiers_for_boundary(frontiers);
  if (bounded_frontiers.empty()) {
    return {};
  }

  const auto apply_startup_heading_preference =
    [&](const FrontierSequence & sequence) -> FrontierSequence {
      if (params.startup_heading_preference_max_translation_m <= 0.0 ||
        !start_pose.has_value())
      {
        return sequence;
      }
      const double startup_translation_m = std::hypot(
        current_pose.position.x - start_pose->pose.position.x,
        current_pose.position.y - start_pose->pose.position.y);
      if (startup_translation_m > params.startup_heading_preference_max_translation_m) {
        return sequence;
      }

      const double current_yaw = detail::yaw_from_quaternion(current_pose.orientation);
      const auto heading_error = [&](const FrontierLike & frontier) {
          const auto [target_x, target_y] = frontier_position(frontier);
          const double target_yaw = std::atan2(
            target_y - current_pose.position.y,
            target_x - current_pose.position.x);
          return std::abs(std::atan2(
              std::sin(target_yaw - current_yaw),
              std::cos(target_yaw - current_yaw)));
        };

      FrontierSequence preferred = sequence;
      std::stable_sort(
        preferred.begin(),
        preferred.end(),
        [&](const FrontierLike & lhs, const FrontierLike & rhs) {
          return heading_error(lhs) < heading_error(rhs);
        });
      if (debug_outputs_enabled() && !preferred.empty() &&
        !are_frontiers_equivalent(preferred.front(), sequence.front()))
      {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << "Startup heading preference reordered initial frontier: translation="
            << startup_translation_m << "m, selected_heading_error="
            << heading_error(preferred.front()) << "rad";
        callbacks.log_debug(oss.str());
      }
      return preferred;
    };

  const double pose_quantum = std::max(params.frontier_visit_tolerance, 0.1);
  const int pose_x_bucket = detail::quantize_bucket(current_pose.position.x, pose_quantum);
  const int pose_y_bucket = detail::quantize_bucket(current_pose.position.y, pose_quantum);
  const int yaw_bucket = detail::quantize_bucket(
    detail::yaw_from_quaternion(current_pose.orientation),
    detail::kPi / 12.0);
  const FrontierSignature signature = frontier_signature(bounded_frontiers);

  // Solver mode and DP bounds participate in the cache key because the same frontier
  // geometry can yield a different order when the route horizon or candidate pool changes.
  if (mrtsp_order_cache.has_value() &&
    mrtsp_order_cache->frontier_signature == signature &&
    mrtsp_order_cache->pose_x_bucket == pose_x_bucket &&
    mrtsp_order_cache->pose_y_bucket == pose_y_bucket &&
    mrtsp_order_cache->yaw_bucket == yaw_bucket &&
    mrtsp_order_cache->sensor_effective_range_m == params.sensor_effective_range_m &&
    mrtsp_order_cache->weight_distance_wd == params.weight_distance_wd &&
    mrtsp_order_cache->weight_gain_ws == params.weight_gain_ws &&
    mrtsp_order_cache->max_linear_speed_vmax == params.max_linear_speed_vmax &&
    mrtsp_order_cache->max_angular_speed_wmax == params.max_angular_speed_wmax &&
    mrtsp_order_cache->mrtsp_solver == params.mrtsp_solver &&
    mrtsp_order_cache->dp_solver_candidate_limit == params.dp_solver_candidate_limit &&
    mrtsp_order_cache->dp_planning_horizon == params.dp_planning_horizon)
  {
    const_cast<FrontierExplorerCore *>(this)->mrtsp_order_cache_hits += 1;
    if (debug_outputs_enabled()) {
      callbacks.log_debug(
        "mrtsp_order_cache: hit, frontiers=" + std::to_string(bounded_frontiers.size()));
    }
    return apply_startup_heading_preference(mrtsp_order_cache->frontier_sequence);
  }

  const std::vector<FrontierCandidate> & candidates = bounded_frontiers;

  RobotState robot_state;
  robot_state.position = {current_pose.position.x, current_pose.position.y};
  robot_state.yaw = detail::yaw_from_quaternion(current_pose.orientation);

  CostWeights weights;
  weights.distance_wd = params.weight_distance_wd;
  weights.gain_ws = params.weight_gain_ws;

  std::vector<std::size_t> order;
  if (params.mrtsp_solver == "dp") {
    // DP mode keeps the matrix compact by scoring all candidates with the MRTSP start
    // cost, retaining the best pool, and evaluating only bounded-horizon sequences.
    const auto pruned = prune_mrtsp_candidates(
      candidates,
      robot_state,
      weights,
      params.sensor_effective_range_m,
      params.max_linear_speed_vmax,
      params.max_angular_speed_wmax,
      MrtspSolverConfig{
        params.dp_solver_candidate_limit,
        params.dp_planning_horizon});

    std::vector<FrontierCandidate> pruned_candidates;
    pruned_candidates.reserve(pruned.size());
    // build_cost_matrix() owns the canonical pairwise cost logic, so the core extracts
    // a compact candidate vector instead of asking the solver to duplicate matrix rules.
    for (const auto & item : pruned) {
      pruned_candidates.push_back(item.candidate);
    }

    const MrtspCostMatrix cost_matrix = build_cost_matrix(
      pruned_candidates,
      robot_state,
      weights,
      params.sensor_effective_range_m,
      params.max_linear_speed_vmax,
      params.max_angular_speed_wmax);
    std::vector<std::size_t> pruned_order = solve_bounded_horizon_mrtsp_order(
      cost_matrix,
      params.dp_planning_horizon);

    // If the bounded solver cannot form a finite route, the pruned matrix can still
    // provide a valid single-step ordering through the standard greedy traversal.
    if (pruned_order.empty()) {
      pruned_order = greedy_mrtsp_order(cost_matrix);
    }

    order.reserve(pruned_order.size());
    // Convert pruned-vector indices back to the candidate vector used by the core.
    for (const std::size_t pruned_index : pruned_order) {
      if (pruned_index < pruned.size()) {
        order.push_back(pruned[pruned_index].original_index);
      }
    }
  } else {
    // Greedy mode intentionally keeps the full candidate set to preserve the simple
    // MRTSP traversal behavior selected by the user-facing solver parameter.
    if (params.mrtsp_solver != "greedy") {
      callbacks.log_warn(
        "Unknown mrtsp_solver='" + params.mrtsp_solver + "'; falling back to greedy");
    }
    const MrtspCostMatrix cost_matrix = build_cost_matrix(
      candidates,
      robot_state,
      weights,
      params.sensor_effective_range_m,
      params.max_linear_speed_vmax,
      params.max_angular_speed_wmax);
    order = greedy_mrtsp_order(cost_matrix);
  }

  FrontierSequence ordered_frontiers;
  ordered_frontiers.reserve(bounded_frontiers.size());
  std::vector<uint8_t> ordered_indices(bounded_frontiers.size(), 0U);
  for (const std::size_t index : order) {
    if (index < bounded_frontiers.size() && !ordered_indices[index]) {
      ordered_frontiers.push_back(bounded_frontiers[index]);
      ordered_indices[index] = 1U;
    }
  }

  if (ordered_frontiers.size() < bounded_frontiers.size()) {
    std::vector<std::size_t> fallback_indices;
    fallback_indices.reserve(bounded_frontiers.size() - ordered_frontiers.size());
    for (std::size_t index = 0; index < bounded_frontiers.size(); ++index) {
      if (!ordered_indices[index]) {
        fallback_indices.push_back(index);
      }
    }

    const auto start_score = [&](std::size_t index) {
        double score = compute_mrtsp_start_cost(
          candidates[index],
          robot_state,
          weights,
          params.sensor_effective_range_m,
          params.max_linear_speed_vmax,
          params.max_angular_speed_wmax);
        return std::isnan(score) ? std::numeric_limits<double>::infinity() : score;
      };

    std::sort(
      fallback_indices.begin(),
      fallback_indices.end(),
      [&](std::size_t lhs, std::size_t rhs) {
        constexpr double kScoreTieEpsilon = 1e-12;
        const double lhs_score = start_score(lhs);
        const double rhs_score = start_score(rhs);
        const bool lhs_finite = std::isfinite(lhs_score);
        const bool rhs_finite = std::isfinite(rhs_score);
        if (lhs_finite != rhs_finite) {
          return lhs_finite;
        }
        if (lhs_finite && std::abs(lhs_score - rhs_score) > kScoreTieEpsilon) {
          return lhs_score < rhs_score;
        }
        if (!lhs_finite && lhs_score != rhs_score) {
          return lhs_score < rhs_score;
        }
        if (candidates[lhs].size != candidates[rhs].size) {
          return candidates[lhs].size > candidates[rhs].size;
        }
        return lhs < rhs;
      });

    for (const std::size_t index : fallback_indices) {
      ordered_frontiers.push_back(bounded_frontiers[index]);
    }
  }

  auto & mutable_self = const_cast<FrontierExplorerCore &>(*this);
  // Cache stores the already mapped FrontierSequence rather than solver indices, which
  // keeps later dispatch code independent from candidate/pruned-vector bookkeeping.
  mutable_self.mrtsp_order_cache = MrtspOrderCacheEntry{
    signature,
    pose_x_bucket,
    pose_y_bucket,
    yaw_bucket,
    params.sensor_effective_range_m,
    params.weight_distance_wd,
    params.weight_gain_ws,
    params.max_linear_speed_vmax,
    params.max_angular_speed_wmax,
    params.mrtsp_solver,
    params.dp_solver_candidate_limit,
    params.dp_planning_horizon,
    ordered_frontiers,
  };
  mutable_self.mrtsp_order_cache_misses += 1;
  if (debug_outputs_enabled()) {
    callbacks.log_debug(
      "mrtsp_order_cache: miss, frontiers=" + std::to_string(bounded_frontiers.size()) +
      ", ordered=" + std::to_string(ordered_frontiers.size()) +
      ", solver=" + params.mrtsp_solver);
  }
  return apply_startup_heading_preference(ordered_frontiers);
}

std::pair<double, double> FrontierExplorerCore::frontier_position(const FrontierLike & frontier) const
{
  return frontier_exploration_ros2::frontier_position(frontier);
}

std::pair<double, double> FrontierExplorerCore::frontier_reference_point(const FrontierLike & frontier) const
{
  return frontier_exploration_ros2::frontier_reference_point(frontier);
}

int FrontierExplorerCore::frontier_size(const FrontierLike & frontier) const
{
  return frontier_exploration_ros2::frontier_size(frontier);
}

std::string FrontierExplorerCore::describe_frontier(const FrontierLike & frontier) const
{
  return frontier_exploration_ros2::describe_frontier(frontier);
}

FrontierSignature FrontierExplorerCore::frontier_signature(const FrontierSequence & frontiers) const
{
  return frontier_exploration_ros2::frontier_signature(frontiers, params.frontier_visit_tolerance);
}

bool FrontierExplorerCore::frontier_snapshot_matches(
  const std::optional<FrontierSnapshot> & snapshot,
  const std::pair<int, int> & robot_map_cell,
  double min_goal_distance) const
{
  // Snapshot reuse follows the actual frontier-search inputs. Raw map generation may advance
  // without changing decision_map output, so decision_map_generation is the map-side key.
  return (
    snapshot.has_value() &&
    snapshot->decision_map_generation == decision_map_generation &&
    snapshot->costmap_generation == costmap_generation &&
    snapshot->local_costmap_generation == local_costmap_generation &&
    snapshot->robot_map_cell == robot_map_cell &&
    snapshot->min_goal_distance == min_goal_distance);
}

void FrontierExplorerCore::throttled_debug(const std::string & message)
{
  const int64_t now_ns = callbacks.now_ns();
  // Shared throttle avoids flooding debug logs in rapid map/costmap callback bursts.
  const int64_t throttle_ns = static_cast<int64_t>(frontier_stats_log_throttle_seconds * 1e9);
  if (!last_frontier_stats_log_time_ns.has_value() ||
    now_ns - *last_frontier_stats_log_time_ns >= throttle_ns)
  {
    // Move throttle window only when we actually emit a message.
    last_frontier_stats_log_time_ns = now_ns;
    callbacks.log_debug(message);
  }
}

void FrontierExplorerCore::log_frontier_snapshot_stats(
  const FrontierSequence & frontiers,
  double duration_ms,
  bool cache_hit)
{
  std::ostringstream oss;
  // Compact log line keeps p50/p95-style frontier timing inspection easy in runtime logs.
  oss << "frontier_snapshot: "
      << (cache_hit ? "hit" : "miss")
      << ", frontiers=" << frontiers.size()
      << ", duration_ms=" << std::fixed << std::setprecision(2) << duration_ms
      << ", hits=" << frontier_snapshot_cache_hits
      << ", misses=" << frontier_snapshot_cache_misses;
  throttled_debug(oss.str());
}

FrontierSnapshot FrontierExplorerCore::get_frontier_snapshot(
  const geometry_msgs::msg::Pose & current_pose,
  double min_goal_distance)
{
  if (!decision_map.has_value() || decision_map_dirty) {
    if (!map.has_value()) {
      throw std::logic_error("Decision map is not initialized");
    }
    refresh_decision_map();
  }
  // Snapshot cache is keyed by decision/costmap generations + robot cell + min_goal_distance.
  const auto robot_map_cell = decision_map->worldToMap(current_pose.position.x, current_pose.position.y);
  if (frontier_snapshot_matches(frontier_snapshot, robot_map_cell, min_goal_distance)) {
    // Cache hit: avoid repeating expensive frontier extraction.
    frontier_snapshot_cache_hits += 1;
    log_frontier_snapshot_stats(frontier_snapshot->frontiers, 0.0, true);
    return *frontier_snapshot;
  }

  const auto started_at = std::chrono::steady_clock::now();
  const auto search_result = callbacks.frontier_search(
    current_pose,
    *decision_map,
    *costmap,
    local_costmap,
    min_goal_distance,
    true);
  const auto finished_at = std::chrono::steady_clock::now();
  const double duration_ms = std::chrono::duration<double, std::milli>(finished_at - started_at).count();

  FrontierSnapshot snapshot;
  // Convert low-level search output into policy-facing representation + cache key metadata.
  const FrontierSequence raw_frontiers = to_frontier_sequence(search_result.frontiers);
  snapshot.frontiers = filter_frontiers_for_boundary(raw_frontiers);
  snapshot.signature = frontier_signature(snapshot.frontiers);
  snapshot.map_generation = map_generation;
  snapshot.decision_map_generation = decision_map_generation;
  snapshot.costmap_generation = costmap_generation;
  snapshot.local_costmap_generation = local_costmap_generation;
  snapshot.robot_map_cell = search_result.robot_map_cell;
  snapshot.min_goal_distance = min_goal_distance;

  frontier_snapshot = snapshot;
  frontier_snapshot_cache_misses += 1;
  if (debug_outputs_enabled() && map.has_value()) {
    std::size_t raw_frontier_count = raw_frontiers.size();
    if (frontier_map_optimization_enabled()) {
      const FrontierSearchOptions options = frontier_search_options();
      if (
        raw_frontier_debug_cache.has_value() &&
        raw_frontier_debug_cache->map_generation == map_generation &&
        raw_frontier_debug_cache->costmap_generation == costmap_generation &&
        raw_frontier_debug_cache->local_costmap_generation == local_costmap_generation &&
        raw_frontier_debug_cache->robot_map_cell == search_result.robot_map_cell &&
        raw_frontier_debug_cache->min_goal_distance == min_goal_distance &&
        raw_frontier_debug_cache->search_options.occ_threshold == options.occ_threshold &&
        raw_frontier_debug_cache->search_options.min_frontier_size_cells == options.min_frontier_size_cells &&
        raw_frontier_debug_cache->search_options.candidate_min_goal_distance_m == options.candidate_min_goal_distance_m)
      {
        raw_frontier_count = raw_frontier_debug_cache->frontier_count;
      } else {
        const auto raw_search_result = get_frontier(
          current_pose,
          *map,
          *costmap,
          local_costmap,
          min_goal_distance,
          false,
          options);
        raw_frontier_count = raw_search_result.frontiers.size();
        raw_frontier_debug_cache = RawFrontierDebugCacheEntry{
          map_generation,
          costmap_generation,
          local_costmap_generation,
          raw_search_result.robot_map_cell,
          min_goal_distance,
          options,
          raw_frontier_count,
        };
      }
    }
    callbacks.log_debug(
      "frontier_counts: raw=" + std::to_string(raw_frontier_count) +
      ", decision=" + std::to_string(snapshot.frontiers.size()));
  }
  if (snapshot.frontiers.size() != raw_frontiers.size()) {
    callbacks.log_info(
      "Filtered " + std::to_string(raw_frontiers.size() - snapshot.frontiers.size()) +
      " frontier(s) outside the exploration boundary");
  }
  log_frontier_snapshot_stats(snapshot.frontiers, duration_ms, false);
  return snapshot;
}

void FrontierExplorerCore::start_post_goal_settle()
{
  awaiting_map_refresh = true;
  post_goal_settle_active = true;
  post_goal_settle_started_at_ns = callbacks.now_ns();
}

void FrontierExplorerCore::wait_for_next_map_refresh()
{
   awaiting_map_refresh = true;

  if (params.post_goal_settle_enabled) {
    post_goal_settle_active = true;
    post_goal_settle_started_at_ns = callbacks.now_ns();
  } else {
    post_goal_settle_active = false;
    post_goal_settle_started_at_ns.reset();
  }
  start_post_goal_settle();
}

void FrontierExplorerCore::clear_post_goal_wait_state()
{
  awaiting_map_refresh = false;
  post_goal_settle_active = false;
  post_goal_settle_started_at_ns.reset();
}

bool FrontierExplorerCore::post_goal_settle_ready() const
{
  if (!awaiting_map_refresh) {
    return true;
  }

  if (!post_goal_settle_active) {
    return !params.post_goal_settle_enabled;
  }

  if (!post_goal_settle_started_at_ns.has_value()) {
    return false;
  }

  const double elapsed = static_cast<double>(callbacks.now_ns() - *post_goal_settle_started_at_ns) / 1e9;
  if (elapsed < params.post_goal_min_settle) {
    return false;
  }
  return true;
}

FrontierSelectionResult FrontierExplorerCore::select_frontier(
  const FrontierSequence & frontiers,
  const geometry_msgs::msg::Pose & current_pose) const
{
  const FrontierSequence ordered_frontiers = build_mrtsp_frontier_sequence(frontiers, current_pose);
  if (ordered_frontiers.empty()) {
    return {std::nullopt, ""};
  }
  return {ordered_frontiers.front(), "mrtsp"};
}

void FrontierExplorerCore::record_start_pose(const geometry_msgs::msg::Pose & current_pose)
{
  if (start_pose.has_value()) {
    // Start pose is recorded once per node lifetime and survives session stop/start cycles.
    return;
  }

  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = params.global_frame;
  pose.pose = current_pose;
  start_pose = pose;

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2)
      << "Recorded exploration start pose: ("
      << current_pose.position.x << ", "
      << current_pose.position.y << ")";
  callbacks.log_info(oss.str());
}

bool FrontierExplorerCore::exploration_boundary_enabled() const
{
  return params.exploration_boundary_radius_m > 0.0 && start_pose.has_value();
}

bool FrontierExplorerCore::point_within_exploration_boundary(
  double wx,
  double wy,
  double margin_m) const
{
  if (!exploration_boundary_enabled()) {
    return true;
  }
  const double radius = std::max(0.0, params.exploration_boundary_radius_m - margin_m);
  const double dx = wx - start_pose->pose.position.x;
  const double dy = wy - start_pose->pose.position.y;
  return (dx * dx + dy * dy) <= (radius * radius + 1e-9);
}

FrontierSequence FrontierExplorerCore::filter_frontiers_for_boundary(
  const FrontierSequence & frontiers) const
{
  if (!exploration_boundary_enabled()) {
    return frontiers;
  }

  FrontierSequence filtered;
  filtered.reserve(frontiers.size());
  for (const auto & frontier : frontiers) {
    const auto point = frontier_position(frontier);
    if (point_within_exploration_boundary(point.first, point.second)) {
      filtered.push_back(frontier);
    }
  }
  return filtered;
}

bool FrontierExplorerCore::are_frontiers_equivalent(
  const std::optional<FrontierLike> & first_frontier,
  const std::optional<FrontierLike> & second_frontier) const
{
  return frontier_exploration_ros2::are_frontiers_equivalent(
    first_frontier,
    second_frontier,
    params.frontier_visit_tolerance);
}

bool FrontierExplorerCore::frontier_exists_in_set(
  const std::optional<FrontierLike> & frontier,
  const FrontierSequence & frontiers) const
{
  if (!frontier.has_value()) {
    return false;
  }

  for (const auto & candidate : frontiers) {
    // Tolerance-based check reuses shared frontier equivalence policy.
    if (are_frontiers_equivalent(frontier, candidate)) {
      return true;
    }
  }

  return false;
}

std::optional<std::string> FrontierExplorerCore::frontier_cost_status(
  const std::optional<FrontierLike> & frontier) const
{
  if (!frontier.has_value()) {
    return std::nullopt;
  }

  const auto goal_point = frontier_position(*frontier);

  const auto local_cost = world_point_cost(local_costmap, goal_point);
  if (local_cost.has_value() && *local_cost >= params.occ_threshold) {
    // Local map blocks have priority because they are most immediate for controller safety.
    return std::string(
      "Current frontier target is blocked in local costmap (cost=") +
      std::to_string(*local_cost) + ")";
  }

  const auto global_cost = world_point_cost(costmap, goal_point);
  if (global_cost.has_value() && *global_cost >= params.occ_threshold) {
    return std::string(
      "Current frontier target is blocked in global costmap (cost=") +
      std::to_string(*global_cost) + ")";
  }

  return std::nullopt;
}

geometry_msgs::msg::PoseStamped FrontierExplorerCore::build_goal_pose(
  const FrontierLike & target_frontier,
  const geometry_msgs::msg::Pose & current_pose) const
{
  const auto [target_x, target_y] = frontier_position(target_frontier);
  geometry_msgs::msg::PoseStamped goal_pose;
  goal_pose.header.frame_id = params.global_frame;
  goal_pose.pose.position.x = target_x;
  goal_pose.pose.position.y = target_y;
  // Default heading follows the travel vector toward the selected target frontier.
  // This avoids forcing a stale "current heading" orientation at goal completion.
  goal_pose.pose.orientation = current_pose.orientation;
  const double to_target_dx = target_x - current_pose.position.x;
  const double to_target_dy = target_y - current_pose.position.y;
  if (std::hypot(to_target_dx, to_target_dy) > 0.05) {
    goal_pose.pose.orientation = detail::quaternion_from_yaw(std::atan2(to_target_dy, to_target_dx));
  }
  return goal_pose;
}

std::optional<std::pair<double, double>> FrontierExplorerCore::resolve_dispatch_goal_point(
  const FrontierLike & target_frontier,
  const geometry_msgs::msg::Pose & current_pose,
  bool bypass_min_distance_dispatch) const
{
  const auto target_point = frontier_position(target_frontier);
  const auto frontier_reference = frontier_reference_point(target_frontier);
  const double robot_distance_to_frontier = std::hypot(
    frontier_reference.first - current_pose.position.x,
    frontier_reference.second - current_pose.position.y);
  const auto makes_frontier_progress =
    [
    this, bypass_min_distance_dispatch, &frontier_reference,
    robot_distance_to_frontier
    ](const std::pair<double, double> & world_point) {
      if (
        bypass_min_distance_dispatch ||
        params.dispatch_min_frontier_progress_m < 0.0)
      {
        return true;
      }
      const double endpoint_distance_to_frontier = std::hypot(
        frontier_reference.first - world_point.first,
        frontier_reference.second - world_point.second);
      const double progress_m =
        robot_distance_to_frontier - endpoint_distance_to_frontier;
      return progress_m + 1e-9 >= params.dispatch_min_frontier_progress_m;
    };
  if (!point_within_exploration_boundary(target_point.first, target_point.second)) {
    return std::nullopt;
  }
  if (!map.has_value()) {
    return makes_frontier_progress(target_point) ?
      std::optional<std::pair<double, double>>(target_point) : std::nullopt;
  }

  int target_map_x = 0;
  int target_map_y = 0;
  if (!map->worldToMapNoThrow(target_point.first, target_point.second, target_map_x, target_map_y)) {
    return std::nullopt;
  }

  if (
    suppression_runtime_active(callbacks.now_ns()) && frontier_suppression_ &&
    frontier_suppression_->is_point_suppressed(target_point))
  {
    return std::nullopt;
  }

  // A frontier reference closer than frontier_selection_min_distance is NOT
  // rejected here: the BFS below can still resolve a dispatch cell that
  // satisfies the min distance (cell_dispatchable / target_cell_dispatchable
  // enforce it per cell). Rejecting on the reference distance alone starves
  // dispatch entirely when the only frontier hugs the robot, e.g. the ring
  // around the seeded footprint at the start of a depth-only bootstrap.

  int robot_map_x = 0;
  int robot_map_y = 0;
  if (!map->worldToMapNoThrow(
      current_pose.position.x,
      current_pose.position.y,
      robot_map_x,
      robot_map_y))
  {
    return std::nullopt;
  }

  const int width = map->getSizeX();
  const int height = map->getSizeY();
  if (width <= 0 || height <= 0) {
    return std::nullopt;
  }

  const double min_robot_distance = bypass_min_distance_dispatch ?
    0.0 :
    std::max(0.0, params.frontier_selection_min_distance);
  const double min_robot_distance_sq = min_robot_distance * min_robot_distance;
  const double clearance_radius = std::max(0.0, params.dispatch_clearance_radius_m);
  const bool enforce_costmap_reachability =
    params.goal_skip_on_blocked_goal || clearance_radius > 0.0;

  const auto in_bounds = [width, height](int map_x, int map_y) {
      return map_x >= 0 && map_y >= 0 && map_x < width && map_y < height;
    };

  const auto grid_cell_blocked = [this](const OccupancyGrid2d & grid, int map_x, int map_y) {
      return grid.getCost(map_x, map_y) >= params.occ_threshold;
    };

  const auto world_blocked_in_grid =
    [this, clearance_radius, &grid_cell_blocked](
    const OccupancyGrid2d & grid,
    const std::pair<double, double> & world_point) {
      int grid_x = 0;
      int grid_y = 0;
      if (!grid.worldToMapNoThrow(world_point.first, world_point.second, grid_x, grid_y)) {
        return false;
      }
      if (clearance_radius <= 0.0) {
        return grid_cell_blocked(grid, grid_x, grid_y);
      }

      const double resolution = grid.map().info.resolution;
      const int radius_cells = static_cast<int>(std::ceil(clearance_radius / resolution));
      for (int y = grid_y - radius_cells; y <= grid_y + radius_cells; ++y) {
        for (int x = grid_x - radius_cells; x <= grid_x + radius_cells; ++x) {
          if (x < 0 || y < 0 || x >= grid.getSizeX() || y >= grid.getSizeY()) {
            continue;
          }
          const auto neighbor_world = grid.mapToWorld(x, y);
          if (
            std::hypot(
              neighbor_world.first - world_point.first,
              neighbor_world.second - world_point.second) <= clearance_radius &&
            grid_cell_blocked(grid, x, y))
          {
            return true;
          }
        }
      }
      return false;
    };

  const auto costmap_cost_at =
    [this](const std::pair<double, double> & world_point) -> std::optional<int> {
      if (!costmap.has_value()) {
        return std::nullopt;
      }
      int grid_x = 0;
      int grid_y = 0;
      if (!costmap->worldToMapNoThrow(
          world_point.first,
          world_point.second,
          grid_x,
          grid_y))
      {
        return std::nullopt;
      }
      return costmap->getCost(grid_x, grid_y);
    };

  // Known-free gating keeps dispatch cells inside the region Nav2 can plan
  // through with allow_unknown: false; without it, dispatch points chosen
  // in unknown space fail with NO_VALID_PATH. Unknown-costmap cells and
  // cells outside the costmap are non-traversable under the gate. Until
  // the first costmap arrives the gate stays inactive rather than
  // starving dispatch at startup.
  const bool require_known_free_costmap =
    params.dispatch_requires_known_free_costmap && costmap.has_value();

  const auto point_blocked_for_connectivity =
    [&](const std::pair<double, double> & world_point) {
      // The global and local costmaps are already inflated for the robot
      // footprint. Applying dispatch_clearance_radius_m around every
      // costmap cell a second time can make the robot's own traversable cell
      // appear disconnected whenever it is inside the inflation gradient.
      // Connectivity therefore follows exact occupied/lethal cells. The
      // explicit raw-map clearance margin remains an endpoint requirement
      // below; Nav2 and the motion guard retain path-level safety ownership.
      const auto map_cost = world_point_cost(map, world_point);
      if (map_cost.has_value() && *map_cost >= params.occ_threshold) {
        return true;
      }
      if (enforce_costmap_reachability) {
        const auto global_cost = world_point_cost(costmap, world_point);
        if (global_cost.has_value() && *global_cost >= params.occ_threshold) {
          return true;
        }
        const auto local_cost = world_point_cost(local_costmap, world_point);
        if (local_cost.has_value() && *local_cost >= params.occ_threshold) {
          return true;
        }
      }
      return false;
    };

  const auto endpoint_clearance_blocked =
    [&](const std::pair<double, double> & world_point) {
      // dispatch_clearance_radius_m is a physical endpoint margin against
      // raw mapped obstacles. Costmap inflation already encodes its own
      // footprint margin and must not be dilated again here.
      return clearance_radius > 0.0 && world_blocked_in_grid(*map, world_point);
    };

  const auto point_suppressed = [&](const std::pair<double, double> & world_point) {
      return suppression_runtime_active(callbacks.now_ns()) && frontier_suppression_ &&
             frontier_suppression_->is_point_suppressed(world_point);
    };

  const auto cell_world = [&](int map_x, int map_y) {
      return map->mapToWorld(map_x, map_y);
    };

  const auto cell_connected = [&](int map_x, int map_y) {
      if (!in_bounds(map_x, map_y)) {
        return false;
      }
      const auto world_point = cell_world(map_x, map_y);
      if (require_known_free_costmap) {
        const auto cost = costmap_cost_at(world_point);
        if (!cost.has_value() || *cost < 0 || *cost >= params.occ_threshold) {
          return false;
        }
      }
      return point_within_exploration_boundary(world_point.first, world_point.second) &&
             !point_blocked_for_connectivity(world_point);
    };

  const auto cell_endpoint_safe = [&](int map_x, int map_y) {
      if (!cell_connected(map_x, map_y)) {
        return false;
      }
      const auto world_point = cell_world(map_x, map_y);
      return !point_suppressed(world_point) &&
             !endpoint_clearance_blocked(world_point);
    };

  const auto cell_dispatchable = [&](int map_x, int map_y) {
      if (!cell_endpoint_safe(map_x, map_y)) {
        return false;
      }
      if (min_robot_distance_sq <= 0.0) {
        return true;
      }
      const auto world_point = cell_world(map_x, map_y);
      const double dx = world_point.first - current_pose.position.x;
      const double dy = world_point.second - current_pose.position.y;
      return (dx * dx + dy * dy) >= min_robot_distance_sq;
    };

  const auto target_cell_dispatchable = [&]() {
      if (!cell_dispatchable(target_map_x, target_map_y)) {
        return false;
      }
      if (!makes_frontier_progress(target_point)) {
        return false;
      }
      if (min_robot_distance_sq <= 0.0) {
        return true;
      }
      const double dx = target_point.first - current_pose.position.x;
      const double dy = target_point.second - current_pose.position.y;
      return (dx * dx + dy * dy) >= min_robot_distance_sq;
    };

  const std::size_t cell_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  std::vector<uint8_t> visited(cell_count, 0U);
  std::deque<std::pair<int, int>> queue;
  const auto index = [width](int map_x, int map_y) {
      return static_cast<std::size_t>(map_y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(map_x);
    };

  visited[index(robot_map_x, robot_map_y)] = 1U;
  queue.emplace_back(robot_map_x, robot_map_y);

  std::optional<std::pair<int, int>> best_cell;
  double best_score = std::numeric_limits<double>::infinity();
  double best_robot_distance_sq = -1.0;
  std::optional<std::pair<int, int>> best_fallback_cell;
  double best_fallback_score = std::numeric_limits<double>::infinity();
  double best_fallback_robot_distance_sq = -1.0;
  const double useful_fallback_distance = std::max(0.0, params.frontier_visit_tolerance);
  const double useful_fallback_distance_sq =
    useful_fallback_distance * useful_fallback_distance;

  // Score is target distance in meters plus an optional costmap-cost
  // penalty, so zero-cost interior cells win over equally-near cells
  // inside the inflation gradient. With the penalty disabled the ordering
  // matches the previous squared-distance comparison exactly.
  const auto consider_dispatch_cell = [&](int map_x, int map_y) {
      if (!cell_endpoint_safe(map_x, map_y)) {
        return;
      }
      const auto world_point = cell_world(map_x, map_y);
      if (!makes_frontier_progress(world_point)) {
        return;
      }
      const double target_distance_sq = squared_distance(world_point, target_point);
      double cost_penalty_m = 0.0;
      if (params.dispatch_costmap_cost_penalty_m > 0.0) {
        const auto cost = costmap_cost_at(world_point);
        if (cost.has_value() && *cost > 0) {
          cost_penalty_m = params.dispatch_costmap_cost_penalty_m *
            (static_cast<double>(*cost) / 100.0);
        }
      }
      const double score = std::sqrt(target_distance_sq) + cost_penalty_m;
      const double robot_dx = world_point.first - current_pose.position.x;
      const double robot_dy = world_point.second - current_pose.position.y;
      const double robot_distance_sq = robot_dx * robot_dx + robot_dy * robot_dy;
      constexpr double kTieEpsilon = 1e-9;

      // frontier_selection_min_distance is a preference, not a reason to
      // starve bootstrap when the robot-connected known-free island is
      // initially smaller than that distance. Keep the best safe fallback
      // outside the visit tolerance so the fallback still commands useful
      // progress instead of redispatching the robot's current cell.
      if (
        robot_distance_sq + kTieEpsilon >= useful_fallback_distance_sq &&
        (score + kTieEpsilon < best_fallback_score ||
        (std::abs(score - best_fallback_score) <= kTieEpsilon &&
        robot_distance_sq > best_fallback_robot_distance_sq)))
      {
        best_fallback_score = score;
        best_fallback_robot_distance_sq = robot_distance_sq;
        best_fallback_cell = {map_x, map_y};
      }

      if (robot_distance_sq + kTieEpsilon < min_robot_distance_sq) {
        return;
      }
      if (
        score + kTieEpsilon < best_score ||
        (std::abs(score - best_score) <= kTieEpsilon &&
        robot_distance_sq > best_robot_distance_sq))
      {
        best_score = score;
        best_robot_distance_sq = robot_distance_sq;
        best_cell = {map_x, map_y};
      }
    };

  static constexpr std::array<std::pair<int, int>, 8> kNeighborOffsets{{
    {-1, -1}, {0, -1}, {1, -1},
    {-1, 0},           {1, 0},
    {-1, 1},  {0, 1},  {1, 1},
  }};

  while (!queue.empty()) {
    const auto [map_x, map_y] = queue.front();
    queue.pop_front();

    if (map_x == target_map_x && map_y == target_map_y && target_cell_dispatchable()) {
      return target_point;
    }
    consider_dispatch_cell(map_x, map_y);

    for (const auto & [dx, dy] : kNeighborOffsets) {
      const int next_x = map_x + dx;
      const int next_y = map_y + dy;
      if (!in_bounds(next_x, next_y)) {
        continue;
      }
      const auto next_index = index(next_x, next_y);
      if (visited[next_index]) {
        continue;
      }
      if (!cell_connected(next_x, next_y)) {
        continue;
      }
      visited[next_index] = 1U;
      queue.emplace_back(next_x, next_y);
    }
  }

  if (!best_cell.has_value()) {
    best_cell = best_fallback_cell;
  }
  if (!best_cell.has_value()) {
    return std::nullopt;
  }

  return cell_world(best_cell->first, best_cell->second);
}

geometry_msgs::msg::PoseStamped FrontierExplorerCore::build_dispatch_goal_pose(
  const FrontierLike & target_frontier,
  const geometry_msgs::msg::Pose & current_pose,
  bool bypass_min_distance_dispatch) const
{
  const auto target_point = resolve_dispatch_goal_point(
    target_frontier,
    current_pose,
    bypass_min_distance_dispatch).value_or(frontier_position(target_frontier));

  geometry_msgs::msg::PoseStamped goal_pose;
  goal_pose.header.frame_id = params.global_frame;
  goal_pose.pose.position.x = target_point.first;
  goal_pose.pose.position.y = target_point.second;
  goal_pose.pose.orientation = current_pose.orientation;
  if (params.goal_yaw_policy == "face_frontier") {
    // Arrive looking at the information target: dispatch resolution can
    // pull the goal point back into known-free space, so aim from the
    // goal toward the frontier reference rather than along the path.
    const auto reference = frontier_reference_point(target_frontier);
    const double face_dx = reference.first - target_point.first;
    const double face_dy = reference.second - target_point.second;
    if (std::hypot(face_dx, face_dy) > 0.05) {
      goal_pose.pose.orientation =
        detail::quaternion_from_yaw(std::atan2(face_dy, face_dx));
      return goal_pose;
    }
  }
  const double to_target_dx = target_point.first - current_pose.position.x;
  const double to_target_dy = target_point.second - current_pose.position.y;
  if (std::hypot(to_target_dx, to_target_dy) > 0.05) {
    goal_pose.pose.orientation = detail::quaternion_from_yaw(std::atan2(to_target_dy, to_target_dx));
  }
  return goal_pose;
}

std::vector<geometry_msgs::msg::PoseStamped> FrontierExplorerCore::build_goal_pose_sequence(
  const FrontierSequence & target_frontiers,
  const geometry_msgs::msg::Pose & current_pose) const
{
  std::vector<geometry_msgs::msg::PoseStamped> goal_sequence;
  // Reserve once to keep goal sequence creation allocation-free for steady-state single-frontier mode.
  goal_sequence.reserve(target_frontiers.size());
  for (std::size_t i = 0; i < target_frontiers.size(); ++i) {
    goal_sequence.push_back(build_goal_pose(target_frontiers[i], current_pose));
  }
  return goal_sequence;
}

FrontierSequence FrontierExplorerCore::select_frontier_sequence(
  const FrontierSequence & frontiers,
  const geometry_msgs::msg::Pose & current_pose,
  const std::optional<FrontierLike> & initial_frontier) const
{
  (void)initial_frontier;
  return build_mrtsp_frontier_sequence(frontiers, current_pose);
}

bool FrontierExplorerCore::are_frontier_sequences_equivalent(
  const FrontierSequence & first_frontier_sequence,
  const FrontierSequence & second_frontier_sequence) const
{
  return frontier_exploration_ros2::are_frontier_sequences_equivalent(
    first_frontier_sequence,
    second_frontier_sequence,
    params.frontier_visit_tolerance);
}

}  // namespace frontier_exploration_ros2
