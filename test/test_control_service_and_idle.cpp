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

#include <gtest/gtest.h>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "frontier_exploration_ros2/frontier_explorer_node.hpp"
#include "frontier_exploration_ros2/srv/control_exploration.hpp"
#include "frontier_exploration_ctl_detail.hpp"

namespace frontier_exploration_ros2
{
namespace
{

using ControlExploration = frontier_exploration_ros2::srv::ControlExploration;

geometry_msgs::msg::Pose make_pose(double x = 0.0, double y = 0.0)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.orientation.w = 1.0;
  return pose;
}

FrontierCandidate make_frontier(double x, double y, int size = 1)
{
  return FrontierCandidate{{x, y}, {x, y}, size};
}

nav_msgs::msg::OccupancyGrid build_grid(int width, int height, int default_value)
{
  nav_msgs::msg::OccupancyGrid msg;
  msg.info.width = static_cast<uint32_t>(width);
  msg.info.height = static_cast<uint32_t>(height);
  msg.info.resolution = 1.0;
  msg.info.origin.orientation.w = 1.0;
  msg.data.assign(static_cast<std::size_t>(width * height), static_cast<int8_t>(default_value));
  return msg;
}

TEST(ControlCoreSessionTests, StopExplorationSessionDisablesScheduling)
{
  int frontier_search_calls = 0;
  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};
  callbacks.get_current_pose = []() {
      return std::optional<geometry_msgs::msg::Pose>(make_pose(1.0, 1.0));
    };
  callbacks.frontier_search =
    [&frontier_search_calls](
    const geometry_msgs::msg::Pose &,
    const OccupancyGrid2d &,
    const OccupancyGrid2d &,
    const std::optional<OccupancyGrid2d> &,
    double,
    bool)
    {
      frontier_search_calls += 1;
      return FrontierSearchResult{};
    };

  FrontierExplorerCore core(FrontierExplorerCoreParams{}, callbacks);
  core.map = OccupancyGrid2d(build_grid(10, 10, 0));
  core.costmap = OccupancyGrid2d(build_grid(10, 10, 0));
  core.stop_exploration_session("test stop");
  core.try_send_next_goal();

  EXPECT_EQ(frontier_search_calls, 0);
}

TEST(ControlCoreSessionTests, RepeatedUndispatchableFrontierSetIsRateLimited)
{
  int64_t now_ns = 1'000'000'000;
  std::vector<std::string> info_logs;
  FrontierExplorerCoreParams params;
  params.frontier_map_optimization_enabled = false;
  params.undispatchable_frontier_retry_interval_s = 2.0;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = [&now_ns]() {return now_ns;};
  callbacks.get_current_pose = []() {
      return std::optional<geometry_msgs::msg::Pose>(make_pose(1.5, 1.5));
    };
  callbacks.frontier_search = [](
    const geometry_msgs::msg::Pose &,
    const OccupancyGrid2d &,
    const OccupancyGrid2d &,
    const std::optional<OccupancyGrid2d> &,
    double,
    bool)
    {
      FrontierSearchResult result;
      result.frontiers = {make_frontier(8.5, 8.5, 20)};
      result.robot_map_cell = {1, 1};
      return result;
    };
  callbacks.log_info = [&info_logs](const std::string & message) {
      info_logs.push_back(message);
    };

  FrontierExplorerCore core(params, callbacks);
  // An occupied map makes the selected frontier genuinely undispatchable.
  // Map and costmap callbacks may still invoke this scheduler independently.
  core.map = OccupancyGrid2d(build_grid(10, 10, 100));
  core.costmap = OccupancyGrid2d(build_grid(10, 10, 100));

  const auto rejection_count = [&info_logs]() {
      return std::count_if(
        info_logs.begin(),
        info_logs.end(),
        [](const std::string & message) {
          return message.find("All selected frontier goals are blocked") !=
                 std::string::npos;
        });
    };

  core.try_send_next_goal();
  EXPECT_EQ(rejection_count(), 1);

  now_ns = 2'000'000'000;
  core.try_send_next_goal();
  EXPECT_EQ(rejection_count(), 1);

  now_ns = 3'100'000'000;
  core.try_send_next_goal();
  EXPECT_EQ(rejection_count(), 2);
}

TEST(ControlCoreSessionTests, StartExplorationSessionResetsSessionState)
{
  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{5'000'000'000};};
  FrontierExplorerCore core(FrontierExplorerCoreParams{}, callbacks);

  geometry_msgs::msg::PoseStamped persistent_start_pose;
  persistent_start_pose.pose = make_pose(2.0, 3.0);
  core.start_pose = persistent_start_pose;
  core.pending_frontier_sequence = {make_frontier(1.0, 1.0)};
  core.pending_frontier_selection_mode = "mrtsp";
  core.return_to_start_completed = true;
  core.no_frontiers_reported = true;
  core.frontier_suppression_ = std::make_unique<FrontierSuppression>(FrontierSuppressionConfig{});
  core.map = OccupancyGrid2d(build_grid(10, 10, 0));
  core.costmap = OccupancyGrid2d(build_grid(10, 10, 0));
  core.map_generation = 1;
  core.costmap_generation = 2;

  core.start_exploration_session();

  EXPECT_TRUE(core.exploration_enabled);
  ASSERT_TRUE(core.start_pose.has_value());
  EXPECT_DOUBLE_EQ(core.start_pose->pose.position.x, persistent_start_pose.pose.position.x);
  EXPECT_DOUBLE_EQ(core.start_pose->pose.position.y, persistent_start_pose.pose.position.y);
  EXPECT_TRUE(core.pending_frontier_sequence.empty());
  EXPECT_TRUE(core.pending_frontier_selection_mode.empty());
  EXPECT_FALSE(core.return_to_start_completed);
  EXPECT_FALSE(core.no_frontiers_reported);
  EXPECT_FALSE(core.map.has_value());
  EXPECT_FALSE(core.costmap.has_value());
  EXPECT_EQ(core.map_generation, 0);
  EXPECT_EQ(core.costmap_generation, 0);
  EXPECT_FALSE(core.frontier_suppression_);
}

TEST(ControlCoreSessionTests, StopAndStartSessionsPreserveOriginalStartPose)
{
  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{5'000'000'000};};
  FrontierExplorerCore core(FrontierExplorerCoreParams{}, callbacks);

  geometry_msgs::msg::PoseStamped persistent_start_pose;
  persistent_start_pose.pose = make_pose(4.0, 6.0);
  core.start_pose = persistent_start_pose;

  core.stop_exploration_session("test stop");
  ASSERT_TRUE(core.start_pose.has_value());
  EXPECT_DOUBLE_EQ(core.start_pose->pose.position.x, persistent_start_pose.pose.position.x);
  EXPECT_DOUBLE_EQ(core.start_pose->pose.position.y, persistent_start_pose.pose.position.y);

  core.start_exploration_session();
  ASSERT_TRUE(core.start_pose.has_value());
  EXPECT_DOUBLE_EQ(core.start_pose->pose.position.x, persistent_start_pose.pose.position.x);
  EXPECT_DOUBLE_EQ(core.start_pose->pose.position.y, persistent_start_pose.pose.position.y);
}

TEST(GlobalFrameConsistencyTests, RejectsMapAndCostmapUpdatesInWrongFrame)
{
  std::vector<std::string> errors;
  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};
  callbacks.log_error = [&errors](const std::string & message) {
      errors.push_back(message);
    };
  FrontierExplorerCore core(FrontierExplorerCoreParams{}, callbacks);

  auto wrong_frame_grid = build_grid(10, 10, 0);
  wrong_frame_grid.header.frame_id = "odom";

  core.occupancyGridCallback(OccupancyGrid2d(wrong_frame_grid));
  EXPECT_FALSE(core.map.has_value());
  EXPECT_EQ(core.map_generation, 0);

  core.costmapCallback(OccupancyGrid2d(wrong_frame_grid));
  EXPECT_FALSE(core.costmap.has_value());
  EXPECT_EQ(core.costmap_generation, 0);

  core.ingestRawMapUpdate(OccupancyGrid2d(wrong_frame_grid));
  EXPECT_FALSE(core.map.has_value());

  EXPECT_EQ(core.frame_mismatch_rejections, 3);
  // now_ns is frozen, so the second and third rejections are throttled.
  ASSERT_EQ(errors.size(), 1U);
  EXPECT_NE(errors[0].find("'odom'"), std::string::npos);
  EXPECT_NE(errors[0].find("'map'"), std::string::npos);
}

TEST(GlobalFrameConsistencyTests, AcceptsMatchingAndUnsetFrames)
{
  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};
  FrontierExplorerCore core(FrontierExplorerCoreParams{}, callbacks);
  core.stop_exploration_session("test stop");

  auto map_frame_grid = build_grid(10, 10, 0);
  map_frame_grid.header.frame_id = "map";
  core.occupancyGridCallback(OccupancyGrid2d(map_frame_grid));
  EXPECT_TRUE(core.map.has_value());
  EXPECT_EQ(core.map_generation, 1);

  auto unset_frame_grid = build_grid(10, 10, 0);
  core.costmapCallback(OccupancyGrid2d(unset_frame_grid));
  EXPECT_TRUE(core.costmap.has_value());
  EXPECT_EQ(core.frame_mismatch_rejections, 0);
}

TEST(GlobalFrameConsistencyTests, LocalCostmapMayUseDifferentFrame)
{
  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};
  FrontierExplorerCore core(FrontierExplorerCoreParams{}, callbacks);
  core.stop_exploration_session("test stop");

  auto odom_grid = build_grid(10, 10, 0);
  odom_grid.header.frame_id = "odom";
  core.localCostmapCallback(OccupancyGrid2d(odom_grid));
  EXPECT_TRUE(core.local_costmap.has_value());
  EXPECT_EQ(core.frame_mismatch_rejections, 0);
}

TEST(GlobalFrameConsistencyTests, MismatchLoggingResumesAfterThrottleWindow)
{
  int64_t now_ns = 0;
  std::vector<std::string> errors;
  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = [&now_ns]() {return now_ns;};
  callbacks.log_error = [&errors](const std::string & message) {
      errors.push_back(message);
    };
  FrontierExplorerCore core(FrontierExplorerCoreParams{}, callbacks);

  auto wrong_frame_grid = build_grid(10, 10, 0);
  wrong_frame_grid.header.frame_id = "odom";

  core.occupancyGridCallback(OccupancyGrid2d(wrong_frame_grid));
  now_ns += int64_t{1'000'000'000};
  core.occupancyGridCallback(OccupancyGrid2d(wrong_frame_grid));
  EXPECT_EQ(errors.size(), 1U);

  now_ns += int64_t{10'000'000'000};
  core.occupancyGridCallback(OccupancyGrid2d(wrong_frame_grid));
  EXPECT_EQ(errors.size(), 2U);
  EXPECT_EQ(core.frame_mismatch_rejections, 3);
}

nav_msgs::msg::OccupancyGrid build_footprint_seed_grid(int fill_value)
{
  // 2 m x 2 m grid at 5 cm resolution centered on the robot with a free
  // disk (r = 0.3 m) around the origin, mirroring the seeded footprint at
  // the start of a depth-only bootstrap.
  auto grid = build_grid(40, 40, fill_value);
  grid.info.resolution = 0.05;
  grid.info.origin.position.x = -1.0;
  grid.info.origin.position.y = -1.0;
  for (int y = 0; y < 40; ++y) {
    for (int x = 0; x < 40; ++x) {
      const double wx = -1.0 + (static_cast<double>(x) + 0.5) * 0.05;
      const double wy = -1.0 + (static_cast<double>(y) + 0.5) * 0.05;
      if (wx * wx + wy * wy <= 0.3 * 0.3) {
        grid.data[static_cast<std::size_t>(y) * 40 + static_cast<std::size_t>(x)] = 0;
      }
    }
  }
  return grid;
}

TEST(DispatchGoalPointTests, CloseFrontierStillResolvesMinDistanceDispatchPoint)
{
  FrontierExplorerCoreParams params;
  params.frontier_selection_min_distance = 0.4;
  params.occ_threshold = 90;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};
  FrontierExplorerCore core(params, callbacks);

  // Unknown everywhere beyond the seeded footprint: unknown cells are
  // traversable for dispatch resolution, so a valid dispatch cell exists.
  core.map = OccupancyGrid2d(build_footprint_seed_grid(-1));

  const auto pose = make_pose(0.0, 0.0);
  const auto frontier = make_frontier(0.21, 0.10, 80);
  const auto dispatch_point = core.resolve_dispatch_goal_point(frontier, pose, false);

  ASSERT_TRUE(dispatch_point.has_value());
  const double robot_distance_sq =
    dispatch_point->first * dispatch_point->first +
    dispatch_point->second * dispatch_point->second;
  EXPECT_GE(robot_distance_sq, 0.4 * 0.4 - 1e-9);
  const double target_dx = dispatch_point->first - 0.21;
  const double target_dy = dispatch_point->second - 0.10;
  EXPECT_LE(target_dx * target_dx + target_dy * target_dy, 0.35 * 0.35);
}

TEST(DispatchGoalPointTests, CloseFrontierWithoutFarTraversableCellsResolvesNothing)
{
  FrontierExplorerCoreParams params;
  params.frontier_selection_min_distance = 0.4;
  params.occ_threshold = 90;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};
  FrontierExplorerCore core(params, callbacks);

  // Occupied everywhere beyond the seeded footprint: every candidate cell is
  // either blocked or already within the visit tolerance, so there is no
  // useful fallback dispatch point.
  core.map = OccupancyGrid2d(build_footprint_seed_grid(100));

  const auto pose = make_pose(0.0, 0.0);
  const auto frontier = make_frontier(0.21, 0.10, 80);
  const auto dispatch_point = core.resolve_dispatch_goal_point(frontier, pose, false);

  EXPECT_FALSE(dispatch_point.has_value());
}

TEST(DispatchGoalPointTests, KnownFreeGateFallsBackInsideShortConnectedComponent)
{
  // Regression for the TurtleBot 4 dock bootstrap corpus: the selected
  // frontier is farther than the preferred dispatch distance, but the
  // robot-connected known-free costmap island is only about 0.5 m long.
  // Dispatch should advance to its safe edge instead of starving until the
  // island somehow grows without robot motion.
  auto map_grid = build_grid(40, 40, 0);
  map_grid.info.resolution = 0.05;
  map_grid.info.origin.position.x = -1.0;
  map_grid.info.origin.position.y = -1.0;

  auto costmap_grid = build_grid(40, 40, -1);
  costmap_grid.info.resolution = 0.05;
  costmap_grid.info.origin.position.x = -1.0;
  costmap_grid.info.origin.position.y = -1.0;
  const int corridor_y = 20;
  for (int x = 20; x <= 30; ++x) {
    costmap_grid.data[
      static_cast<std::size_t>(corridor_y) * 40 + static_cast<std::size_t>(x)] = 0;
  }

  FrontierExplorerCoreParams params;
  params.frontier_selection_min_distance = 0.8;
  params.frontier_visit_tolerance = 0.4;
  params.occ_threshold = 90;
  params.dispatch_requires_known_free_costmap = true;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};
  FrontierExplorerCore core(params, callbacks);
  core.map = OccupancyGrid2d(map_grid);
  core.costmap = OccupancyGrid2d(costmap_grid);

  const auto pose = make_pose(0.025, 0.025);
  const auto dispatch_point = core.resolve_dispatch_goal_point(
    make_frontier(0.825, 0.025, 40), pose, false);

  ASSERT_TRUE(dispatch_point.has_value());
  const double robot_distance = std::hypot(
    dispatch_point->first - pose.position.x,
    dispatch_point->second - pose.position.y);
  EXPECT_GE(robot_distance, params.frontier_visit_tolerance - 1e-9);
  EXPECT_LT(robot_distance, params.frontier_selection_min_distance);
  EXPECT_NEAR(dispatch_point->first, 0.525, 1e-6);
  EXPECT_NEAR(dispatch_point->second, 0.025, 1e-6);
}

TEST(DispatchGoalPointTests, InflatedCostmapDoesNotApplyEndpointClearanceTwice)
{
  // Exact regression shape from the 2026-07-18 depth-fidelity run. The
  // robot occupies a traversable inflation-gradient cell (cost 53) with a
  // lethal costmap cell less than one endpoint-clearance radius away. The
  // old resolver dilated that already-inflated costmap again, reducing the
  // connected component to the robot cell even though a known-free corridor
  // led to safe dispatch endpoints.
  auto map_grid = build_grid(60, 60, 0);
  map_grid.info.resolution = 0.05;
  map_grid.info.origin.position.x = -1.5;
  map_grid.info.origin.position.y = -1.5;

  auto costmap_grid = build_grid(60, 60, -1);
  costmap_grid.info.resolution = 0.05;
  costmap_grid.info.origin.position.x = -1.5;
  costmap_grid.info.origin.position.y = -1.5;
  constexpr int robot_x = 20;
  constexpr int corridor_y = 30;
  for (int x = robot_x; x <= 45; ++x) {
    costmap_grid.data[
      static_cast<std::size_t>(corridor_y) * 60 + static_cast<std::size_t>(x)] = 0;
  }
  costmap_grid.data[
    static_cast<std::size_t>(corridor_y) * 60 + robot_x] = 53;
  costmap_grid.data[
    static_cast<std::size_t>(corridor_y + 6) * 60 + robot_x] = 100;

  FrontierExplorerCoreParams params;
  params.frontier_selection_min_distance = 0.8;
  params.frontier_visit_tolerance = 0.4;
  params.dispatch_clearance_radius_m = 0.344;
  params.dispatch_requires_known_free_costmap = true;
  params.occ_threshold = 90;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};
  FrontierExplorerCore core(params, callbacks);
  core.map = OccupancyGrid2d(map_grid);
  core.costmap = OccupancyGrid2d(costmap_grid);

  const double robot_world_x = -1.5 + (robot_x + 0.5) * 0.05;
  const double world_y = -1.5 + (corridor_y + 0.5) * 0.05;
  const double target_world_x = -1.5 + (44.5 * 0.05);
  const auto dispatch_point = core.resolve_dispatch_goal_point(
    make_frontier(target_world_x, world_y, 80),
    make_pose(robot_world_x, world_y),
    false);

  ASSERT_TRUE(dispatch_point.has_value());
  EXPECT_NEAR(dispatch_point->first, target_world_x, 1e-9);
  EXPECT_NEAR(dispatch_point->second, world_y, 1e-9);
}

TEST(DispatchGoalPointTests, KnownFreeCostmapGateRestrictsDispatchComponent)
{
  // Map: free disk around the robot, unknown beyond. Costmap: known free
  // only on the disk plus a forward (+x) strip. A frontier BEHIND the
  // robot resolves behind it without the gate (unknown map cells are
  // traversable), but with the gate the dispatch cell must stay inside
  // the costmap's known-free component: the forward strip.
  auto costmap_grid = build_footprint_seed_grid(-1);
  for (int y = 0; y < 40; ++y) {
    for (int x = 0; x < 40; ++x) {
      const double wx = -1.0 + (static_cast<double>(x) + 0.5) * 0.05;
      const double wy = -1.0 + (static_cast<double>(y) + 0.5) * 0.05;
      if (wx >= 0.0 && wx <= 0.7 && wy >= -0.2 && wy <= 0.2) {
        costmap_grid.data[
          static_cast<std::size_t>(y) * 40 + static_cast<std::size_t>(x)] = 0;
      }
    }
  }

  FrontierExplorerCoreParams params;
  params.frontier_selection_min_distance = 0.4;
  params.occ_threshold = 90;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};

  const auto pose = make_pose(0.0, 0.0);
  const auto behind_frontier = make_frontier(-0.21, -0.10, 80);

  FrontierExplorerCore ungated(params, callbacks);
  ungated.map = OccupancyGrid2d(build_footprint_seed_grid(-1));
  ungated.costmap = OccupancyGrid2d(costmap_grid);
  const auto ungated_point =
    ungated.resolve_dispatch_goal_point(behind_frontier, pose, false);
  ASSERT_TRUE(ungated_point.has_value());
  EXPECT_LT(ungated_point->first, 0.0);

  params.dispatch_requires_known_free_costmap = true;
  FrontierExplorerCore gated(params, callbacks);
  gated.map = OccupancyGrid2d(build_footprint_seed_grid(-1));
  gated.costmap = OccupancyGrid2d(costmap_grid);
  const auto gated_point =
    gated.resolve_dispatch_goal_point(behind_frontier, pose, false);
  ASSERT_TRUE(gated_point.has_value());
  EXPECT_GE(gated_point->first, 0.0);
  EXPECT_LE(gated_point->first, 0.75);
  EXPECT_GE(gated_point->second, -0.25);
  EXPECT_LE(gated_point->second, 0.25);
  const double robot_distance_sq =
    gated_point->first * gated_point->first +
    gated_point->second * gated_point->second;
  EXPECT_GE(robot_distance_sq, 0.4 * 0.4 - 1e-9);
}

TEST(DispatchGoalPointTests, KnownFreeGateInactiveWithoutCostmap)
{
  FrontierExplorerCoreParams params;
  params.frontier_selection_min_distance = 0.4;
  params.occ_threshold = 90;
  params.dispatch_requires_known_free_costmap = true;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};
  FrontierExplorerCore core(params, callbacks);
  core.map = OccupancyGrid2d(build_footprint_seed_grid(-1));
  // No costmap yet (startup): the gate must not starve dispatch.

  const auto dispatch_point = core.resolve_dispatch_goal_point(
    make_frontier(0.21, 0.10, 80),
    make_pose(0.0, 0.0),
    false);
  ASSERT_TRUE(dispatch_point.has_value());
}

TEST(DispatchGoalPointTests, CostmapCostPenaltyPrefersInteriorCells)
{
  // All-free map except the (occupied) target cell, so the BFS must pick
  // a nearby cell. The costmap charges cost 60 to the upper half-plane;
  // with the penalty the zero-cost lower neighbor wins over the
  // equally-near upper neighbors that the robot-distance tie-break would
  // otherwise select.
  auto map_grid = build_grid(40, 40, 0);
  map_grid.info.resolution = 0.05;
  map_grid.info.origin.position.x = -1.0;
  map_grid.info.origin.position.y = -1.0;
  map_grid.data[static_cast<std::size_t>(20) * 40 + 32] = 100;
  const double target_x = -1.0 + (32 + 0.5) * 0.05;
  const double target_y = -1.0 + (20 + 0.5) * 0.05;

  auto costmap_grid = build_grid(40, 40, 0);
  costmap_grid.info.resolution = 0.05;
  costmap_grid.info.origin.position.x = -1.0;
  costmap_grid.info.origin.position.y = -1.0;
  for (int y = 20; y < 40; ++y) {
    for (int x = 0; x < 40; ++x) {
      costmap_grid.data[
        static_cast<std::size_t>(y) * 40 + static_cast<std::size_t>(x)] = 60;
    }
  }
  costmap_grid.data[static_cast<std::size_t>(20) * 40 + 32] = 60;

  FrontierExplorerCoreParams params;
  params.frontier_selection_min_distance = 0.4;
  params.occ_threshold = 90;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};

  const auto pose = make_pose(0.0, 0.0);
  const auto frontier = make_frontier(target_x, target_y, 40);

  FrontierExplorerCore no_penalty(params, callbacks);
  no_penalty.map = OccupancyGrid2d(map_grid);
  no_penalty.costmap = OccupancyGrid2d(costmap_grid);
  const auto tie_break_point =
    no_penalty.resolve_dispatch_goal_point(frontier, pose, false);
  ASSERT_TRUE(tie_break_point.has_value());
  EXPECT_GT(tie_break_point->second, 0.0);

  params.dispatch_costmap_cost_penalty_m = 0.5;
  FrontierExplorerCore penalized(params, callbacks);
  penalized.map = OccupancyGrid2d(map_grid);
  penalized.costmap = OccupancyGrid2d(costmap_grid);
  const auto interior_point =
    penalized.resolve_dispatch_goal_point(frontier, pose, false);
  ASSERT_TRUE(interior_point.has_value());
  EXPECT_LT(interior_point->second, 0.0);
}

TEST(ExplorationBoundaryTests, FiltersFrontiersOutsideStartRadius)
{
  FrontierExplorerCoreParams params;
  params.exploration_boundary_radius_m = 1.0;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};

  FrontierExplorerCore core(params, callbacks);
  core.record_start_pose(make_pose(0.0, 0.0));

  const FrontierSequence filtered = core.filter_frontiers_for_boundary({
      make_frontier(0.5, 0.0),
      make_frontier(1.0, 0.0),
      make_frontier(1.5, 0.0),
    });

  ASSERT_EQ(filtered.size(), 2U);
  EXPECT_DOUBLE_EQ(core.frontier_position(filtered[0]).first, 0.5);
  EXPECT_DOUBLE_EQ(core.frontier_position(filtered[1]).first, 1.0);
}

TEST(ExplorationBoundaryTests, DispatchRejectsOutOfBoundaryTarget)
{
  FrontierExplorerCoreParams params;
  params.exploration_boundary_radius_m = 1.0;
  params.frontier_selection_min_distance = 0.0;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};

  FrontierExplorerCore core(params, callbacks);
  core.map = OccupancyGrid2d(build_grid(10, 10, 0));
  core.costmap = OccupancyGrid2d(build_grid(10, 10, 0));
  const auto start_pose = make_pose(1.0, 1.0);
  core.record_start_pose(start_pose);

  EXPECT_TRUE(core.resolve_dispatch_goal_point(
      make_frontier(1.5, 1.0),
      start_pose).has_value());
  EXPECT_FALSE(core.resolve_dispatch_goal_point(
      make_frontier(2.5, 1.0),
      start_pose).has_value());
}

class FrontierControlNodeTests : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (!rclcpp::ok()) {
      int argc = 0;
      rclcpp::init(argc, nullptr);
    }

    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    helper_node_ = std::make_shared<rclcpp::Node>("frontier_control_test_helper");
    executor_->add_node(helper_node_);
  }

  void TearDown() override
  {
    if (node_) {
      executor_->remove_node(node_);
      node_.reset();
    }
    if (helper_node_) {
      executor_->remove_node(helper_node_);
      helper_node_.reset();
    }
    executor_.reset();
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }

  void create_node(bool autostart, bool control_service_enabled = true)
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
      rclcpp::Parameter("autostart", autostart),
      rclcpp::Parameter("control_service_enabled", control_service_enabled),
      rclcpp::Parameter("completion_event_enabled", false),
      rclcpp::Parameter("frontier_suppression_enabled", false),
    });
    node_ = std::make_shared<FrontierExplorerNode>(options);
    executor_->add_node(node_);
  }

  std::shared_ptr<ControlExploration::Response> call_control_service(
    uint8_t action,
    double delay_seconds = 0.0,
    bool quit_after_stop = false)
  {
    auto client = helper_node_->create_client<ControlExploration>("/control_exploration");
    if (!client->wait_for_service(std::chrono::seconds(2))) {
      ADD_FAILURE() << "control_exploration service did not become ready";
      return nullptr;
    }

    auto request = std::make_shared<ControlExploration::Request>();
    request->action = action;
    request->delay_seconds = static_cast<float>(delay_seconds);
    request->quit_after_stop = quit_after_stop;

    auto future = client->async_send_request(request);
    if (
      executor_->spin_until_future_complete(future, std::chrono::seconds(2)) !=
      rclcpp::FutureReturnCode::SUCCESS)
    {
      ADD_FAILURE() << "control_exploration service call timed out";
      return nullptr;
    }
    return future.get();
  }

  bool wait_for_condition(
    const std::function<bool()> & predicate,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(1000))
  {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      executor_->spin_some();
      if (predicate()) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    executor_->spin_some();
    return predicate();
  }

  bool wait_for_control_service_availability(
    bool expected_available,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(1000))
  {
    auto client = helper_node_->create_client<ControlExploration>("/control_exploration");
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      executor_->spin_some();
      const bool available = client->wait_for_service(std::chrono::milliseconds(0));
      if (available == expected_available) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    executor_->spin_some();
    return client->wait_for_service(std::chrono::milliseconds(0)) == expected_available;
  }

  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  rclcpp::Node::SharedPtr helper_node_;
  std::shared_ptr<FrontierExplorerNode> node_;
};

TEST(ControlCliParserTests, ParsesImmediateStart)
{
  const auto parsed = parse_control_command_args({"start"});
  ASSERT_TRUE(parsed.ok);
  EXPECT_EQ(parsed.command.action, ControlExploration::Request::ACTION_START);
  EXPECT_DOUBLE_EQ(parsed.command.delay_seconds, 0.0);
  EXPECT_FALSE(parsed.command.quit_after_stop);
  EXPECT_EQ(parsed.command.service_name, "control_exploration");
}

TEST(ControlCliParserTests, ParsesDelayedStopWithQuit)
{
  const auto parsed = parse_control_command_args({"stop", "-t", "10", "-q"});
  ASSERT_TRUE(parsed.ok);
  EXPECT_EQ(parsed.command.action, ControlExploration::Request::ACTION_STOP);
  EXPECT_DOUBLE_EQ(parsed.command.delay_seconds, 10.0);
  EXPECT_TRUE(parsed.command.quit_after_stop);
}

TEST(ControlCliParserTests, RejectsQuitOnStart)
{
  const auto parsed = parse_control_command_args({"start", "-q"});
  EXPECT_FALSE(parsed.ok);
}

TEST_F(FrontierControlNodeTests, AutostartFalseKeepsSubscriptionsInactive)
{
  create_node(false);

  ASSERT_TRUE(wait_for_condition([this]() { return !node_->hasActiveExplorationSubscriptions(); }));
}

TEST_F(FrontierControlNodeTests, AutostartTrueCreatesSubscriptions)
{
  create_node(true);

  ASSERT_TRUE(wait_for_condition(
    [this]() { return node_->hasActiveExplorationSubscriptions(); },
    std::chrono::milliseconds(2000)));
}

TEST_F(FrontierControlNodeTests, ControlServiceCanBeDisabledWhenAutostartIsTrue)
{
  create_node(true, false);

  EXPECT_FALSE(node_->hasControlService());
  EXPECT_TRUE(wait_for_condition(
    [this]() { return node_->hasActiveExplorationSubscriptions(); },
    std::chrono::milliseconds(2000)));
  EXPECT_TRUE(wait_for_control_service_availability(false));
}

TEST_F(FrontierControlNodeTests, ColdIdleForcesControlServiceOn)
{
  create_node(false, false);

  EXPECT_TRUE(node_->hasControlService());
  EXPECT_TRUE(wait_for_condition([this]() { return !node_->hasActiveExplorationSubscriptions(); }));
  EXPECT_TRUE(wait_for_control_service_availability(true));
}

TEST_F(FrontierControlNodeTests, StartServiceActivatesSubscriptions)
{
  create_node(false);

  const auto response = call_control_service(ControlExploration::Request::ACTION_START);
  ASSERT_NE(response, nullptr);
  ASSERT_TRUE(response->accepted);
  EXPECT_FALSE(response->scheduled);
  EXPECT_EQ(response->state, ControlExploration::Request::STATE_RUNNING);

  ASSERT_TRUE(wait_for_condition([this]() { return node_->hasActiveExplorationSubscriptions(); }));
}

TEST_F(FrontierControlNodeTests, DelayedStartServiceActivatesSubscriptionsLater)
{
  create_node(false);

  const auto response = call_control_service(ControlExploration::Request::ACTION_START, 0.1);
  ASSERT_NE(response, nullptr);
  ASSERT_TRUE(response->accepted);
  EXPECT_TRUE(response->scheduled);
  EXPECT_EQ(response->state, ControlExploration::Request::STATE_START_SCHEDULED);

  ASSERT_TRUE(wait_for_condition(
    [this]() { return node_->hasActiveExplorationSubscriptions(); },
    std::chrono::milliseconds(2000)));
}

TEST_F(FrontierControlNodeTests, StopServiceReturnsNodeToColdIdle)
{
  create_node(true);
  ASSERT_TRUE(wait_for_condition([this]() { return node_->hasActiveExplorationSubscriptions(); }));

  const auto response = call_control_service(ControlExploration::Request::ACTION_STOP);
  ASSERT_NE(response, nullptr);
  ASSERT_TRUE(response->accepted);
  EXPECT_FALSE(response->scheduled);
  EXPECT_EQ(response->state, ControlExploration::Request::STATE_IDLE);

  ASSERT_TRUE(wait_for_condition([this]() { return !node_->hasActiveExplorationSubscriptions(); }));
}

TEST_F(FrontierControlNodeTests, DelayedStopIsRejectedWhileColdIdle)
{
  create_node(false);
  ASSERT_TRUE(wait_for_condition([this]() { return !node_->hasActiveExplorationSubscriptions(); }));

  const auto response = call_control_service(ControlExploration::Request::ACTION_STOP, 0.1);
  ASSERT_NE(response, nullptr);
  EXPECT_FALSE(response->accepted);
  EXPECT_FALSE(response->scheduled);
  EXPECT_EQ(response->state, ControlExploration::Request::STATE_IDLE);
}

TEST_F(FrontierControlNodeTests, ImmediateStartClearsPendingScheduledStop)
{
  create_node(true);
  ASSERT_TRUE(wait_for_condition([this]() { return node_->hasActiveExplorationSubscriptions(); }));

  const auto scheduled_stop = call_control_service(ControlExploration::Request::ACTION_STOP, 0.2);
  ASSERT_NE(scheduled_stop, nullptr);
  ASSERT_TRUE(scheduled_stop->accepted);
  EXPECT_TRUE(scheduled_stop->scheduled);
  EXPECT_EQ(scheduled_stop->state, ControlExploration::Request::STATE_STOP_SCHEDULED);

  const auto start_response = call_control_service(ControlExploration::Request::ACTION_START);
  ASSERT_NE(start_response, nullptr);
  EXPECT_TRUE(start_response->accepted);
  EXPECT_FALSE(start_response->scheduled);
  EXPECT_EQ(start_response->state, ControlExploration::Request::STATE_RUNNING);

  ASSERT_TRUE(wait_for_condition(
    [this]() { return node_->hasActiveExplorationSubscriptions(); },
    std::chrono::milliseconds(500)));
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  executor_->spin_some();
  EXPECT_TRUE(node_->hasActiveExplorationSubscriptions());
}

TEST_F(FrontierControlNodeTests, StopWithQuitRequestsOnlyExplorerExit)
{
  create_node(false);
  ASSERT_TRUE(wait_for_condition([this]() { return !node_->hasActiveExplorationSubscriptions(); }));

  const auto response = call_control_service(
    ControlExploration::Request::ACTION_STOP,
    0.0,
    true);
  ASSERT_NE(response, nullptr);
  EXPECT_TRUE(response->accepted);
  EXPECT_FALSE(response->scheduled);
  EXPECT_EQ(response->state, ControlExploration::Request::STATE_SHUTDOWN_PENDING);

  ASSERT_TRUE(wait_for_condition([this]() { return node_->quitRequested(); }));
  EXPECT_TRUE(rclcpp::ok());
}

double yaw_of(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(
    2.0 * (q.w * q.z + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

TEST(GoalYawPolicyTests, FaceFrontierAimsAtReferenceFromDispatchPoint)
{
  // The lone frontier hugs the robot, so dispatch resolves an adjusted
  // point at least frontier_selection_min_distance away. path_direction
  // faces travel; face_frontier turns back toward the frontier reference.
  FrontierExplorerCoreParams params;
  params.frontier_selection_min_distance = 0.4;
  params.occ_threshold = 90;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = []() {return int64_t{1'000'000'000};};

  const auto pose = make_pose(0.0, 0.0);
  const auto frontier = make_frontier(0.21, 0.10, 80);

  FrontierExplorerCore path_core(params, callbacks);
  path_core.map = OccupancyGrid2d(build_footprint_seed_grid(-1));
  const auto path_goal =
    path_core.build_dispatch_goal_pose(frontier, pose, false);

  params.goal_yaw_policy = "face_frontier";
  FrontierExplorerCore face_core(params, callbacks);
  face_core.map = OccupancyGrid2d(build_footprint_seed_grid(-1));
  const auto face_goal =
    face_core.build_dispatch_goal_pose(frontier, pose, false);

  // Same dispatch position under both policies.
  EXPECT_DOUBLE_EQ(path_goal.pose.position.x, face_goal.pose.position.x);
  EXPECT_DOUBLE_EQ(path_goal.pose.position.y, face_goal.pose.position.y);

  const double path_yaw = yaw_of(path_goal.pose.orientation);
  const double expected_path_yaw = std::atan2(
    path_goal.pose.position.y - 0.0, path_goal.pose.position.x - 0.0);
  EXPECT_NEAR(path_yaw, expected_path_yaw, 1e-6);

  const double face_yaw = yaw_of(face_goal.pose.orientation);
  const double expected_face_yaw = std::atan2(
    0.10 - face_goal.pose.position.y, 0.21 - face_goal.pose.position.x);
  EXPECT_NEAR(face_yaw, expected_face_yaw, 1e-6);
  // The two policies genuinely differ here: the dispatch point sits
  // beyond the frontier, so facing it means looking back — more than
  // 90 degrees away from the travel heading.
  EXPECT_GT(
    std::abs(std::atan2(
      std::sin(face_yaw - path_yaw), std::cos(face_yaw - path_yaw))),
    1.5707);
}
}  // namespace
}  // namespace frontier_exploration_ros2
