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

#include <action_msgs/msg/goal_status.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "frontier_exploration_ros2/frontier_suppression.hpp"
#include "frontier_exploration_ros2/frontier_explorer_core.hpp"

namespace frontier_exploration_ros2
{
namespace
{

class FakeGoalHandle : public GoalHandleInterface
{
public:
  void cancel_goal_async(
    std::function<void(bool accepted, const std::string & error_message)> callback) override
  {
    cancel_calls += 1;
    pending_callback = std::move(callback);
  }

  void resolve_cancel(bool accepted = true, const std::string & error_message = "")
  {
    if (pending_callback) {
      pending_callback(accepted, error_message);
      pending_callback = nullptr;
    }
  }

  int cancel_calls{0};

private:
  std::function<void(bool accepted, const std::string & error_message)> pending_callback;
};

geometry_msgs::msg::Pose make_pose(double x = 0.0, double y = 0.0)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.orientation.w = 1.0;
  return pose;
}

FrontierCandidate make_candidate(double x, double y)
{
  return FrontierCandidate{{x, y}, {x, y}, 8};
}

nav_msgs::msg::OccupancyGrid build_grid(
  int width, int height, int default_value, double resolution = 1.0)
{
  nav_msgs::msg::OccupancyGrid msg;
  msg.info.width = static_cast<uint32_t>(width);
  msg.info.height = static_cast<uint32_t>(height);
  msg.info.resolution = resolution;
  msg.info.origin.position.x = 0.0;
  msg.info.origin.position.y = 0.0;
  msg.info.origin.orientation.w = 1.0;
  msg.data.assign(static_cast<std::size_t>(width * height), static_cast<int8_t>(default_value));
  return msg;
}

void set_grid_cell(nav_msgs::msg::OccupancyGrid & msg, int x, int y, int8_t value)
{
  msg.data[static_cast<std::size_t>(y) * msg.info.width + static_cast<std::size_t>(x)] = value;
}

std::unique_ptr<FrontierExplorerCore> make_suppression_core(
  int64_t * now_ns,
  int * dispatch_calls,
  std::vector<std::string> * info_logs = nullptr,
  std::vector<std::string> * warn_logs = nullptr,
  geometry_msgs::msg::Pose * current_pose = nullptr)
{
  FrontierExplorerCoreParams params;
  params.frontier_suppression_enabled = true;
  params.frontier_suppression_attempt_threshold = 1;
  params.frontier_suppression_timeout_s = 90.0;
  params.frontier_suppression_no_progress_timeout_s = 5.0;
  params.frontier_suppression_progress_epsilon_m = 0.5;
  params.frontier_suppression_startup_grace_period_s = 0.0;
  params.return_to_start_on_complete = false;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = [now_ns]() {return *now_ns;};
  callbacks.get_current_pose = [current_pose]() {
      return std::optional<geometry_msgs::msg::Pose>(
        current_pose == nullptr ? make_pose(1.0, 1.0) : *current_pose);
    };
  callbacks.wait_for_action_server = [](double) {return true;};
  callbacks.dispatch_goal_request = [dispatch_calls](const GoalDispatchRequest &) {
      *dispatch_calls += 1;
    };
  callbacks.log_info = [info_logs](const std::string & message) {
      if (info_logs) {
        info_logs->push_back(message);
      }
    };
  callbacks.log_warn = [warn_logs](const std::string & message) {
      if (warn_logs) {
        warn_logs->push_back(message);
      }
    };
  callbacks.log_debug = [](const std::string &) {};
  callbacks.log_error = [](const std::string &) {};
  callbacks.frontier_search = [](
    const geometry_msgs::msg::Pose &,
    const OccupancyGrid2d &,
    const OccupancyGrid2d &,
    const std::optional<OccupancyGrid2d> &,
    double,
    bool)
    {
      FrontierSearchResult result;
      result.frontiers = {make_candidate(4.0, 4.0)};
      result.robot_map_cell = {1, 1};
      return result;
    };

  auto core = std::make_unique<FrontierExplorerCore>(params, callbacks);
  auto map_msg = build_grid(20, 20, 0);
  auto costmap_msg = build_grid(20, 20, 0);
  core->map = OccupancyGrid2d(map_msg);
  core->costmap = OccupancyGrid2d(costmap_msg);
  core->map_generation = 1;
  core->costmap_generation = 1;
  core->local_costmap_generation = 0;
  return core;
}

TEST(FrontierSuppressionTests, ThresholdCreatesRegionAndClearsAttempt)
{
  FrontierSuppression suppression(FrontierSuppressionConfig{
    2, 2.0, 1.0, 90.0, 20.0, 0.25, 8, 4, 0.3});

  suppression.record_failed_attempt(make_candidate(2.0, 2.0), 1'000'000'000);
  EXPECT_EQ(suppression.attempt_count(), 1U);
  EXPECT_EQ(suppression.region_count(), 0U);

  suppression.record_failed_attempt(make_candidate(2.0, 2.0), 2'000'000'000);
  EXPECT_EQ(suppression.attempt_count(), 0U);
  ASSERT_EQ(suppression.region_count(), 1U);
  EXPECT_DOUBLE_EQ(suppression.regions().front().center.first, 2.0);
  EXPECT_DOUBLE_EQ(suppression.regions().front().side_length_m, 2.0);
}

TEST(FrontierSuppressionTests, ExpansionRingDoublesRegionAndMovesCenter)
{
  FrontierSuppression suppression(FrontierSuppressionConfig{
    1, 2.0, 2.0, 90.0, 20.0, 0.25, 8, 4, 0.3});

  suppression.record_failed_attempt(make_candidate(0.0, 0.0), 1'000'000'000);
  suppression.record_failed_attempt(make_candidate(1.5, 0.0), 2'000'000'000);

  ASSERT_EQ(suppression.region_count(), 1U);
  EXPECT_DOUBLE_EQ(suppression.regions().front().center.first, 0.75);
  EXPECT_DOUBLE_EQ(suppression.regions().front().center.second, 0.0);
  EXPECT_DOUBLE_EQ(suppression.regions().front().side_length_m, 4.0);
}

TEST(FrontierSuppressionTests, ExpansionSizeIsRingWidthOnEachSide)
{
  FrontierSuppression suppression(FrontierSuppressionConfig{
    1, 2.0, 1.0, 90.0, 20.0, 0.25, 8, 4, 0.3});

  suppression.record_failed_attempt(make_candidate(0.0, 0.0), 1'000'000'000);
  suppression.record_failed_attempt(make_candidate(1.75, 0.0), 2'000'000'000);

  ASSERT_EQ(suppression.region_count(), 1U);
  EXPECT_DOUBLE_EQ(suppression.regions().front().center.first, 0.875);
  EXPECT_DOUBLE_EQ(suppression.regions().front().center.second, 0.0);
  EXPECT_DOUBLE_EQ(suppression.regions().front().side_length_m, 4.0);
}

TEST(FrontierSuppressionTests, PruneExpiredRemovesAttemptsAndRegions)
{
  FrontierSuppression suppression(FrontierSuppressionConfig{
    2, 2.0, 1.0, 5.0, 20.0, 0.25, 8, 4, 0.3});

  suppression.record_failed_attempt(make_candidate(2.0, 2.0), 0);
  EXPECT_EQ(suppression.attempt_count(), 1U);
  suppression.prune_expired(6'000'000'000);
  EXPECT_EQ(suppression.attempt_count(), 0U);

  suppression.record_failed_attempt(make_candidate(3.0, 3.0), 7'000'000'000);
  suppression.record_failed_attempt(make_candidate(3.0, 3.0), 8'000'000'000);
  ASSERT_EQ(suppression.region_count(), 1U);
  suppression.prune_expired(14'000'000'000);
  EXPECT_EQ(suppression.region_count(), 0U);
}

TEST(FrontierSuppressionTests, AttemptCapacityEvictsOldestRecord)
{
  FrontierSuppression suppression(FrontierSuppressionConfig{
    3, 2.0, 1.0, 90.0, 20.0, 0.25, 2, 4, 0.3});
  std::vector<std::string> warn_logs;
  const auto log_warn = [&warn_logs](const std::string & message) {warn_logs.push_back(message);};

  suppression.record_failed_attempt(make_candidate(1.0, 1.0), 1'000'000'000, log_warn);
  suppression.record_failed_attempt(make_candidate(2.0, 2.0), 2'000'000'000, log_warn);
  suppression.record_failed_attempt(make_candidate(3.0, 3.0), 3'000'000'000, log_warn);

  EXPECT_EQ(suppression.attempt_count(), 2U);
  EXPECT_FALSE(warn_logs.empty());
}

TEST(FrontierSuppressionTests, NoProgressTimeoutTriggersCancelRequest)
{
  FrontierSuppression suppression(FrontierSuppressionConfig{
    3, 2.0, 1.0, 90.0, 5.0, 0.5, 8, 4, 0.3});

  suppression.start_goal_progress_tracking(3, 0);
  suppression.note_goal_progress(3, 10.0, 0);
  EXPECT_FALSE(suppression.mark_timeout_cancel_if_needed(3, 4'000'000'000));
  EXPECT_TRUE(suppression.mark_timeout_cancel_if_needed(3, 6'000'000'000));
  EXPECT_TRUE(suppression.progress_timeout_cancel_requested());
}

TEST(FrontierSuppressionTests, MeaningfulProgressResetsTimeout)
{
  FrontierSuppression suppression(FrontierSuppressionConfig{
    3, 2.0, 1.0, 90.0, 5.0, 0.5, 8, 4, 0.3});

  suppression.start_goal_progress_tracking(7, 0);
  suppression.note_goal_progress(7, 10.0, 0);
  suppression.note_goal_progress(7, 9.4, 3'000'000'000);
  EXPECT_FALSE(suppression.mark_timeout_cancel_if_needed(7, 7'000'000'000));
  EXPECT_TRUE(suppression.mark_timeout_cancel_if_needed(7, 9'000'000'000));
}

TEST(FrontierSuppressionCoreTests, RejectedFrontierBecomesSuppressedAndStopsRedispatch)
{
  int64_t now_ns = 1'000'000'000;
  int dispatch_calls = 0;
  std::vector<std::string> info_logs;
  auto core = make_suppression_core(&now_ns, &dispatch_calls, &info_logs, nullptr);

  core->try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 1);
  EXPECT_TRUE(core->suppression_state_allocated());

  core->goal_response_callback(core->current_dispatch_id, nullptr, false, "rejected");
  EXPECT_TRUE(core->suppression_state_allocated());
  EXPECT_EQ(core->suppressed_region_count(), 1U);

  core->try_send_next_goal();
  EXPECT_EQ(dispatch_calls, 1);
  EXPECT_FALSE(info_logs.empty());
}

TEST(FrontierSuppressionCoreTests, NoProgressTimeoutCanceledGoalCreatesSuppressedRegion)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  auto core = make_suppression_core(&now_ns, &dispatch_calls, nullptr, nullptr);

  core->try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 1);
  auto fake_handle = std::make_shared<FakeGoalHandle>();
  core->goal_response_callback(core->current_dispatch_id, fake_handle, true, "");
  core->feedback_callback(10.0, core->current_dispatch_id);

  now_ns = 6'000'000'000;
  EXPECT_TRUE(core->evaluate_active_goal_progress_timeout());
  EXPECT_EQ(fake_handle->cancel_calls, 1);
  fake_handle->resolve_cancel(true, "");
  core->get_result_callback(
    core->current_dispatch_id,
    action_msgs::msg::GoalStatus::STATUS_CANCELED,
    0,
    "");

  EXPECT_EQ(core->suppressed_region_count(), 1U);
}

TEST(FrontierSuppressionCoreTests, EuclideanGoalProgressDelaysNoProgressCancel)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  geometry_msgs::msg::Pose current_pose = make_pose(1.0, 1.0);
  auto core = make_suppression_core(
    &now_ns,
    &dispatch_calls,
    nullptr,
    nullptr,
    &current_pose);

  core->try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 1);
  auto fake_handle = std::make_shared<FakeGoalHandle>();
  core->goal_response_callback(core->current_dispatch_id, fake_handle, true, "");
  core->feedback_callback(10.0, core->current_dispatch_id);

  now_ns = 6'000'000'000;
  current_pose = make_pose(2.0, 2.0);
  EXPECT_FALSE(core->evaluate_active_goal_progress_timeout());
  EXPECT_EQ(fake_handle->cancel_calls, 0);

  now_ns = 12'000'000'000;
  EXPECT_TRUE(core->evaluate_active_goal_progress_timeout());
  EXPECT_EQ(fake_handle->cancel_calls, 1);
}

TEST(FrontierSuppressionCoreTests, AbortedFrontierSuppressesRegionBeforeRetryThreshold)
{
  int64_t now_ns = 1'000'000'000;
  std::vector<GoalDispatchRequest> dispatched_requests;

  FrontierExplorerCoreParams params;
  params.frontier_suppression_enabled = true;
  params.frontier_suppression_attempt_threshold = 10;
  params.frontier_suppression_startup_grace_period_s = 0.0;
  params.post_goal_settle_enabled = false;
  params.return_to_start_on_complete = false;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = [&now_ns]() {return now_ns;};
  callbacks.get_current_pose = []() {
      return std::optional<geometry_msgs::msg::Pose>(make_pose(1.0, 1.0));
    };
  callbacks.wait_for_action_server = [](double) {return true;};
  callbacks.dispatch_goal_request = [&dispatched_requests](const GoalDispatchRequest & request) {
      dispatched_requests.push_back(request);
    };
  callbacks.log_info = [](const std::string &) {};
  callbacks.log_warn = [](const std::string &) {};
  callbacks.log_debug = [](const std::string &) {};
  callbacks.log_error = [](const std::string &) {};
  callbacks.frontier_search = [](
    const geometry_msgs::msg::Pose &,
    const OccupancyGrid2d &,
    const OccupancyGrid2d &,
    const std::optional<OccupancyGrid2d> &,
    double,
    bool)
    {
      FrontierSearchResult result;
      result.frontiers = {make_candidate(4.0, 4.0), make_candidate(8.0, 8.0)};
      result.robot_map_cell = {1, 1};
      return result;
    };

  FrontierExplorerCore core(params, callbacks);
  auto map_msg = build_grid(20, 20, 0);
  auto costmap_msg = build_grid(20, 20, 0);
  core.map = OccupancyGrid2d(map_msg);
  core.costmap = OccupancyGrid2d(costmap_msg);
  core.map_generation = 1;
  core.costmap_generation = 1;
  core.local_costmap_generation = 0;

  core.try_send_next_goal();
  ASSERT_EQ(dispatched_requests.size(), 1U);
  ASSERT_TRUE(dispatched_requests.back().frontier.has_value());
  EXPECT_EQ(
    frontier_position(*dispatched_requests.back().frontier),
    (std::pair<double, double>{4.0, 4.0}));

  auto fake_handle = std::make_shared<FakeGoalHandle>();
  core.goal_response_callback(core.current_dispatch_id, fake_handle, true, "");
  core.get_result_callback(
    core.current_dispatch_id,
    action_msgs::msg::GoalStatus::STATUS_ABORTED,
    208,
    "");

  EXPECT_EQ(core.suppressed_region_count(), 1U);
  ASSERT_EQ(dispatched_requests.size(), 2U);
  ASSERT_TRUE(dispatched_requests.back().frontier.has_value());
  EXPECT_EQ(
    frontier_position(*dispatched_requests.back().frontier),
    (std::pair<double, double>{8.0, 8.0}));
}

TEST(FrontierSuppressionCoreTests, AllSuppressedCanDispatchTemporaryReturnToStart)
{
  int64_t now_ns = 1'000'000'000;
  int dispatch_calls = 0;
  geometry_msgs::msg::Pose current_pose = make_pose(1.0, 1.0);
  std::vector<std::string> dispatched_goal_kinds;

  FrontierExplorerCoreParams params;
  params.frontier_suppression_enabled = true;
  params.frontier_suppression_attempt_threshold = 1;
  params.frontier_suppression_startup_grace_period_s = 0.0;
  params.all_frontiers_suppressed_behavior = "return_to_start";
  params.return_to_start_on_complete = false;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = [&now_ns]() {return now_ns;};
  callbacks.get_current_pose = [&current_pose]() {
      return std::optional<geometry_msgs::msg::Pose>(current_pose);
    };
  callbacks.wait_for_action_server = [](double) {return true;};
  callbacks.dispatch_goal_request = [&dispatch_calls, &dispatched_goal_kinds](const GoalDispatchRequest & request) {
      dispatch_calls += 1;
      dispatched_goal_kinds.push_back(request.goal_kind);
    };
  callbacks.log_info = [](const std::string &) {};
  callbacks.log_warn = [](const std::string &) {};
  callbacks.log_debug = [](const std::string &) {};
  callbacks.log_error = [](const std::string &) {};
  callbacks.frontier_search = [](
    const geometry_msgs::msg::Pose &,
    const OccupancyGrid2d &,
    const OccupancyGrid2d &,
    const std::optional<OccupancyGrid2d> &,
    double,
    bool)
    {
      FrontierSearchResult result;
      result.frontiers = {make_candidate(4.0, 4.0)};
      result.robot_map_cell = {1, 1};
      return result;
    };

  FrontierExplorerCore core(params, callbacks);
  auto map_msg = build_grid(20, 20, 0);
  auto costmap_msg = build_grid(20, 20, 0);
  core.map = OccupancyGrid2d(map_msg);
  core.costmap = OccupancyGrid2d(costmap_msg);
  core.map_generation = 1;
  core.costmap_generation = 1;

  core.try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 1);
  EXPECT_EQ(dispatched_goal_kinds.back(), "frontier");

  core.goal_response_callback(core.current_dispatch_id, nullptr, false, "rejected");
  current_pose = make_pose(3.0, 3.0);
  core.try_send_next_goal();

  ASSERT_EQ(dispatch_calls, 2);
  EXPECT_EQ(dispatched_goal_kinds.back(), "suppressed_return_to_start");
  EXPECT_FALSE(core.return_to_start_completed);
}

TEST(FrontierSuppressionCoreTests, SuppressedReturnFailureWaitsForNewFrontiersBeforeRetry)
{
  int64_t now_ns = 1'000'000'000;
  int dispatch_calls = 0;
  geometry_msgs::msg::Pose current_pose = make_pose(1.0, 1.0);
  std::vector<std::string> dispatched_goal_kinds;
  bool use_suppressed_frontier = true;

  FrontierExplorerCoreParams params;
  params.frontier_suppression_enabled = true;
  params.frontier_suppression_attempt_threshold = 1;
  params.frontier_suppression_startup_grace_period_s = 0.0;
  params.all_frontiers_suppressed_behavior = "return_to_start";
  params.return_to_start_on_complete = false;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = [&now_ns]() {return now_ns;};
  callbacks.get_current_pose = [&current_pose]() {
      return std::optional<geometry_msgs::msg::Pose>(current_pose);
    };
  callbacks.wait_for_action_server = [](double) {return true;};
  callbacks.dispatch_goal_request = [&dispatch_calls, &dispatched_goal_kinds](const GoalDispatchRequest & request) {
      dispatch_calls += 1;
      dispatched_goal_kinds.push_back(request.goal_kind);
    };
  callbacks.log_info = [](const std::string &) {};
  callbacks.log_warn = [](const std::string &) {};
  callbacks.log_debug = [](const std::string &) {};
  callbacks.log_error = [](const std::string &) {};
  callbacks.frontier_search = [&use_suppressed_frontier](
    const geometry_msgs::msg::Pose &,
    const OccupancyGrid2d &,
    const OccupancyGrid2d &,
    const std::optional<OccupancyGrid2d> &,
    double,
    bool)
    {
      FrontierSearchResult result;
      result.frontiers = {
        use_suppressed_frontier ? make_candidate(4.0, 4.0) : make_candidate(8.0, 8.0)};
      result.robot_map_cell = {1, 1};
      return result;
    };

  FrontierExplorerCore core(params, callbacks);
  auto map_msg = build_grid(20, 20, 0);
  auto costmap_msg = build_grid(20, 20, 0);
  core.map = OccupancyGrid2d(map_msg);
  core.costmap = OccupancyGrid2d(costmap_msg);
  core.map_generation = 1;
  core.costmap_generation = 1;

  core.try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 1);
  core.goal_response_callback(core.current_dispatch_id, nullptr, false, "rejected");

  current_pose = make_pose(3.0, 3.0);
  core.try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 2);
  ASSERT_EQ(dispatched_goal_kinds.back(), "suppressed_return_to_start");

  auto fake_handle = std::make_shared<FakeGoalHandle>();
  const int return_dispatch_id = core.current_dispatch_id;
  core.goal_response_callback(return_dispatch_id, fake_handle, true, "");
  core.get_result_callback(
    return_dispatch_id,
    action_msgs::msg::GoalStatus::STATUS_ABORTED,
    208,
    "");

  core.try_send_next_goal();
  EXPECT_EQ(dispatch_calls, 2);

  use_suppressed_frontier = false;
  core.try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 3);
  EXPECT_EQ(dispatched_goal_kinds.back(), "frontier");
}

TEST(FrontierSuppressionCoreTests, AllSuppressedCanCompleteExploration)
{
  int64_t now_ns = 1'000'000'000;
  int dispatch_calls = 0;
  int completion_calls = 0;
  geometry_msgs::msg::Pose current_pose = make_pose(1.0, 1.0);

  FrontierExplorerCoreParams params;
  params.frontier_suppression_enabled = true;
  params.frontier_suppression_attempt_threshold = 1;
  params.frontier_suppression_startup_grace_period_s = 0.0;
  params.all_frontiers_suppressed_behavior = "complete";
  params.return_to_start_on_complete = false;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = [&now_ns]() {return now_ns;};
  callbacks.get_current_pose = [&current_pose]() {
      return std::optional<geometry_msgs::msg::Pose>(current_pose);
    };
  callbacks.wait_for_action_server = [](double) {return true;};
  callbacks.dispatch_goal_request = [&dispatch_calls](const GoalDispatchRequest &) {
      dispatch_calls += 1;
    };
  callbacks.on_exploration_complete = [&completion_calls]() {
      completion_calls += 1;
    };
  callbacks.log_info = [](const std::string &) {};
  callbacks.log_warn = [](const std::string &) {};
  callbacks.log_debug = [](const std::string &) {};
  callbacks.log_error = [](const std::string &) {};
  callbacks.frontier_search = [](
    const geometry_msgs::msg::Pose &,
    const OccupancyGrid2d &,
    const OccupancyGrid2d &,
    const std::optional<OccupancyGrid2d> &,
    double,
    bool)
    {
      FrontierSearchResult result;
      result.frontiers = {make_candidate(4.0, 4.0)};
      result.robot_map_cell = {1, 1};
      return result;
    };

  FrontierExplorerCore core(params, callbacks);
  auto map_msg = build_grid(20, 20, 0);
  auto costmap_msg = build_grid(20, 20, 0);
  core.map = OccupancyGrid2d(map_msg);
  core.costmap = OccupancyGrid2d(costmap_msg);
  core.map_generation = 1;
  core.costmap_generation = 1;

  core.try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 1);

  core.goal_response_callback(core.current_dispatch_id, nullptr, false, "rejected");
  current_pose = make_pose(3.0, 3.0);
  core.try_send_next_goal();

  EXPECT_EQ(dispatch_calls, 1);
  EXPECT_EQ(completion_calls, 1);
  EXPECT_TRUE(core.return_to_start_completed);
}

TEST(FrontierSuppressionCoreTests, TemporaryReturnToStartPreemptsWhenFrontiersBecomeAvailableAgain)
{
  int64_t now_ns = 1'000'000'000;
  int dispatch_calls = 0;
  geometry_msgs::msg::Pose current_pose = make_pose(1.0, 1.0);
  std::vector<std::string> dispatched_goal_kinds;
  bool use_suppressed_frontier = true;

  FrontierExplorerCoreParams params;
  params.frontier_suppression_enabled = true;
  params.frontier_suppression_attempt_threshold = 1;
  params.frontier_suppression_startup_grace_period_s = 0.0;
  params.all_frontiers_suppressed_behavior = "return_to_start";
  params.return_to_start_on_complete = false;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = [&now_ns]() {return now_ns;};
  callbacks.get_current_pose = [&current_pose]() {
      return std::optional<geometry_msgs::msg::Pose>(current_pose);
    };
  callbacks.wait_for_action_server = [](double) {return true;};
  callbacks.dispatch_goal_request = [&dispatch_calls, &dispatched_goal_kinds](const GoalDispatchRequest & request) {
      dispatch_calls += 1;
      dispatched_goal_kinds.push_back(request.goal_kind);
    };
  callbacks.log_info = [](const std::string &) {};
  callbacks.log_warn = [](const std::string &) {};
  callbacks.log_debug = [](const std::string &) {};
  callbacks.log_error = [](const std::string &) {};
  callbacks.frontier_search = [&use_suppressed_frontier](
    const geometry_msgs::msg::Pose &,
    const OccupancyGrid2d &,
    const OccupancyGrid2d &,
    const std::optional<OccupancyGrid2d> &,
    double,
    bool)
    {
      FrontierSearchResult result;
      result.frontiers = {
        use_suppressed_frontier ? make_candidate(4.0, 4.0) : make_candidate(8.0, 8.0)};
      result.robot_map_cell = {1, 1};
      return result;
    };

  FrontierExplorerCore core(params, callbacks);
  auto map_msg = build_grid(20, 20, 0);
  auto costmap_msg = build_grid(20, 20, 0);
  core.map = OccupancyGrid2d(map_msg);
  core.costmap = OccupancyGrid2d(costmap_msg);
  core.map_generation = 1;
  core.costmap_generation = 1;

  core.try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 1);
  core.goal_response_callback(core.current_dispatch_id, nullptr, false, "rejected");

  current_pose = make_pose(3.0, 3.0);
  core.try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 2);
  ASSERT_EQ(dispatched_goal_kinds.back(), "suppressed_return_to_start");

  auto fake_handle = std::make_shared<FakeGoalHandle>();
  core.goal_response_callback(core.current_dispatch_id, fake_handle, true, "");

  use_suppressed_frontier = false;
  core.occupancyGridCallback(OccupancyGrid2d(map_msg));
  EXPECT_EQ(fake_handle->cancel_calls, 0);
  EXPECT_TRUE(core.pending_frontier_sequence.empty());

  ASSERT_EQ(dispatch_calls, 3);
  EXPECT_EQ(dispatched_goal_kinds.back(), "frontier");
}

TEST(FrontierSuppressionCoreTests, StartupGracePeriodDefersSuppressionFailures)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  FrontierExplorerCoreParams params;
  params.frontier_suppression_enabled = true;
  params.frontier_suppression_attempt_threshold = 1;
  params.frontier_suppression_startup_grace_period_s = 5.0;
  params.return_to_start_on_complete = false;

  FrontierExplorerCoreCallbacks callbacks;
  callbacks.now_ns = [&now_ns]() {return now_ns;};
  callbacks.get_current_pose = []() {
      return std::optional<geometry_msgs::msg::Pose>(make_pose(1.0, 1.0));
    };
  callbacks.wait_for_action_server = [](double) {return true;};
  callbacks.dispatch_goal_request = [&dispatch_calls](const GoalDispatchRequest &) {
      dispatch_calls += 1;
    };
  callbacks.log_debug = [](const std::string &) {};
  callbacks.log_info = [](const std::string &) {};
  callbacks.log_warn = [](const std::string &) {};
  callbacks.log_error = [](const std::string &) {};
  callbacks.frontier_search = [](
    const geometry_msgs::msg::Pose &,
    const OccupancyGrid2d &,
    const OccupancyGrid2d &,
    const std::optional<OccupancyGrid2d> &,
    double,
    bool)
    {
      FrontierSearchResult result;
      result.frontiers = {make_candidate(4.0, 4.0)};
      result.robot_map_cell = {1, 1};
      return result;
    };

  FrontierExplorerCore core(params, callbacks);
  auto map_msg = build_grid(20, 20, 0);
  auto costmap_msg = build_grid(20, 20, 0);
  core.map = OccupancyGrid2d(map_msg);
  core.costmap = OccupancyGrid2d(costmap_msg);
  core.map_generation = 1;
  core.costmap_generation = 1;

  core.try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 1);
  core.goal_response_callback(core.current_dispatch_id, nullptr, false, "rejected");
  EXPECT_FALSE(core.suppression_state_allocated());
  EXPECT_EQ(core.suppressed_region_count(), 0U);

  now_ns = 6'000'000'000;
  core.try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 2);
  core.goal_response_callback(core.current_dispatch_id, nullptr, false, "rejected");
  EXPECT_TRUE(core.suppression_state_allocated());
  EXPECT_EQ(core.suppressed_region_count(), 1U);
}

TEST(FrontierSuppressionCoreTests, SafetyBlockBypassesStartupGraceAndStopsRedispatch)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  auto core = make_suppression_core(&now_ns, &dispatch_calls);
  core->params.frontier_suppression_startup_grace_period_s = 15.0;
  core->start_exploration_session();

  auto map_msg = build_grid(20, 20, 0);
  auto costmap_msg = build_grid(20, 20, 0);
  core->map = OccupancyGrid2d(map_msg);
  core->costmap = OccupancyGrid2d(costmap_msg);
  core->map_generation = 1;
  core->costmap_generation = 1;

  core->try_send_next_goal();
  ASSERT_EQ(dispatch_calls, 1);
  auto fake_handle = std::make_shared<FakeGoalHandle>();
  core->goal_response_callback(core->current_dispatch_id, fake_handle, true, "");

  core->handle_navigation_blocked_event("depth guard stop");

  EXPECT_EQ(core->suppressed_region_count(), 1U);
  EXPECT_EQ(fake_handle->cancel_calls, 1);
  fake_handle->resolve_cancel(true, "");
  core->get_result_callback(
    core->current_dispatch_id,
    action_msgs::msg::GoalStatus::STATUS_CANCELED,
    0,
    "");
  core->try_send_next_goal();
  EXPECT_EQ(dispatch_calls, 1);
}

TEST(FrontierSuppressionCoreTests, AttributedCollisionStopRequestsFrontierBlock)
{
  EXPECT_TRUE(attributed_watchdog_stop_requests_frontier_block(
    R"({"event":"collision_stop","recent_safety_stops":1})", 1));
}

TEST(FrontierSuppressionCoreTests, AttributedDepthStopRequestsFrontierBlock)
{
  EXPECT_TRUE(attributed_watchdog_stop_requests_frontier_block(
    R"({"event":"depth_guard_stop","recent_safety_stops":1})", 1));
}

TEST(FrontierSuppressionCoreTests, UnknownStopNeverRequestsFrontierBlock)
{
  EXPECT_FALSE(attributed_watchdog_stop_requests_frontier_block(
    R"({"event":"safety_stop_unknown","recent_safety_stops":4})", 1));
}

TEST(FrontierSuppressionCoreTests, AttributedStopHonorsConfiguredThreshold)
{
  EXPECT_FALSE(attributed_watchdog_stop_requests_frontier_block(
    R"({"event":"depth_guard_stop","recent_safety_stops":1})", 2));
  EXPECT_TRUE(attributed_watchdog_stop_requests_frontier_block(
    R"({"event":"depth_guard_stop","recent_safety_stops":2})", 2));
}

TEST(FrontierSuppressionCoreTests, InitialSuppressedPointSurvivesWorkerRestart)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  auto core = make_suppression_core(&now_ns, &dispatch_calls);
  core->params.initial_suppressed_points = {{4.0, 4.0}};

  core->start_exploration_session();
  EXPECT_EQ(core->suppressed_region_count(), 1U);
  core->try_send_next_goal();
  EXPECT_EQ(dispatch_calls, 0);
}

TEST(FrontierDispatchResolutionTests, PreservesReachableFrontierGoal)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  auto core = make_suppression_core(&now_ns, &dispatch_calls);
  auto map_msg = build_grid(10, 5, 0);
  auto costmap_msg = build_grid(10, 5, 0);
  core->map = OccupancyGrid2d(map_msg);
  core->costmap = OccupancyGrid2d(costmap_msg);
  core->map_generation = 1;
  core->costmap_generation = 1;

  const auto resolved = core->resolve_dispatch_goal_point(
    make_candidate(8.5, 2.5),
    make_pose(1.5, 2.5));

  ASSERT_TRUE(resolved.has_value());
  EXPECT_DOUBLE_EQ(resolved->first, 8.5);
  EXPECT_DOUBLE_EQ(resolved->second, 2.5);
}

TEST(FrontierDispatchResolutionTests, NoDispatchableCandidateSkipsWithoutSuppressionAndTriesFallback)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  auto core = make_suppression_core(&now_ns, &dispatch_calls);
  core->params.frontier_suppression_attempt_threshold = 2;
  core->params.frontier_selection_min_distance = 2.0;

  std::optional<GoalDispatchRequest> captured_request;
  core->callbacks.dispatch_goal_request =
    [&dispatch_calls, &captured_request](const GoalDispatchRequest & request) {
      dispatch_calls += 1;
      captured_request = request;
    };

  // The first candidate lies outside the map grid, so no dispatch cell
  // can be resolved for it. (A merely too-close candidate no longer
  // qualifies: dispatch now adjusts those to a min-distance cell.)
  const FrontierSequence frontiers{
    make_candidate(-3.0, 1.0),
    make_candidate(5.0, 5.0),
  };

  const bool dispatched = core->send_frontier_goal(
    frontiers,
    make_pose(1.0, 1.0),
    "test dispatch");

  EXPECT_TRUE(dispatched);
  EXPECT_EQ(dispatch_calls, 1);
  EXPECT_EQ(core->suppression_attempt_count(), 0U);
  EXPECT_EQ(core->suppressed_region_count(), 0U);
  ASSERT_TRUE(captured_request.has_value());
  ASSERT_TRUE(captured_request->frontier.has_value());
  EXPECT_TRUE(core->are_frontiers_equivalent(captured_request->frontier, frontiers[1]));
  EXPECT_DOUBLE_EQ(captured_request->goal_pose.pose.position.x, 5.0);
  EXPECT_DOUBLE_EQ(captured_request->goal_pose.pose.position.y, 5.0);
}

TEST(FrontierDispatchResolutionTests, TrimsFrontierGoalToReachableCellBeforeOccupancyWall)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  auto core = make_suppression_core(&now_ns, &dispatch_calls);
  auto map_msg = build_grid(10, 5, 0);
  auto costmap_msg = build_grid(10, 5, 0);
  for (int y = 0; y < 5; ++y) {
    set_grid_cell(map_msg, 4, y, 100);
  }
  core->map = OccupancyGrid2d(map_msg);
  core->costmap = OccupancyGrid2d(costmap_msg);
  core->map_generation = 1;
  core->costmap_generation = 1;

  const auto resolved = core->resolve_dispatch_goal_point(
    make_candidate(8.5, 2.5),
    make_pose(1.5, 2.5));

  ASSERT_TRUE(resolved.has_value());
  EXPECT_DOUBLE_EQ(resolved->first, 3.5);
  EXPECT_DOUBLE_EQ(resolved->second, 2.5);
}

TEST(FrontierDispatchResolutionTests, TrimsFrontierGoalToReachableCellBeforeCostmapWall)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  auto core = make_suppression_core(&now_ns, &dispatch_calls);
  core->params.goal_skip_on_blocked_goal = true;
  auto map_msg = build_grid(10, 5, 0);
  auto costmap_msg = build_grid(10, 5, 0);
  for (int y = 0; y < 5; ++y) {
    set_grid_cell(costmap_msg, 4, y, 100);
  }
  core->map = OccupancyGrid2d(map_msg);
  core->costmap = OccupancyGrid2d(costmap_msg);
  core->map_generation = 1;
  core->costmap_generation = 1;

  const auto resolved = core->resolve_dispatch_goal_point(
    make_candidate(8.5, 2.5),
    make_pose(1.5, 2.5));

  ASSERT_TRUE(resolved.has_value());
  EXPECT_DOUBLE_EQ(resolved->first, 3.5);
  EXPECT_DOUBLE_EQ(resolved->second, 2.5);
}

TEST(FrontierDispatchResolutionTests, LetsNav2TryGoalBeyondCostmapWallWhenBlockedGoalSkipDisabled)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  auto core = make_suppression_core(&now_ns, &dispatch_calls);
  auto map_msg = build_grid(10, 5, 0);
  auto costmap_msg = build_grid(10, 5, 0);
  for (int y = 0; y < 5; ++y) {
    set_grid_cell(costmap_msg, 4, y, 100);
  }
  core->map = OccupancyGrid2d(map_msg);
  core->costmap = OccupancyGrid2d(costmap_msg);
  core->map_generation = 1;
  core->costmap_generation = 1;

  const auto resolved = core->resolve_dispatch_goal_point(
    make_candidate(8.5, 2.5),
    make_pose(1.5, 2.5));

  ASSERT_TRUE(resolved.has_value());
  EXPECT_DOUBLE_EQ(resolved->first, 8.5);
  EXPECT_DOUBLE_EQ(resolved->second, 2.5);
}

TEST(FrontierDispatchResolutionTests, TrimsOccupiedTargetWhenExplicitClearanceCheckEnabled)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  auto core = make_suppression_core(&now_ns, &dispatch_calls);
  core->params.dispatch_clearance_radius_m = 1.1;
  auto map_msg = build_grid(10, 5, 0);
  auto costmap_msg = build_grid(10, 5, 0);
  set_grid_cell(map_msg, 5, 2, 100);
  core->map = OccupancyGrid2d(map_msg);
  core->costmap = OccupancyGrid2d(costmap_msg);
  core->map_generation = 1;
  core->costmap_generation = 1;

  const auto resolved = core->resolve_dispatch_goal_point(
    make_candidate(5.5, 2.5),
    make_pose(1.5, 2.5));

  ASSERT_TRUE(resolved.has_value());
  EXPECT_NEAR(
    std::hypot(resolved->first - 5.5, resolved->second - 2.5),
    std::sqrt(2.0),
    1e-9);
  EXPECT_FALSE(resolved->first == 5.5 && resolved->second == 2.5);
}

TEST(FrontierDispatchResolutionTests, KeepsFootprintMarginFromThinFurniture)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  auto core = make_suppression_core(&now_ns, &dispatch_calls);
  core->params.dispatch_clearance_radius_m = 0.344;
  auto map_msg = build_grid(60, 20, 0, 0.05);
  auto costmap_msg = build_grid(60, 20, 0, 0.05);
  set_grid_cell(map_msg, 40, 10, 100);
  core->map = OccupancyGrid2d(map_msg);
  core->costmap = OccupancyGrid2d(costmap_msg);
  core->map_generation = 1;
  core->costmap_generation = 1;

  const auto obstacle = map_msg.info.origin.position.x + (40.5 * 0.05);
  const auto target_y = map_msg.info.origin.position.y + (10.5 * 0.05);
  const auto resolved = core->resolve_dispatch_goal_point(
    make_candidate(obstacle, target_y),
    make_pose(0.275, target_y));

  ASSERT_TRUE(resolved.has_value());
  EXPECT_GE(std::hypot(resolved->first - obstacle, resolved->second - target_y), 0.344);
}

TEST(FrontierDispatchResolutionTests, RejectsSuppressedFrontierGoal)
{
  int64_t now_ns = 0;
  int dispatch_calls = 0;
  auto core = make_suppression_core(&now_ns, &dispatch_calls);
  auto map_msg = build_grid(10, 5, 0);
  auto costmap_msg = build_grid(10, 5, 0);
  core->map = OccupancyGrid2d(map_msg);
  core->costmap = OccupancyGrid2d(costmap_msg);
  core->map_generation = 1;
  core->costmap_generation = 1;

  const auto target = make_candidate(8.5, 2.5);
  const auto current_pose = make_pose(1.5, 2.5);
  const auto first_resolved = core->resolve_dispatch_goal_point(target, current_pose);
  ASSERT_TRUE(first_resolved.has_value());
  EXPECT_DOUBLE_EQ(first_resolved->first, 8.5);
  EXPECT_DOUBLE_EQ(first_resolved->second, 2.5);

  FrontierSuppressionConfig suppression_config;
  suppression_config.base_size_m = 1.0;
  suppression_config.expansion_size_m = 0.5;
  suppression_config.timeout_s = 90.0;
  suppression_config.equivalence_tolerance = 0.3;
  core->frontier_suppression_ =
    std::make_unique<FrontierSuppression>(suppression_config);
  core->frontier_suppression_->suppress_region(
    core->frontier_with_goal_point(target, *first_resolved),
    now_ns);

  const auto second_resolved = core->resolve_dispatch_goal_point(target, current_pose);
  EXPECT_FALSE(second_resolved.has_value());
}

}  // namespace
}  // namespace frontier_exploration_ros2
