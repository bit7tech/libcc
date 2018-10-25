//----------------------------------------------------------------------------------------------------------------------
//! @file       test_main.cc
//! @brief      Main Tests
//!
//! @author     Matt Davies
//----------------------------------------------------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cc/raptor.h>
#include <chrono>
#include <thread>
#include <vector>

using namespace std;
using namespace cc;

constexpr int kNumberTasks = 100;

//----------------------------------------------------------------------------------------------------------------------

int computeMe()
{
    this_thread::sleep_for(chrono::milliseconds(10));
    return 123;
}

void sleep()
{
    this_thread::sleep_for(chrono::milliseconds(5));
}

//----------------------------------------------------------------------------------------------------------------------

TEST(Scheduler, Wait)
{
    Scheduler scheduler;
    auto queue = scheduler.createQueue(QueueType::Concurrent);

    vector<Task<int>> taskVector;
    for (int i = 0; i < kNumberTasks; ++i)
    {
        Task<int> t = queue->async(&computeMe);
        taskVector.push_back(t);
    }

    Task<vector<int>> waitAll = scheduler.when_all(taskVector.begin(), taskVector.end());
    waitAll.wait();

    for (int i = 0; i < kNumberTasks; ++i)
    {
        EXPECT_TRUE(taskVector[i].hasFinished());
        int v = taskVector[i].get();
        EXPECT_EQ(v, 123);
    }
}

//----------------------------------------------------------------------------------------------------------------------

TEST(Scheduler, VoidTasks)
{
    Scheduler scheduler;
    auto queue = scheduler.createQueue(QueueType::Concurrent);

    vector<Task<void>> taskVector;
    for (int i = 0; i < kNumberTasks; ++i)
    {
        auto t = queue->async(&sleep);
        taskVector.push_back(t);
    }

    Task<void> waitAll = scheduler.when_all(taskVector.begin(), taskVector.end());
    waitAll.wait();

    for (int i = 0; i < kNumberTasks; ++i)
    {
        EXPECT_TRUE(taskVector[i].hasFinished());
    }
}

//----------------------------------------------------------------------------------------------------------------------

TEST(Scheduler, Serial)
{
    Scheduler scheduler;
    auto queue = scheduler.createQueue(QueueType::Serial);

    vector<Task<int>> taskVector;
    taskVector.push_back(queue->async(&computeMe));
    
    for (int i = 1; i < kNumberTasks; ++i)
    {
        auto f = [](Task<int> prevResult) -> int {
            EXPECT_TRUE(prevResult.hasFinished());
            EXPECT_EQ(prevResult.get(), 123);
            return computeMe();
        };
        auto task = std::bind(f, taskVector[i - 1]);
        std::function<int()> ff = task;
        auto t = queue->async(ff);
        taskVector.push_back(t);
    }

    Task<vector<int>> waitAll = scheduler.when_all(taskVector.begin(), taskVector.end());
    waitAll.wait();

    for (int i = 0; i < kNumberTasks; ++i)
    {
        EXPECT_TRUE(taskVector[i].hasFinished());
        int v = taskVector[i].get();
        EXPECT_EQ(v, 123);
    }
}

//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------
