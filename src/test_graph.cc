//----------------------------------------------------------------------------------------------------------------------
//! @file       test_graph.cc
//! @brief      Main Tests
//!
//! @author     Matt Davies
//----------------------------------------------------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cc/raptor.h>

//----------------------------------------------------------------------------------------------------------------------

class Sum : public cc::Node
{
public:
    Sum()
    {
        input.addPort<int>("input_a", "input_b");
        output.addPort<int>("sum");
    }

    RunState run() override
    {
        int a, b;
        input["input_a"].pop(a);
        input["input_b"].pop(b);
        output.push(a + b);

        return RunState::Complete;
    }
};

//----------------------------------------------------------------------------------------------------------------------

class Gen : public cc::Node
{
    int m_count;

public:
    Gen(int n)
        : m_count(n)
    {
        output.addPort<int>("output");
    }

    RunState run() override
    {
        for (int i = 0; i < m_count; ++i)
        {
            output.push(i);
        }

        return RunState::Complete;
    }
};

//----------------------------------------------------------------------------------------------------------------------

class Print : public cc::Node
{
    int m_expected;

public:
    Print(int p)
        : m_expected(0)
        , m_p(p)
    {
        input.addPort<int>("input");
    }

    RunState run() override
    {
        int t;
        input.pop(t);
        EXPECT_EQ(t, m_expected);
        m_expected += m_p;
        printf("%d\n", t);

        return RunState::Complete;
    }

    int m_p;
};

//----------------------------------------------------------------------------------------------------------------------

TEST(Graph, GenKernel)
{
    Gen g1(10), g2(10);
    Sum s;
    Print p(2);
    Print p2(1);
    cc::Scheduler scheduler;

    cc::Graph G;

    G += g1 >> s["input_a"];
    G += g1 >> p2;
    G += g2 >> s["input_b"];
    G += s >> p;

    G.exe(scheduler);
}

