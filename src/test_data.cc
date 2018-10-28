//----------------------------------------------------------------------------------------------------------------------
//! @file       test_data.cc
//! @brief      Tests the Data class.
//! @author     Matt Davies
//! @copyright  Copyright (C)2018 Matt Davies, all rights reserved.
//----------------------------------------------------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <cc/data.h>

using namespace cc;

TEST(Data, Empty)
{
    Data d;

    EXPECT_EQ((void *)d, nullptr);
    EXPECT_EQ((u8 *)d, nullptr);
    EXPECT_EQ((const void *)d, nullptr);
    EXPECT_EQ((const u8 *)d, nullptr);
    EXPECT_EQ(d.size(), 0);
}

TEST(Data, CreateAndRead)
{
    // Create a file.
    {
        Data out("test.out", 64);

        for (u8& b : out)
        {
            b = 42;
        }
    }

    // Open it again
    {
        Data in("test.out");
        EXPECT_EQ(in.size(), 64);
        for (u8& b : in)
        {
            EXPECT_EQ(b, 42);
        }

        for (int i = 0; i < in.size(); ++i)
        {
            EXPECT_EQ(in[i], 42);
        }
    }
}

