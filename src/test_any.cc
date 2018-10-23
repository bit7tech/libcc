//----------------------------------------------------------------------------------------------------------------------
//! @file       test_any.cc
//! @brief      Tests the any class.
//! @author     Matt Davies
//! @copyright  Copyright (C)2018 Matt Davies, all rights reserved.
//----------------------------------------------------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <cc/any.h>

TEST(Any, Integer)
{
    cc::any a = 42;
    EXPECT_EQ(a.cast<int>(), 42);
}



