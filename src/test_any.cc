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
    EXPECT_THROW(a.cast<char>(), cc::impl::BadAnyCast);
}

TEST(Any, StringLiterals)
{
    cc::any a = 42;
    a = "hello";
    EXPECT_EQ(*a.cast<const char *>(), 'h');
    size_t l = strlen(a.cast<const char*>());
    EXPECT_EQ(l, 5);
    int eq = strcmp(a.cast<const char*>(), "hello");
    EXPECT_EQ(eq, 0);
}

TEST(Any, StdString)
{
    cc::any a = std::string("1234567890");
    EXPECT_EQ(a.cast<std::string>(), "1234567890");
    EXPECT_EQ(a.cast<std::string>().size(), 10);
}

TEST(Any, Pointers)
{
    int n = 42;
    cc::any a = &n;
    EXPECT_EQ(*a.cast<int *>(), 42);
}

