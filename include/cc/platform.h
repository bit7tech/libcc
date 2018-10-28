//----------------------------------------------------------------------------------------------------------------------
//! @file       cc/platform.h
//! @brief      Platform detection.
//! @author     Matt Davies
//! @copyright  Copyright (C)2018 Bit-7 Technology.
//----------------------------------------------------------------------------------------------------------------------

#pragma once

#define YES (1)
#define NO (0)

// Compiler defines
#define CC_COMPILER_MSVC     NO

// OS defines
#define CC_OS_WIN32          NO

// CPU defines
#define CC_CPU_X86           NO
#define CC_CPU_X64           NO

// Configuration defines
#define CC_DEBUG             NO
#define CC_RELEASE           NO

// Endianess
#define CC_LITTLE_ENDIAN     YES
#define CC_BIG_ENDIAN        NO

//----------------------------------------------------------------------------------------------------------------------
// Compiler determination

#ifdef _MSC_VER
#   undef CC_COMPILER_MSVC
#   define CC_COMPILER_MSVC YES
#else
#   error Unknown compiler.  Please define COMPILE_XXX macro for your compiler.
#endif

//----------------------------------------------------------------------------------------------------------------------
// OS determination

#ifdef _WIN32
#   undef CC_OS_WIN32
#   define CC_OS_WIN32 YES
#else
#   error Unknown OS.  Please define OS_XXX macro for your operating system.
#endif

//----------------------------------------------------------------------------------------------------------------------
// CPU determination

#if CC_COMPILER_MSVC
#   if defined(_M_X64)
#       undef CC_CPU_X64
#       define CC_CPU_X64 YES
#   elif defined(_M_IX86)
#       undef CC_CPU_X86
#       define CC_CPU_X86 YES
#   else
#       error Can not determine processor - something's gone very wrong here!
#   endif
#else
#   error Add CPU determination code for your compiler.
#endif

//----------------------------------------------------------------------------------------------------------------------
// Configuration

#ifdef _DEBUG
#   undef CC_DEBUG
#   define CC_DEBUG YES
#else
#   undef CC_RELEASE
#   define CC_RELEASE YES
#endif

//----------------------------------------------------------------------------------------------------------------------
// Standard headers

#if CC_OS_WIN32
#   define WIN32_LEAN_AND_MEAN
#   define NOMINMAX
#   include <Windows.h>
#   ifdef _DEBUG
#       include <crtdbg.h>
#   endif
#endif

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------
// Debugging
//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

#if CC_OS_WIN32 && CC_DEBUG
#   define CC_BREAK() ::DebugBreak()
#else
#   define CC_BREAK()
#endif

#define CC_ASSERT(x, ...) assert(x)

//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------
