# LIBCC

_LibCC_ is a header-only library for useful C++ code.

## Building

There are 2 ways to build the executable.  One is via Visual Studio 2017, and another is via a batch file.  The
executables will be found in the folder `_bin/<platform>_<configuration>_cpplib`, where `<platform>` is currently `Win64` and `<configuration>` is either `Debug` or `Release`.

### Building via Visual Studio 2017

To build **libcc.exe** (the library tests CLI command):

* Run **gen.bat** to generate the solutions and projects inside the **_build** folder.
* Run **edit.bat** to run Visual Studio 2017 with the solution open.
* Use Visual Studio as normal to build either _Debug_ or _Release_ builds.

### Building via a batch file

To build **libcc.exe** run **build.bat**.  This will set up your CLI environment ready for Visual Studio command line tools, build the release version of **libcc.exe** using _msbuild_.

## Cleaning

All files generated by the build are placed in folders that start with an underscore.  You can run **clean.bat**, which
will delete all those folders.

## Installation

The build environment provides a **install.bat** file that will copy the release version of **libcc.exe** to the folder
determined by the environment variable **INSTALL_PATH**.  If **INSTALL_PATH** is not defined, the batch file will
warn you of this fact.  This folder should be included in your system's **PATH** variable so it can be found.

## Documentation

The development environment uses Doxygen and HTML Help Workshop to generate the HTML files and CHM file respectively.  The help build requires Doxygen and HTMLHelp.  When these tools are installed you only need to run **docs.bat** to generate the CHM and HTML files, which can be found at `_bin/docs/html` where an **index.html** file can be opened.  Additionally, you can run **help.bat** that will open the CHM file too.
