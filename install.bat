@echo off
if [%INSTALL_PATH%] NEQ [] (
    copy /y _bin\Win64_Release_libcc\libcc.exe %INSTALL_PATH%
) else (
    echo Please define INSTALL_PATH in the environment
)

