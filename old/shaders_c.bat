@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "GLSLC=C:\VulkanSDK\1.4.341.1\Bin\glslc.exe"
set "SHADER_DIR=shaders"
set "ERRORS=0"

if not exist "%GLSLC%" (
    echo [ERROR] glslc not found: %GLSLC%
    exit /b 1
)

if not exist "%SHADER_DIR%" (
    echo [ERROR] shader directory not found: %SHADER_DIR%
    exit /b 1
)

echo Compiling shaders...

for %%F in ("%SHADER_DIR%\*.vert" "%SHADER_DIR%\*.frag") do (
    if exist "%%~fF" (
        echo [GLSLC] %%~nxF -^> %%~nxF.spv

        "%GLSLC%" "%%~fF" ^
            -I"%SHADER_DIR%" ^
            -O ^
            -o "%%~fF.spv"

        if errorlevel 1 (
            echo [FAILED] %%~fF
            set /a ERRORS+=1
        )
    )
)

if not "%ERRORS%"=="0" (
    echo.
    echo [ERROR] %ERRORS% shader compile failed.
    exit /b 1
)

echo.
echo [OK] all shaders compiled.
exit /b 0