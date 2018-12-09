@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
if "%SPHINXINTL%" == "" (
	set SPHINXINTL=sphinx-intl
)
set SOURCEDIR=source
set BUILDDIR=build
set SPHINXPROJ=mars
set I18NSPHINXOPTS=%SPHINXOPTS% %SOURCEDIR%
set I18NSPHINXLANGS=-l zh_CN

if "%1" == "" goto help
if "%1" == "gettext" goto gettext

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:gettext
%SPHINXBUILD% -b gettext %I18NSPHINXOPTS% %BUILDDIR%/locale
if errorlevel 1 exit /b 1
%SPHINXINTL% update -p %BUILDDIR%/locale %I18NSPHINXLANGS%
if errorlevel 1 exit /b 1
echo.
echo.Build finished. The message catalogs are in %BUILDDIR%/locale.
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
