# 创建/修复桌面快捷方式，并把图标绑定到 assets/newworld.ico
#
# 用法（在 newworldpro 目录或任意位置执行）：
#   powershell -ExecutionPolicy Bypass -File tools/update_shortcut.ps1
#
# 说明：
# - 目标固定为 start_world.bat --debug，工作目录为项目根目录；
# - 图标来自 tools/make_icon.py 从 ref/blueprint.pdf 生成的多尺寸 .ico；
# - 如果图标不存在，本脚本会提示先运行 python tools/make_icon.py；
# - 写入桌面属于工作区之外的操作，可能需要在允许的权限下运行。
param(
    [string]$ShortcutPath = (Join-Path ([Environment]::GetFolderPath("Desktop")) "newworld 像素世界.lnk"),
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [switch]$NoIcon
)

$ErrorActionPreference = "Stop"

$target = Join-Path $ProjectRoot "start_world.bat"
if (-not (Test-Path $target)) {
    throw "找不到启动脚本：$target（请在项目目录内运行）"
}

$icon = Join-Path $ProjectRoot "assets\newworld.ico"
$useIcon = (-not $NoIcon) -and (Test-Path $icon)
if (-not $NoIcon -and -not (Test-Path $icon)) {
    Write-Host "提示：未找到 $icon，将使用系统默认图标。"
    Write-Host "      先运行：python tools/make_icon.py"
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($ShortcutPath)
$shortcut.TargetPath = $target
$shortcut.Arguments = "--debug"
$shortcut.WorkingDirectory = $ProjectRoot
$shortcut.Description = "打开 newworld 像素文明世界（鼠标点击观察个体；底部按钮控制仿真）"
if ($useIcon) {
    $shortcut.IconLocation = "$icon,0"
}
$shortcut.Save()

$check = $shell.CreateShortcut($ShortcutPath)
Write-Host "快捷方式已更新：$ShortcutPath"
Write-Host "  目标：$($check.TargetPath) $($check.Arguments)"
Write-Host "  目录：$($check.WorkingDirectory)"
Write-Host "  图标：$($check.IconLocation)"
Write-Host "  说明：$($check.Description)"
