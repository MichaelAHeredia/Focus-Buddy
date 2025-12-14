# NotificationBridge (Focus Buddy)

WinUI 3 desktop app that listens to Windows notifications
and forwards them to a local Python service.

## Requirements
- Windows 10 1809+ or Windows 11
- Visual Studio 2022
- Workloads:
  - Windows App SDK / WinUI
  - .NET Desktop Development
- Python 3.9+

## One-time setup
1. Enable Developer Mode:
   Settings → Privacy & security → For developers → Developer Mode

2. Clone repository:
   git clone https://github.com/your-org/NotificationBridge.git

3. Open NotificationBridge.sln in Visual Studio

4. Restore NuGet packages

## Build & run
1. Set NotificationBridge as Startup Project
2. Target: Local Machine
3. Press F5

4. When prompted, allow notification access

## Python receiver
cd python
pip install flask
python notification_receiver.py
