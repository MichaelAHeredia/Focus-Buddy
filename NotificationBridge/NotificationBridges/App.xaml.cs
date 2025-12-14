using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Navigation;
using Microsoft.UI.Xaml.Shapes;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Windows.ApplicationModel;
using Windows.ApplicationModel.Activation;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Notifications;
using Windows.UI.Notifications.Management;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace NotificationBridges
{
    /// <summary>
    /// Provides application-specific behavior to supplement the default Application class.
    /// </summary>
    public partial class App : Application
    {
        private Window? _window;

        private UserNotificationListener _listener;
        private static readonly HttpClient _httpClient = new HttpClient();

        private const string PythonEndpoint = "http://localhost:5005/notification";
        /// <summary>
        /// Initializes the singleton application object.  This is the first line of authored code
        /// executed, and as such is the logical equivalent of main() or WinMain().
        /// </summary>
        public App()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Invoked when the application is launched.
        /// </summary>
        /// <param name="args">Details about the launch request and process.</param>
        protected override void OnLaunched(Microsoft.UI.Xaml.LaunchActivatedEventArgs args)
        {
            _ = new MainWindow();
            _ = InitializeNotificationListenerAsync();
        }

        private async Task InitializeNotificationListenerAsync()
        {
            try
            {
                _listener = UserNotificationListener.Current;

                var access = await _listener.RequestAccessAsync();
                System.Diagnostics.Debug.WriteLine($"Notification access status: {access}");
                if (access != UserNotificationListenerAccessStatus.Allowed)
                {
                    System.Diagnostics.Debug.WriteLine("Notification access denied.");
                    return;
                }

                _listener.NotificationChanged += Listener_NotificationChanged;
                System.Diagnostics.Debug.WriteLine("Notification listener initialized.");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"InitializeNotificationListenerAsync exception: {ex}");
            }
        }

        private async void Listener_NotificationChanged(
            UserNotificationListener sender,
            UserNotificationChangedEventArgs args)
        {
            try
            {
                var userNotification = sender.GetNotification(args.UserNotificationId);
                if (userNotification == null)
                    return;

                var appName =
                    userNotification.AppInfo?.DisplayInfo?.DisplayName ?? "UnknownApp";

                string[] textParts = Array.Empty<string>();

                try
                {
                    var binding =
                        userNotification.Notification?.Visual?
                        .GetBinding(KnownNotificationBindings.ToastGeneric);

                    if (binding != null)
                    {
                        textParts = binding.GetTextElements()
                                           .Select(t => t.Text)
                                           .ToArray();
                    }
                }
                catch
                {
                    // Ignore malformed notifications
                }

                var payload = new
                {
                    App = appName,
                    NotificationId = args.UserNotificationId,
                    ChangeType = args.ChangeKind.ToString(),
                    Text = string.Join(" ", textParts),
                    Timestamp = DateTimeOffset.Now.ToString("o")
                };

                var json = JsonSerializer.Serialize(payload);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                await _httpClient.PostAsync(PythonEndpoint, content);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Listener error: {ex}");
            }
        }
    }
}

