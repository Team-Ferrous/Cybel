#ifndef HARDWARE_HARDWARE_CONTROL_DAEMON_H_
#define HARDWARE_HARDWARE_CONTROL_DAEMON_H_

#include <string>
#include <unordered_set>
#include <sys/types.h>

// This class runs as a privileged daemon to handle hardware control requests.
class HardwareControlDaemon {
public:
    HardwareControlDaemon();
    ~HardwareControlDaemon();

    void run();

private:
    bool handle_request(const std::string& request);
    bool authorize_client(int client_sockfd) const;
    void load_allowlist();
    bool is_uid_allowed(uid_t uid) const;
    void configure_socket_path();
    void configure_socket_permissions() const;

    // Add private members for socket communication, e.g., socket fd.
    int sockfd_;
    std::string socket_path_;
    std::string allowlist_path_;
    std::unordered_set<uid_t> allowed_uids_;
    bool allow_all_clients_;
};

#endif // HARDWARE_HARDWARE_CONTROL_DAEMON_H_
