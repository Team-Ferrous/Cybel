#include "hardware_control_daemon.h"
#include <iostream>

int main() {
    try {
        HardwareControlDaemon daemon;
        daemon.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error in hardware control daemon: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}