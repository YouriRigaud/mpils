#ifndef FILESYSTEM_UTILS_H
#define FILESYSTEM_UTILS_H

#include <filesystem>
#include <string>

inline void ensureParentDirectoryForFile(const std::string& file_path) {
    const std::filesystem::path path(file_path);
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

inline void ensureDirectoryExists(const std::string& directory_path) {
    if (!directory_path.empty()) {
        std::filesystem::create_directories(directory_path);
    }
}

#endif // FILESYSTEM_UTILS_H
