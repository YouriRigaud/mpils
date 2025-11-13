/**
 * @file logger.h
 * @brief Lightweight logging utility with verbosity control and timestamps.
 *
 * The Logger class allows printing information and debug messages with
 * adjustable verbosity levels. Messages include timestamps and can be
 * redirected to any std::ostream (e.g., console, file, or buffer).
 *
 * Example usage:
 * @code
 * Logger log(Verbosity::Debug, std::cout);
 * log.info("Tuning started");
 * log.debug("Current parameter set:", params.id());
 * @endcode
 *
 * @author Youri Rigaud
 * @copyright Copyright 2025 Youri Rigaud. All rights reserved.
 *            This software is licensed under the GNU General Public License v3.0.
 *            See the accompanying LICENSE file for full details.
 */

#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>

/**
 * @brief Verbosity levels for the Logger.
 */
enum class Verbosity {
    Normal,
    Debug,
};

/**
 * @brief Simple Logger class with verbosity control and timestamps.
 * 
 * This class supports formatted output to any std::ostream,
 * with optional debug messages controlled by a verbosity level.
 */
class Logger {
    private:
        Verbosity level_;   ///< Current verbosity level.
        std::ostream* out_; ///< Target stream for logs.

        /**
         * @brief Internal method to print a message with a given prefix.
         * @tparam Args Variadic template parameters for message components.
         * @param prefix Prefix string (e.g., "[INFO]:", "[DEBUG]:").
         * @param args   Message components to be printed.
         */
        template<typename... Args>
        void print(const std::string& prefix, Args&&... args) const {
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            (*out_) << prefix;
            (*out_) << std::put_time(std::localtime(&now), "[%H:%M:%S] ");
            ((*out_) << ... << args) << std::endl;
        }

    public:
        /**
         * @brief Construct a Logger object.
         * @param level Initial verbosity level.
         * @param out   Output stream (default: std::cout).
         */
        Logger(Verbosity level, std::ostream& out): level_(level), out_(&out) {}

        /** @brief Change the verbosity level. */
        void setLevel(Verbosity lvl) { level_ = lvl; }

        /** @brief Change the output stream. */
        void setOutput(std::ostream& output) { out_ = &output; }

        /**
         * @brief Print an info-level message.
         * @tparam Args Variadic template parameters for message components.
         */
        template<typename... Args>
        void info(Args&&... args) const {
            if (level_ >= Verbosity::Normal)
                print("[INFO]:", std::forward<Args>(args)...);
        }

        /**
         * @brief Print a debug-level message.
         * @tparam Args Variadic template parameters for message components.
         */
        template<typename... Args>
        void debug(Args&&... args) const {
            if (level_ == Verbosity::Debug)
                print("[DEBUG]:", std::forward<Args>(args)...);
        }
};

#endif // LOGGER_H