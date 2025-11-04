#include "rag_middleware.hpp"
#include <cpp-httplib/httplib.h>
#include "log.h"
#include <sstream>
#include <chrono>
#include <ctime>
#include <iomanip>

namespace llama {

// RAGMiddleware implementation

RAGMiddleware::RAGMiddleware(const RAGConfig& config)
    : config_(config) {
    if (config_.enabled) {
        init_http_client();
        LOG_INF("RAG Middleware initialized: %s:%d\n", 
                config_.aurapai_host.c_str(), config_.aurapai_port);
    } else {
        LOG_INF("RAG Middleware disabled\n");
    }
}

RAGMiddleware::~RAGMiddleware() {
    // HTTP client cleanup handled by unique_ptr
}

void RAGMiddleware::init_http_client() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    std::string host = config_.aurapai_host;
    bool use_https = false;
    int port = config_.aurapai_port;
    bool url_has_scheme = false;
    
    // Parse URL if full URL provided
    if (host.find("https://") == 0) {
        use_https = true;
        host = host.substr(8);  // Remove "https://"
        url_has_scheme = true;
        // Use standard HTTPS port for full URLs (they route internally)
        port = 443;
    } else if (host.find("http://") == 0) {
        use_https = false;
        host = host.substr(7);  // Remove "http://"
        url_has_scheme = true;
        // Use standard HTTP port for full URLs (they route internally)
        port = 80;
    }
    
    // Remove trailing slash if present
    if (!host.empty() && host.back() == '/') {
        host.pop_back();
    }
    
    // Log the connection details
    LOG_INF("RAG connecting to: %s://%s:%d\n", 
            use_https ? "https" : "http", host.c_str(), port);
    
    // Create appropriate client based on scheme
    if (use_https) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        auto ssl_client = std::make_unique<httplib::SSLClient>(host, port);
        
        // Enable SNI (Server Name Indication) for proper SSL/TLS
        ssl_client->enable_server_certificate_verification(true);
        
        // Set CA certificate path for verification
        ssl_client->set_ca_cert_path("/etc/ssl/certs");
        
        http_client_ = std::move(ssl_client);
        LOG_INF("RAG: HTTPS/SSL client initialized with certificate verification\n");
#else
        LOG_ERR("HTTPS requested but OpenSSL support not compiled in\n");
        http_client_ = std::make_unique<httplib::Client>(host, port);
#endif
    } else {
        http_client_ = std::make_unique<httplib::Client>(host, port);
    }
    
    // Set timeout
    http_client_->set_read_timeout(config_.timeout_ms / 1000, 
                                   (config_.timeout_ms % 1000) * 1000);
    http_client_->set_write_timeout(config_.timeout_ms / 1000, 
                                    (config_.timeout_ms % 1000) * 1000);
}

RAGResponse RAGMiddleware::augment_query(const std::string& query,
                                         const std::string& session_id) {
    RAGResponse response;
    response.success = false;
    
    if (!config_.enabled) {
        response.error_message = "RAG disabled";
        return response;
    }
    
    if (query.empty()) {
        response.error_message = "Empty query";
        return response;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Build request JSON
        nlohmann::json request_body = {
            {"query", query},
            {"max_results", config_.max_results},
            {"similarity_threshold", config_.similarity_threshold},
            {"include_tools", config_.include_tools}
        };
        
        if (!session_id.empty()) {
            request_body["session_id"] = session_id;
        }
        
        // Make request
        auto json_response = make_request("/api/v1/llama/augment", request_body);
        
        if (json_response.has_value()) {
            response = parse_response(json_response.value());
            response.success = true;
        } else {
            response.error_message = "Failed to get response from Aurapai";
        }
        
    } catch (const std::exception& e) {
        LOG_ERR("RAG augmentation error: %s\n", e.what());
        response.error_message = std::string("Exception: ") + e.what();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    );
    response.latency_ms = static_cast<float>(duration.count());
    
    return response;
}

bool RAGMiddleware::is_healthy() {
    if (!config_.enabled) {
        return false;
    }
    
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        
        if (!http_client_) {
            return false;
        }
        
        auto res = http_client_->Get("/api/v1/llama/health");
        
        if (res && res->status == 200) {
            try {
                auto json_response = nlohmann::json::parse(res->body);
                return json_response.value("ready", false);
            } catch (...) {
                return false;
            }
        }
        
        return false;
        
    } catch (const std::exception& e) {
        LOG_ERR("RAG health check error: %s\n", e.what());
        return false;
    }
}

void RAGMiddleware::update_config(const RAGConfig& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    bool need_reinit = (config.aurapai_host != config_.aurapai_host ||
                       config.aurapai_port != config_.aurapai_port ||
                       config.enabled != config_.enabled);
    
    config_ = config;
    
    if (need_reinit && config_.enabled) {
        init_http_client();
    }
}

std::optional<nlohmann::json> RAGMiddleware::make_request(
    const std::string& endpoint,
    const nlohmann::json& body) {
    
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    if (!http_client_) {
        LOG_ERR("HTTP client not initialized\n");
        return std::nullopt;
    }
    
    try {
        std::string json_str = body.dump();
        
        auto res = http_client_->Post(endpoint.c_str(), json_str, 
                                     "application/json");
        
        if (!res) {
            LOG_ERR("HTTP request failed: %s\n", 
                   httplib::to_string(res.error()).c_str());
            return std::nullopt;
        }
        
        if (res->status != 200) {
            LOG_ERR("HTTP request returned status %d\n", res->status);
            return std::nullopt;
        }
        
        return nlohmann::json::parse(res->body);
        
    } catch (const std::exception& e) {
        LOG_ERR("Request error: %s\n", e.what());
        return std::nullopt;
    }
}

RAGResponse RAGMiddleware::parse_response(const nlohmann::json& json_response) {
    RAGResponse response;
    response.success = true;
    
    try {
        response.augmented_context = json_response.value("augmented_context", "");
        response.latency_ms = json_response.value("latency_ms", 0.0f);
        
        // Parse chunks
        if (json_response.contains("chunks")) {
            for (const auto& chunk_json : json_response["chunks"]) {
                ContextChunk chunk;
                chunk.content = chunk_json.value("content", "");
                chunk.source = chunk_json.value("source", "unknown");
                chunk.similarity = chunk_json.value("similarity", 0.0f);
                response.chunks.push_back(chunk);
            }
        }
        
        // Parse suggested tools
        if (json_response.contains("suggested_tools")) {
            for (const auto& tool : json_response["suggested_tools"]) {
                if (tool.is_string()) {
                    response.suggested_tools.push_back(tool.get<std::string>());
                }
            }
        }
        
        LOG_INF("RAG response: %zu chunks, %.1f ms\n", 
               response.chunks.size(), response.latency_ms);
        
    } catch (const std::exception& e) {
        LOG_ERR("Error parsing RAG response: %s\n", e.what());
        response.success = false;
        response.error_message = std::string("Parse error: ") + e.what();
    }
    
    return response;
}

nlohmann::json RAGMiddleware::inject_context_into_messages(
    const nlohmann::json& messages,
    const std::string& rag_context) {
    // Prepare an injection string: use RAG context if present; otherwise, inject current date
    std::string injection = rag_context;

    if (injection.empty()) {
        // Compute current date in ISO format (YYYY-MM-DD)
        auto now = std::chrono::system_clock::now();
        std::time_t tnow = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &tnow);
#else
        localtime_r(&tnow, &tm);
#endif
        std::ostringstream oss;
        oss << "[System Note] Current date: " << std::put_time(&tm, "%Y-%m-%d");
        injection = oss.str();
    }

    if (!messages.is_array()) {
        return messages;
    }

    nlohmann::json modified_messages = messages;

    // Find the last user message and inject context before it
    for (auto it = modified_messages.rbegin(); it != modified_messages.rend(); ++it) {
        if (it->contains("role") && (*it)["role"] == "user") {
            std::string original_content = it->value("content", "");

            // Prepend injection (RAG context or date) to user message
            (*it)["content"] = injection + "\n\nUser Query: " + original_content;

            break;
        }
    }

    return modified_messages;
}

// Helper functions

bool should_use_rag(const nlohmann::json& messages, 
                   const nlohmann::json& params) {
    // Check if explicitly disabled in params
    if (params.contains("rag_enabled")) {
        return params["rag_enabled"].get<bool>();
    }
    
    // Check if this is a system-only message (don't use RAG)
    if (!messages.is_array() || messages.empty()) {
        return false;
    }
    
    // Check if there's a user message
    for (const auto& msg : messages) {
        if (msg.contains("role") && msg["role"] == "user") {
            return true;  // Use RAG for user queries
        }
    }
    
    return false;
}

std::string format_rag_context(const std::vector<ContextChunk>& chunks) {
    if (chunks.empty()) {
        return "";
    }
    
    std::ostringstream oss;
    oss << "[Retrieved Context]\n";
    
    for (size_t i = 0; i < chunks.size(); ++i) {
        oss << "\n[Source " << (i + 1) << ": " << chunks[i].source 
            << " (relevance: " << chunks[i].similarity << ")]\n";
        oss << chunks[i].content << "\n";
    }
    
    oss << "\n[End Retrieved Context]\n";
    
    return oss.str();
}

} // namespace llama
