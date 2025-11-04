#pragma once

#include <nlohmann/json.hpp>
#include <cpp-httplib/httplib.h>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <mutex>

namespace llama {

/**
 * Configuration for RAG middleware
 */
struct RAGConfig {
    std::string aurapai_host = "localhost";
    int aurapai_port = 8001;
    int max_results = 5;
    float similarity_threshold = 0.3f;
    bool include_tools = false;
    int timeout_ms = 5000;  // 5 second timeout
    bool enabled = false;
};

/**
 * A single context chunk retrieved from RAG
 */
struct ContextChunk {
    std::string content;
    std::string source;
    float similarity;
};

/**
 * Response from RAG augmentation
 */
struct RAGResponse {
    std::string augmented_context;
    std::vector<ContextChunk> chunks;
    std::vector<std::string> suggested_tools;
    float latency_ms;
    bool success;
    std::string error_message;
};

/**
 * RAG Middleware for llama.cpp server
 * 
 * Integrates with Aurapai service to provide retrieval-augmented generation
 * capabilities. Handles HTTP communication, error handling, and graceful
 * degradation when RAG service is unavailable.
 */
class RAGMiddleware {
public:
    /**
     * Constructor
     * @param config Configuration for RAG middleware
     */
    explicit RAGMiddleware(const RAGConfig& config);
    
    /**
     * Destructor
     */
    ~RAGMiddleware();

    /**
     * Augment a user query with RAG context
     * 
     * @param query The user's query text
     * @param session_id Optional session ID for context continuity
     * @return RAGResponse containing augmented context or error
     */
    RAGResponse augment_query(const std::string& query, 
                             const std::string& session_id = "");

    /**
     * Check if RAG service is healthy and ready
     * 
     * @return true if service is operational, false otherwise
     */
    bool is_healthy();

    /**
     * Get current configuration
     * 
     * @return Reference to current config
     */
    const RAGConfig& get_config() const { return config_; }

    /**
     * Update configuration (thread-safe)
     * 
     * @param config New configuration
     */
    void update_config(const RAGConfig& config);

    /**
     * Helper function to inject RAG context into messages array
     * 
     * @param messages JSON array of chat messages
     * @param rag_context The context to inject
     * @return Modified messages array with context injected
     */
    static nlohmann::json inject_context_into_messages(
        const nlohmann::json& messages,
        const std::string& rag_context
    );

private:
    RAGConfig config_;
    std::unique_ptr<httplib::Client> http_client_;
    mutable std::mutex config_mutex_;

    /**
     * Initialize HTTP client
     */
    void init_http_client();

    /**
     * Make HTTP POST request to Aurapai
     * 
     * @param endpoint API endpoint path
     * @param body JSON body
     * @return Optional JSON response
     */
    std::optional<nlohmann::json> make_request(
        const std::string& endpoint,
        const nlohmann::json& body
    );

    /**
     * Parse RAG response from JSON
     * 
     * @param json_response JSON from Aurapai
     * @return Parsed RAGResponse
     */
    RAGResponse parse_response(const nlohmann::json& json_response);
};

/**
 * Helper function to check if a chat request should use RAG
 * 
 * @param messages JSON array of chat messages
 * @param params Request parameters JSON
 * @return true if RAG should be used
 */
bool should_use_rag(const nlohmann::json& messages, 
                   const nlohmann::json& params);

/**
 * Helper function to format RAG context for system message
 * 
 * @param chunks Vector of context chunks
 * @return Formatted string for system message
 */
std::string format_rag_context(const std::vector<ContextChunk>& chunks);

} // namespace llama
