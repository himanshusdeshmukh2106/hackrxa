-- Database initialization script for LLM Query Retrieval System
-- This script sets up the initial database schema and configuration

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT NOT NULL UNIQUE,
    content_type VARCHAR(100),
    file_size BIGINT,
    status VARCHAR(50) DEFAULT 'pending',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Create text_chunks table
CREATE TABLE IF NOT EXISTS text_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding_vector FLOAT8[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- Create query_logs table
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    response TEXT,
    document_url TEXT,
    processing_time_ms INTEGER,
    confidence_score FLOAT,
    source_chunks TEXT[],
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Create system_config table
CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);

CREATE INDEX IF NOT EXISTS idx_text_chunks_document_id ON text_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_text_chunks_chunk_index ON text_chunks(chunk_index);
CREATE INDEX IF NOT EXISTS idx_text_chunks_content_gin ON text_chunks USING gin(to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_query_logs_document_url ON query_logs(document_url);
CREATE INDEX IF NOT EXISTS idx_query_logs_user_id ON query_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_session_id ON query_logs(session_id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_config_updated_at 
    BEFORE UPDATE ON system_config 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default system configuration
INSERT INTO system_config (key, value, description) VALUES
    ('embedding_model', '"sentence-transformers/all-MiniLM-L6-v2"', 'Default embedding model name'),
    ('max_chunk_size', '1000', 'Maximum size of text chunks in characters'),
    ('chunk_overlap', '200', 'Overlap between consecutive chunks in characters'),
    ('similarity_threshold', '0.7', 'Minimum similarity threshold for search results'),
    ('max_search_results', '10', 'Maximum number of search results to return'),
    ('max_context_chunks', '5', 'Maximum number of chunks to include in LLM context'),
    ('min_confidence_threshold', '0.6', 'Minimum confidence threshold for answers'),
    ('rate_limit_requests', '100', 'Number of requests allowed per time window'),
    ('rate_limit_window_seconds', '3600', 'Rate limiting time window in seconds'),
    ('request_timeout_seconds', '30', 'Request timeout in seconds'),
    ('retry_attempts', '3', 'Number of retry attempts for failed operations'),
    ('cache_ttl_seconds', '3600', 'Default cache TTL in seconds'),
    ('log_retention_days', '30', 'Number of days to retain query logs')
ON CONFLICT (key) DO NOTHING;

-- Create view for document statistics
CREATE OR REPLACE VIEW document_stats AS
SELECT 
    status,
    COUNT(*) as count,
    AVG(file_size) as avg_file_size,
    SUM(file_size) as total_file_size,
    MIN(created_at) as first_created,
    MAX(created_at) as last_created
FROM documents 
GROUP BY status;

-- Create view for query statistics
CREATE OR REPLACE VIEW query_stats AS
SELECT 
    DATE(created_at) as query_date,
    COUNT(*) as total_queries,
    AVG(processing_time_ms) as avg_processing_time,
    AVG(confidence_score) as avg_confidence,
    COUNT(CASE WHEN confidence_score >= 0.8 THEN 1 END) as high_confidence_queries,
    COUNT(CASE WHEN processing_time_ms > 10000 THEN 1 END) as slow_queries
FROM query_logs 
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY query_date DESC;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO llm_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO llm_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO llm_user;

-- Create cleanup function for old logs
CREATE OR REPLACE FUNCTION cleanup_old_logs()
RETURNS INTEGER AS $$
DECLARE
    retention_days INTEGER;
    deleted_count INTEGER;
BEGIN
    -- Get retention period from config
    SELECT (value::TEXT)::INTEGER INTO retention_days 
    FROM system_config 
    WHERE key = 'log_retention_days';
    
    -- Default to 30 days if not configured
    IF retention_days IS NULL THEN
        retention_days := 30;
    END IF;
    
    -- Delete old logs
    DELETE FROM query_logs 
    WHERE created_at < CURRENT_TIMESTAMP - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get system health
CREATE OR REPLACE FUNCTION get_system_health()
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'database_size', pg_size_pretty(pg_database_size(current_database())),
        'total_documents', (SELECT COUNT(*) FROM documents),
        'processed_documents', (SELECT COUNT(*) FROM documents WHERE status = 'completed'),
        'total_chunks', (SELECT COUNT(*) FROM text_chunks),
        'total_queries', (SELECT COUNT(*) FROM query_logs),
        'queries_last_24h', (SELECT COUNT(*) FROM query_logs WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'),
        'avg_processing_time_24h', (SELECT COALESCE(AVG(processing_time_ms), 0) FROM query_logs WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'),
        'timestamp', CURRENT_TIMESTAMP
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Create indexes for better query performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_query_logs_created_at_desc ON query_logs(created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_status_created_at ON documents(status, created_at);

-- Analyze tables for better query planning
ANALYZE documents;
ANALYZE text_chunks;
ANALYZE query_logs;
ANALYZE system_config;

-- Log successful initialization
INSERT INTO query_logs (query, response, processing_time_ms, metadata) VALUES
    ('DATABASE_INIT', 'Database initialization completed successfully', 0, '{"event": "database_init", "version": "1.0.0"}');