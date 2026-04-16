-- Schema mínimo para tests de integración.
-- Ejecutar antes de los tests si la BD está vacía.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TABLE IF NOT EXISTS legal_documents (
    id UUID PRIMARY KEY,
    "blobContainer" VARCHAR(255),
    "blobPath" VARCHAR(1000) UNIQUE NOT NULL,
    title VARCHAR(500),
    category_id UUID,
    "publishDate" DATE,
    "contentHash" VARCHAR(64) UNIQUE NOT NULL,
    "sourceType" VARCHAR(100),
    "sourceUri" VARCHAR(1000),
    jurisdiction VARCHAR(100),
    "docType" VARCHAR(100),
    "lawName" VARCHAR(255),
    metadata JSONB,
    "createdAt" TIMESTAMP DEFAULT NOW(),
    "updatedAt" TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS legal_chunks (
    id UUID PRIMARY KEY,
    "documentId" UUID NOT NULL REFERENCES legal_documents(id) ON DELETE CASCADE,
    "chunkNo" INT NOT NULL,
    "chunkType" VARCHAR(50),
    "articleRef" VARCHAR(100),
    heading VARCHAR(500),
    text TEXT NOT NULL,
    "tokenCount" INT DEFAULT 0,
    "startPage" INT,
    "endPage" INT,
    embedding vector(1536),
    tsv tsvector
);

CREATE TABLE IF NOT EXISTS index_runs (
    id UUID PRIMARY KEY,
    "startedAt" TIMESTAMP DEFAULT NOW(),
    "endedAt" TIMESTAMP,
    status VARCHAR(50) NOT NULL,
    "docsTotal" INT DEFAULT 0,
    "docsIndexed" INT DEFAULT 0,
    "chunksTotal" INT DEFAULT 0,
    error TEXT
);
