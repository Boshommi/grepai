package embedder

import (
	"strings"
	"testing"
)

func TestParseLMStudioContextLengthError_MaxContextSize(t *testing.T) {
	texts := []string{
		"small",
		strings.Repeat("x", 1400),
	}

	err := parseLMStudioContextLengthError(
		"input (276 tokens) is larger than the max context size (256 tokens). skipping",
		texts,
	)
	if err == nil {
		t.Fatal("expected context length error")
	}
	if err.EstimatedTokens != 276 {
		t.Fatalf("EstimatedTokens = %d, want 276", err.EstimatedTokens)
	}
	if err.MaxTokens != 256 {
		t.Fatalf("MaxTokens = %d, want 256", err.MaxTokens)
	}
	if err.ChunkIndex != 1 {
		t.Fatalf("ChunkIndex = %d, want 1", err.ChunkIndex)
	}
}

func TestParseLMStudioContextLengthError_PhysicalBatchSize(t *testing.T) {
	texts := []string{
		strings.Repeat("x", 2200),
	}

	err := parseLMStudioContextLengthError(
		"input (605 tokens) is too large to process. increase the physical batch size (current batch size: 512)",
		texts,
	)
	if err == nil {
		t.Fatal("expected context length error")
	}
	if err.EstimatedTokens != 605 {
		t.Fatalf("EstimatedTokens = %d, want 605", err.EstimatedTokens)
	}
	if err.MaxTokens != 512 {
		t.Fatalf("MaxTokens = %d, want 512", err.MaxTokens)
	}
	if err.ChunkIndex != 0 {
		t.Fatalf("ChunkIndex = %d, want 0", err.ChunkIndex)
	}
}

func TestParseLMStudioContextLengthError_MaximumContextVariant(t *testing.T) {
	texts := []string{
		"small",
		strings.Repeat("x", 1500),
	}

	err := parseLMStudioContextLengthError(
		"input exceeds the maximum context length for this model",
		texts,
	)
	if err == nil {
		t.Fatal("expected context length error")
	}
	if err.ChunkIndex != 1 {
		t.Fatalf("ChunkIndex = %d, want 1", err.ChunkIndex)
	}
	if err.EstimatedTokens == 0 {
		t.Fatal("expected estimated tokens to be populated")
	}
}
