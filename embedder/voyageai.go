package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync/atomic"
	"time"
)

const (
	defaultVoyageAIEndpoint = "https://api.voyageai.com/v1"
	defaultVoyageAIModel    = "voyage-code-3"
	voyageAIDimensions      = 1024
)

// VoyageAIEmbedder implements the Embedder and BatchEmbedder interfaces for the Voyage AI API.
// Voyage AI provides best-in-class code embeddings with the voyage-code-3 model.
type VoyageAIEmbedder struct {
	endpoint        string
	model           string
	apiKey          string
	dimensions      *int
	outputDimension *int // maps to Voyage's "output_dimension" JSON field
	retryPolicy     RetryPolicy
	client          *http.Client
}

type voyageAIEmbedRequest struct {
	Model           string   `json:"model"`
	Input           []string `json:"input"`
	InputType       string   `json:"input_type,omitempty"`
	OutputDimension *int     `json:"output_dimension,omitempty"`
}

type voyageAIEmbedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

type voyageAIErrorResponse struct {
	Detail string `json:"detail"`
}

type VoyageAIOption func(*VoyageAIEmbedder)

func WithVoyageAIEndpoint(endpoint string) VoyageAIOption {
	return func(e *VoyageAIEmbedder) {
		e.endpoint = endpoint
	}
}

func WithVoyageAIModel(model string) VoyageAIOption {
	return func(e *VoyageAIEmbedder) {
		e.model = model
	}
}

func WithVoyageAIKey(key string) VoyageAIOption {
	return func(e *VoyageAIEmbedder) {
		e.apiKey = key
	}
}

func WithVoyageAIDimensions(dimensions int) VoyageAIOption {
	return func(e *VoyageAIEmbedder) {
		e.dimensions = &dimensions
		e.outputDimension = &dimensions
	}
}

func NewVoyageAIEmbedder(opts ...VoyageAIOption) (*VoyageAIEmbedder, error) {
	e := &VoyageAIEmbedder{
		endpoint:    defaultVoyageAIEndpoint,
		model:       defaultVoyageAIModel,
		dimensions:  nil,
		retryPolicy: DefaultRetryPolicy(),
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}

	for _, opt := range opts {
		opt(e)
	}

	if e.apiKey == "" {
		e.apiKey = os.Getenv("VOYAGE_API_KEY")
	}

	if e.apiKey == "" {
		return nil, fmt.Errorf("voyage AI API key not set (use VOYAGE_API_KEY environment variable)")
	}

	return e, nil
}

func (e *VoyageAIEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return embeddings[0], nil
}

func (e *VoyageAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	reqBody := voyageAIEmbedRequest{
		Model:           e.model,
		Input:           texts,
		InputType:       "document",
		OutputDimension: e.outputDimension,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/embeddings", e.endpoint)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", e.apiKey))

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to Voyage AI: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, e.handleErrorResponse(resp.StatusCode, body)
	}

	var result voyageAIEmbedResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Data) != len(texts) {
		return nil, fmt.Errorf("expected %d embeddings, got %d", len(texts), len(result.Data))
	}

	// Sort by index to maintain order
	embeddings := make([][]float32, len(texts))
	for _, item := range result.Data {
		embeddings[item.Index] = item.Embedding
	}

	return embeddings, nil
}

func (e *VoyageAIEmbedder) Dimensions() int {
	if e.dimensions == nil {
		return voyageAIDimensions
	}
	return *e.dimensions
}

func (e *VoyageAIEmbedder) Close() error {
	return nil
}

// BatchLimits returns Voyage AI-specific batch constraints.
// Voyage AI has a 120,000 token limit per batch (we use 110,000 for safety margin).
// Voyage's tokenizer averages ~3 chars/token for code, vs the default ~4.
func (e *VoyageAIEmbedder) BatchLimits() BatchLimits {
	return BatchLimits{MaxSize: 128, MaxTokens: 110000, CharsPerToken: 3}
}

// EmbedBatches implements the BatchEmbedder interface.
// It processes batches sequentially with retry logic for transient errors.
func (e *VoyageAIEmbedder) EmbedBatches(ctx context.Context, batches []Batch, progress BatchProgress) ([]BatchResult, error) {
	if len(batches) == 0 {
		return nil, nil
	}

	totalChunks := 0
	for _, batch := range batches {
		totalChunks += batch.Size()
	}

	var completedChunks atomic.Int64
	results := make([]BatchResult, len(batches))

	for i := range batches {
		batch := batches[i]

		embeddings, err := e.embedBatchWithRetry(ctx, batch, len(batches), totalChunks, &completedChunks, progress)
		if err != nil {
			return nil, err
		}

		results[batch.Index] = BatchResult{
			BatchIndex: batch.Index,
			Embeddings: embeddings,
		}
	}

	return results, nil
}

func (e *VoyageAIEmbedder) embedBatchWithRetry(
	ctx context.Context,
	batch Batch,
	totalBatches int,
	totalChunks int,
	completedChunks *atomic.Int64,
	progress BatchProgress,
) ([][]float32, error) {
	contents := batch.Contents()

	for attempt := 0; ; attempt++ {
		embeddings, err := e.EmbedBatch(ctx, contents)
		if err == nil {
			newCompleted := completedChunks.Add(int64(batch.Size()))
			if progress != nil {
				progress(batch.Index, totalBatches, int(newCompleted), totalChunks, false, 0, 0)
			}
			return embeddings, nil
		}

		retryErr, isRetryable := err.(*RetryableError)
		if !isRetryable || !retryErr.Retryable {
			return nil, err
		}

		if !e.retryPolicy.ShouldRetry(attempt) {
			return nil, fmt.Errorf("batch %d failed after %d attempts: %w", batch.Index, attempt+1, err)
		}

		if progress != nil {
			progress(batch.Index, totalBatches, int(completedChunks.Load()), totalChunks, true, attempt+1, retryErr.StatusCode)
		}

		delay := e.retryPolicy.Calculate(attempt)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
		}
	}
}

// handleErrorResponse parses a Voyage AI error response.
// Voyage AI returns errors as {"detail": "..."} rather than OpenAI's {"error": {"message": "..."}}.
func (e *VoyageAIEmbedder) handleErrorResponse(statusCode int, body []byte) error {
	var errResp voyageAIErrorResponse
	msg := string(body)
	if json.Unmarshal(body, &errResp) == nil && errResp.Detail != "" {
		msg = errResp.Detail
	}

	return NewRetryableError(statusCode, fmt.Sprintf("voyage AI API error (status %d): %s", statusCode, msg))
}

// Ping checks if Voyage AI API is reachable.
func (e *VoyageAIEmbedder) Ping(ctx context.Context) error {
	url := fmt.Sprintf("%s/embeddings", e.endpoint)
	pingReq := voyageAIEmbedRequest{
		Model:     e.model,
		Input:     []string{"test"},
		InputType: "document",
	}
	jsonData, err := json.Marshal(pingReq)
	if err != nil {
		return fmt.Errorf("failed to marshal ping request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", e.apiKey))

	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to reach Voyage AI at %s: %w", e.endpoint, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("voyage AI returned status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}
