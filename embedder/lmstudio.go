package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	defaultLMStudioEndpoint    = "http://127.0.0.1:1234"
	defaultLMStudioModel       = "text-embedding-nomic-embed-text-v1.5"
	lmStudioNomicDimensions    = 768
	defaultLMStudioParallelism = 1
	defaultLMStudioTimeout     = 5 * time.Minute
	lmStudioMaxAttempts        = 3
	lmStudioRetryBaseDelay     = 750 * time.Millisecond
)

var (
	lmStudioInputTokensPattern   = regexp.MustCompile(`input\s*\((\d+)\s+tokens?\)`)
	lmStudioMaxContextPattern    = regexp.MustCompile(`max context size\s*\((\d+)\s+tokens?\)`)
	lmStudioPhysicalBatchPattern = regexp.MustCompile(`current batch size:\s*(\d+)`)
)

type LMStudioEmbedder struct {
	endpoint   string
	model      string
	dimensions int
	parallelism int
	client     *http.Client
	rateLimiter *AdaptiveRateLimiter

	gateMu   sync.Mutex
	inFlight int
	notifyCh chan struct{}
}

type lmStudioEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type lmStudioEmbedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

type lmStudioErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

type LMStudioOption func(*LMStudioEmbedder)

func WithLMStudioEndpoint(endpoint string) LMStudioOption {
	return func(e *LMStudioEmbedder) {
		e.endpoint = endpoint
	}
}

func WithLMStudioModel(model string) LMStudioOption {
	return func(e *LMStudioEmbedder) {
		e.model = model
	}
}
func WithLMStudioDimensions(dimensions int) LMStudioOption {
	return func(e *LMStudioEmbedder) {
		e.dimensions = dimensions
	}
}

func WithLMStudioParallelism(parallelism int) LMStudioOption {
	return func(e *LMStudioEmbedder) {
		if parallelism > 0 {
			e.parallelism = parallelism
		}
	}
}

func NewLMStudioEmbedder(opts ...LMStudioOption) *LMStudioEmbedder {
	e := &LMStudioEmbedder{
		endpoint:    defaultLMStudioEndpoint,
		model:       defaultLMStudioModel,
		dimensions:  lmStudioNomicDimensions,
		parallelism: defaultLMStudioParallelism,
		client: &http.Client{
			Timeout: defaultLMStudioTimeout,
		},
		notifyCh: make(chan struct{}, 1),
	}

	for _, opt := range opts {
		opt(e)
	}

	e.rateLimiter = NewAdaptiveRateLimiter(e.parallelism)

	return e
}

func (e *LMStudioEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return embeddings[0], nil
}

func (e *LMStudioEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	var lastErr error
	for attempt := 1; attempt <= lmStudioMaxAttempts; attempt++ {
		if err := e.acquirePermit(ctx); err != nil {
			return nil, err
		}
		embeddings, retryable, err := e.embedBatchOnce(ctx, texts)
		e.releasePermit()
		if err == nil {
			return embeddings, nil
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil {
			return nil, err
		}
		lastErr = err
		if !retryable || attempt == lmStudioMaxAttempts {
			return nil, err
		}
		if err := sleepWithContext(ctx, lmStudioRetryDelay(attempt)); err != nil {
			return nil, err
		}
	}

	return nil, lastErr
}

func (e *LMStudioEmbedder) Dimensions() int {
	return e.dimensions
}

func (e *LMStudioEmbedder) EmbeddingParallelism() int {
	if e.parallelism <= 0 {
		return 1
	}
	return e.parallelism
}

func (e *LMStudioEmbedder) Close() error {
	return nil
}

func (e *LMStudioEmbedder) Ping(ctx context.Context) error {
	url := fmt.Sprintf("%s/v1/models", e.endpoint)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to reach LM Studio at %s: %w", e.endpoint, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("LM Studio returned status %d", resp.StatusCode)
	}

	return nil
}

func parseLMStudioContextLengthError(msg string, texts []string) *ContextLengthError {
	normalized := strings.ToLower(msg)
	if !strings.Contains(normalized, "context length") &&
		!strings.Contains(normalized, "too many tokens") &&
		!strings.Contains(normalized, "maximum context") &&
		!strings.Contains(normalized, "max context size") &&
		!strings.Contains(normalized, "too large to process") {
		return nil
	}

	estimatedTokens := extractLMStudioTokenCount(lmStudioInputTokensPattern, normalized)
	if estimatedTokens == 0 {
		estimatedTokens = estimateLMStudioBatchTokens(texts)
	}

	maxTokens := extractLMStudioTokenCount(lmStudioMaxContextPattern, normalized)
	if maxTokens == 0 {
		maxTokens = extractLMStudioTokenCount(lmStudioPhysicalBatchPattern, normalized)
	}

	return NewContextLengthError(
		estimateLMStudioFailedInputIndex(texts, maxTokens),
		estimatedTokens,
		maxTokens,
		msg,
	)
}

func extractLMStudioTokenCount(pattern *regexp.Regexp, msg string) int {
	match := pattern.FindStringSubmatch(msg)
	if len(match) < 2 {
		return 0
	}

	value, err := strconv.Atoi(match[1])
	if err != nil {
		return 0
	}
	return value
}

func estimateLMStudioBatchTokens(texts []string) int {
	total := 0
	for _, text := range texts {
		total += EstimateTokens(text)
	}
	return total
}

func estimateLMStudioFailedInputIndex(texts []string, maxTokens int) int {
	if len(texts) == 0 {
		return 0
	}

	if maxTokens > 0 {
		for i, text := range texts {
			if EstimateTokens(text) > maxTokens {
				return i
			}
		}
	}

	longestIndex := 0
	longestTokens := EstimateTokens(texts[0])
	for i := 1; i < len(texts); i++ {
		tokens := EstimateTokens(texts[i])
		if tokens > longestTokens {
			longestTokens = tokens
			longestIndex = i
		}
	}

	return longestIndex
}

func (e *LMStudioEmbedder) acquirePermit(ctx context.Context) error {
	for {
		e.gateMu.Lock()
		limit := 1
		if e.rateLimiter != nil {
			limit = e.rateLimiter.CurrentWorkers()
		}
		if e.inFlight < limit {
			e.inFlight++
			e.gateMu.Unlock()
			return nil
		}
		notifyCh := e.notifyCh
		e.gateMu.Unlock()

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-notifyCh:
		}
	}
}

func (e *LMStudioEmbedder) releasePermit() {
	e.gateMu.Lock()
	if e.inFlight > 0 {
		e.inFlight--
	}
	e.gateMu.Unlock()
	e.signalWaiters()
}

func (e *LMStudioEmbedder) signalWaiters() {
	if e.notifyCh == nil {
		return
	}
	select {
	case e.notifyCh <- struct{}{}:
	default:
	}
}

func (e *LMStudioEmbedder) recordSuccess() {
	if e.rateLimiter != nil && e.rateLimiter.OnSuccess() {
		e.signalWaiters()
	}
}

func (e *LMStudioEmbedder) recordHTTPFailure(statusCode int) {
	if statusCode == http.StatusTooManyRequests ||
		statusCode == http.StatusBadGateway ||
		statusCode == http.StatusServiceUnavailable ||
		statusCode == http.StatusGatewayTimeout {
		e.recordTransientFailure(nil)
	}
}

func (e *LMStudioEmbedder) recordTransientFailure(err error) {
	if e.rateLimiter == nil {
		return
	}
	if err != nil && !isLMStudioTransientTransportError(err) {
		return
	}
	if e.rateLimiter.OnTransientFailure() {
		e.signalWaiters()
	}
}

func (e *LMStudioEmbedder) embedBatchOnce(ctx context.Context, texts []string) ([][]float32, bool, error) {
	reqBody := lmStudioEmbedRequest{
		Model: e.model,
		Input: texts,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, false, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/embeddings", e.endpoint)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, false, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, false, ctx.Err()
		}
		e.recordTransientFailure(err)
		return nil, true, fmt.Errorf("failed to send request to LM Studio: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		if ctx.Err() != nil {
			return nil, false, ctx.Err()
		}
		if isLMStudioTransientTransportError(err) {
			e.recordTransientFailure(err)
			return nil, true, fmt.Errorf("failed to read response: %w", err)
		}
		return nil, false, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp lmStudioErrorResponse
		msg := string(body)
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			msg = errResp.Error.Message
		}

		if ctxErr := parseLMStudioContextLengthError(msg, texts); ctxErr != nil {
			return nil, false, ctxErr
		}

		retryable := isLMStudioRetryableStatus(resp.StatusCode)
		if retryable {
			e.recordHTTPFailure(resp.StatusCode)
		}

		return nil, retryable, fmt.Errorf("LM Studio returned status %d: %s", resp.StatusCode, msg)
	}

	var result lmStudioEmbedResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, false, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Data) != len(texts) {
		return nil, false, fmt.Errorf("expected %d embeddings, got %d", len(texts), len(result.Data))
	}

	embeddings := make([][]float32, len(texts))
	for _, item := range result.Data {
		embeddings[item.Index] = item.Embedding
	}
	e.recordSuccess()

	return embeddings, false, nil
}

func isLMStudioRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusTooManyRequests ||
		statusCode == http.StatusBadGateway ||
		statusCode == http.StatusServiceUnavailable ||
		statusCode == http.StatusGatewayTimeout
}

func lmStudioRetryDelay(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	delay := lmStudioRetryBaseDelay << (attempt - 1)
	maxDelay := 5 * time.Second
	if delay > maxDelay {
		return maxDelay
	}
	return delay
}

func sleepWithContext(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return nil
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func isLMStudioTransientTransportError(err error) bool {
	if err == nil {
		return true
	}
	if errors.Is(err, io.EOF) {
		return true
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}

	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "timeout") ||
		strings.Contains(msg, "eof") ||
		strings.Contains(msg, "connection reset") ||
		strings.Contains(msg, "broken pipe")
}
