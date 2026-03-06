package embedder

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type roundTripFunc func(req *http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func newLMStudioSuccessResponse(req *http.Request) (*http.Response, error) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	_ = req.Body.Close()

	var embedReq lmStudioEmbedRequest
	if err := json.Unmarshal(body, &embedReq); err != nil {
		return nil, err
	}

	items := make([]string, len(embedReq.Input))
	for i := range embedReq.Input {
		items[i] = fmt.Sprintf(`{"embedding":[0.1,0.2,0.3],"index":%d}`, i)
	}

	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(`{"data":[` + strings.Join(items, ",") + `]}`)),
		Header:     make(http.Header),
	}, nil
}

type scriptedLMStudioTransport struct {
	mu    sync.Mutex
	steps []scriptedLMStudioStep
}

type scriptedLMStudioStep struct {
	status int
	body   string
	err    error
}

type timeoutError struct{}

func (timeoutError) Error() string   { return "timeout" }
func (timeoutError) Timeout() bool   { return true }
func (timeoutError) Temporary() bool { return true }

func (s *scriptedLMStudioTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.steps) == 0 {
		return newLMStudioSuccessResponse(req)
	}
	step := s.steps[0]
	s.steps = s.steps[1:]

	if step.err != nil {
		return nil, step.err
	}
	if step.status == 0 {
		step.status = http.StatusOK
	}
	if step.status == http.StatusOK && step.body == "" {
		return newLMStudioSuccessResponse(req)
	}

	return &http.Response{
		StatusCode: step.status,
		Body:       io.NopCloser(strings.NewReader(step.body)),
		Header:     make(http.Header),
	}, nil
}

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

func TestLMStudioEmbedder_EmbeddingParallelism(t *testing.T) {
	emb := NewLMStudioEmbedder(WithLMStudioParallelism(7))
	if emb.EmbeddingParallelism() != 7 {
		t.Fatalf("EmbeddingParallelism = %d, want 7", emb.EmbeddingParallelism())
	}
}

func TestLMStudioEmbedder_RespectsConcurrencyCeiling(t *testing.T) {
	var (
		active    atomic.Int32
		maxActive atomic.Int32
		releaseCh = make(chan struct{})
	)

	emb := NewLMStudioEmbedder(WithLMStudioParallelism(2))
	emb.client = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			current := active.Add(1)
			for {
				max := maxActive.Load()
				if current <= max || maxActive.CompareAndSwap(max, current) {
					break
				}
			}
			defer active.Add(-1)

			<-releaseCh
			return newLMStudioSuccessResponse(req)
		}),
	}

	ctx := context.Background()
	errCh := make(chan error, 4)
	for i := 0; i < 4; i++ {
		go func() {
			_, err := emb.EmbedBatch(ctx, []string{"hello"})
			errCh <- err
		}()
	}

	time.Sleep(100 * time.Millisecond)
	if maxActive.Load() > 2 {
		t.Fatalf("max active requests = %d, want <= 2", maxActive.Load())
	}

	close(releaseCh)
	for i := 0; i < 4; i++ {
		if err := <-errCh; err != nil {
			t.Fatalf("EmbedBatch failed: %v", err)
		}
	}
}

func TestLMStudioEmbedder_ReducesAndRestoresParallelismOnTransientOverload(t *testing.T) {
	transport := &scriptedLMStudioTransport{
		steps: []scriptedLMStudioStep{
			{status: http.StatusServiceUnavailable, body: `{"error":{"message":"busy"}}`},
			{status: http.StatusServiceUnavailable, body: `{"error":{"message":"busy"}}`},
			{status: http.StatusServiceUnavailable, body: `{"error":{"message":"busy"}}`},
		},
	}

	emb := NewLMStudioEmbedder(WithLMStudioParallelism(4))
	emb.client = &http.Client{Transport: transport}

	for i := 0; i < 3; i++ {
		if _, err := emb.EmbedBatch(context.Background(), []string{"hello"}); err == nil {
			t.Fatal("expected transient overload error")
		}
	}
	if emb.rateLimiter.CurrentWorkers() != 2 {
		t.Fatalf("CurrentWorkers after overload = %d, want 2", emb.rateLimiter.CurrentWorkers())
	}

	for i := 0; i < 10; i++ {
		if _, err := emb.EmbedBatch(context.Background(), []string{"hello"}); err != nil {
			t.Fatalf("EmbedBatch success #%d failed: %v", i+1, err)
		}
	}
	if emb.rateLimiter.CurrentWorkers() != 3 {
		t.Fatalf("CurrentWorkers after restoration = %d, want 3", emb.rateLimiter.CurrentWorkers())
	}
}

func TestLMStudioEmbedder_TransportEOFReducesParallelism(t *testing.T) {
	transport := &scriptedLMStudioTransport{
		steps: []scriptedLMStudioStep{
			{err: io.EOF},
			{err: io.EOF},
			{err: io.EOF},
		},
	}

	emb := NewLMStudioEmbedder(WithLMStudioParallelism(4))
	emb.client = &http.Client{Transport: transport}

	for i := 0; i < 3; i++ {
		if _, err := emb.EmbedBatch(context.Background(), []string{"hello"}); err == nil {
			t.Fatal("expected transport error")
		}
	}
	if emb.rateLimiter.CurrentWorkers() != 2 {
		t.Fatalf("CurrentWorkers after EOF failures = %d, want 2", emb.rateLimiter.CurrentWorkers())
	}
}

func TestLMStudioEmbedder_ContextLengthDoesNotReduceParallelism(t *testing.T) {
	transport := &scriptedLMStudioTransport{
		steps: []scriptedLMStudioStep{
			{status: http.StatusBadRequest, body: `{"error":{"message":"input (276 tokens) is larger than the max context size (256 tokens). skipping"}}`},
			{status: http.StatusBadRequest, body: `{"error":{"message":"input (276 tokens) is larger than the max context size (256 tokens). skipping"}}`},
			{status: http.StatusBadRequest, body: `{"error":{"message":"input (276 tokens) is larger than the max context size (256 tokens). skipping"}}`},
		},
	}

	emb := NewLMStudioEmbedder(WithLMStudioParallelism(4))
	emb.client = &http.Client{Transport: transport}

	for i := 0; i < 3; i++ {
		_, err := emb.EmbedBatch(context.Background(), []string{strings.Repeat("x", 1400)})
		if err == nil {
			t.Fatal("expected context length error")
		}
		if AsContextLengthError(err) == nil {
			t.Fatalf("expected ContextLengthError, got %v", err)
		}
	}
	if emb.rateLimiter.CurrentWorkers() != 4 {
		t.Fatalf("CurrentWorkers after context errors = %d, want 4", emb.rateLimiter.CurrentWorkers())
	}
}

func TestIsLMStudioTransientTransportError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "timeout", err: timeoutError{}, want: true},
		{name: "eof", err: io.EOF, want: true},
		{name: "reset", err: errors.New("read tcp: connection reset by peer"), want: true},
		{name: "other", err: errors.New("connection refused"), want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isLMStudioTransientTransportError(tt.err); got != tt.want {
				t.Fatalf("isLMStudioTransientTransportError() = %v, want %v", got, tt.want)
			}
		})
	}
}
