package embedder

import (
	"encoding/json"
	"testing"
)

func TestNewVoyageAIEmbedder_Defaults(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "test-key")

	e, err := NewVoyageAIEmbedder()
	if err != nil {
		t.Fatalf("failed to create VoyageAIEmbedder: %v", err)
	}

	if e.endpoint != defaultVoyageAIEndpoint {
		t.Errorf("expected endpoint %s, got %s", defaultVoyageAIEndpoint, e.endpoint)
	}

	if e.model != defaultVoyageAIModel {
		t.Errorf("expected model %s, got %s", defaultVoyageAIModel, e.model)
	}

	if e.dimensions != nil {
		t.Errorf("expected nil dimensions, got %v", e.dimensions)
	}

	if e.outputDimension != nil {
		t.Errorf("expected nil outputDimension, got %v", e.outputDimension)
	}
}

func TestNewVoyageAIEmbedder_WithOptions(t *testing.T) {
	customEndpoint := "https://custom.voyageai.com/v1"
	customModel := "voyage-4-large"
	customKey := "pa-custom-key"
	customDimensions := 512

	e, err := NewVoyageAIEmbedder(
		WithVoyageAIEndpoint(customEndpoint),
		WithVoyageAIModel(customModel),
		WithVoyageAIKey(customKey),
		WithVoyageAIDimensions(customDimensions),
	)
	if err != nil {
		t.Fatalf("failed to create VoyageAIEmbedder: %v", err)
	}

	if e.endpoint != customEndpoint {
		t.Errorf("expected endpoint %s, got %s", customEndpoint, e.endpoint)
	}

	if e.model != customModel {
		t.Errorf("expected model %s, got %s", customModel, e.model)
	}

	if e.apiKey != customKey {
		t.Errorf("expected apiKey %s, got %s", customKey, e.apiKey)
	}

	if e.dimensions == nil || *e.dimensions != customDimensions {
		t.Errorf("expected dimensions %d, got %v", customDimensions, e.dimensions)
	}

	if e.outputDimension == nil || *e.outputDimension != customDimensions {
		t.Errorf("expected outputDimension %d, got %v", customDimensions, e.outputDimension)
	}
}

func TestNewVoyageAIEmbedder_RequiresAPIKey(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "")

	_, err := NewVoyageAIEmbedder()
	if err == nil {
		t.Fatal("expected error when API key is not set")
	}
}

func TestNewVoyageAIEmbedder_UsesEnvAPIKey(t *testing.T) {
	envKey := "pa-env-test-key"
	t.Setenv("VOYAGE_API_KEY", envKey)

	e, err := NewVoyageAIEmbedder()
	if err != nil {
		t.Fatalf("failed to create VoyageAIEmbedder: %v", err)
	}

	if e.apiKey != envKey {
		t.Errorf("expected apiKey from env %s, got %s", envKey, e.apiKey)
	}
}

func TestNewVoyageAIEmbedder_ExplicitKeyOverridesEnv(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "env-key")
	explicitKey := "pa-explicit-key"

	e, err := NewVoyageAIEmbedder(WithVoyageAIKey(explicitKey))
	if err != nil {
		t.Fatalf("failed to create VoyageAIEmbedder: %v", err)
	}

	if e.apiKey != explicitKey {
		t.Errorf("expected explicit apiKey %s, got %s", explicitKey, e.apiKey)
	}
}

func TestVoyageAIEmbedder_Dimensions(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "test-key")

	tests := []struct {
		name       string
		dimensions int
	}{
		{"default", voyageAIDimensions},
		{"custom 256", 256},
		{"custom 512", 512},
		{"custom 2048", 2048},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var e *VoyageAIEmbedder
			var err error
			if tt.dimensions == voyageAIDimensions {
				e, err = NewVoyageAIEmbedder()
			} else {
				e, err = NewVoyageAIEmbedder(WithVoyageAIDimensions(tt.dimensions))
			}
			if err != nil {
				t.Fatalf("failed to create embedder: %v", err)
			}

			if e.Dimensions() != tt.dimensions {
				t.Errorf("expected Dimensions() to return %d, got %d", tt.dimensions, e.Dimensions())
			}
		})
	}
}

func TestVoyageAIEmbedder_BatchLimits(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "test-key")

	e, err := NewVoyageAIEmbedder()
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}

	limits := e.BatchLimits()
	if limits.MaxSize != 128 {
		t.Errorf("expected BatchLimits().MaxSize = 128, got %d", limits.MaxSize)
	}
	if limits.MaxTokens != 110000 {
		t.Errorf("expected BatchLimits().MaxTokens = 110000, got %d", limits.MaxTokens)
	}
	if limits.CharsPerToken != 2 {
		t.Errorf("expected BatchLimits().CharsPerToken = 2, got %d", limits.CharsPerToken)
	}
}

func TestVoyageAIEmbedder_Close(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "test-key")

	e, err := NewVoyageAIEmbedder()
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Errorf("Close() returned error: %v", err)
	}
}

func TestVoyageAIEmbedder_EndpointVariants(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "test-key")

	tests := []struct {
		name     string
		endpoint string
	}{
		{"default", "https://api.voyageai.com/v1"},
		{"custom proxy", "https://proxy.example.com/v1"},
		{"local proxy", "http://localhost:8080/v1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, err := NewVoyageAIEmbedder(WithVoyageAIEndpoint(tt.endpoint))
			if err != nil {
				t.Fatalf("failed to create embedder: %v", err)
			}
			if e.endpoint != tt.endpoint {
				t.Errorf("expected endpoint %s, got %s", tt.endpoint, e.endpoint)
			}
		})
	}
}

func TestVoyageAIEmbedRequest_Serialization(t *testing.T) {
	dim := 512
	req := voyageAIEmbedRequest{
		Model:           "voyage-code-3",
		Input:           []string{"hello world"},
		InputType:       "document",
		OutputDimension: &dim,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	jsonStr := string(data)

	// Must use "output_dimension", NOT "dimensions"
	if !contains(jsonStr, `"output_dimension":512`) {
		t.Errorf("expected output_dimension in JSON, got: %s", jsonStr)
	}
	if contains(jsonStr, `"dimensions"`) {
		t.Errorf("unexpected 'dimensions' field in JSON (should be 'output_dimension'): %s", jsonStr)
	}
}

func TestVoyageAIEmbedRequest_InputType(t *testing.T) {
	req := voyageAIEmbedRequest{
		Model:     "voyage-code-3",
		Input:     []string{"test"},
		InputType: "document",
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	jsonStr := string(data)
	if !contains(jsonStr, `"input_type":"document"`) {
		t.Errorf("expected input_type in JSON, got: %s", jsonStr)
	}
}

func TestVoyageAIEmbedRequest_OmitsNilOutputDimension(t *testing.T) {
	req := voyageAIEmbedRequest{
		Model:     "voyage-code-3",
		Input:     []string{"test"},
		InputType: "document",
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	jsonStr := string(data)
	if contains(jsonStr, "output_dimension") {
		t.Errorf("expected output_dimension to be omitted when nil, got: %s", jsonStr)
	}
}

func TestVoyageAIErrorResponse_Parsing(t *testing.T) {
	// Voyage AI returns errors as {"detail": "..."} not {"error": {"message": "..."}}
	body := []byte(`{"detail":"Invalid API key provided"}`)

	var errResp voyageAIErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil {
		t.Fatalf("failed to unmarshal error response: %v", err)
	}

	if errResp.Detail != "Invalid API key provided" {
		t.Errorf("expected detail 'Invalid API key provided', got %q", errResp.Detail)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
