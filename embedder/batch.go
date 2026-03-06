package embedder

// MaxBatchSize is the maximum number of inputs per OpenAI embedding API call.
// OpenAI allows 2048, but we use 2000 as a safety margin.
const MaxBatchSize = 2000

// MaxBatchTokens is the maximum total tokens per OpenAI embedding API batch.
// OpenAI has a 300,000 token limit. We use 280,000 for safety margin.
const MaxBatchTokens = 280000

// DefaultCharsPerToken is the default characters-per-token ratio for estimation.
// Works well for OpenAI and most providers with English text.
const DefaultCharsPerToken = 4

// BatchLimits defines provider-specific batch constraints.
type BatchLimits struct {
	MaxSize       int // max inputs per batch
	MaxTokens     int // max tokens per batch
	CharsPerToken int // characters per token for estimation (0 = use DefaultCharsPerToken)
}

// charsPerToken returns the effective chars-per-token ratio.
func (l BatchLimits) charsPerToken() int {
	if l.CharsPerToken > 0 {
		return l.CharsPerToken
	}
	return DefaultCharsPerToken
}

// DefaultBatchLimits are the limits used by OpenAI and other providers that don't declare their own.
var DefaultBatchLimits = BatchLimits{MaxSize: MaxBatchSize, MaxTokens: MaxBatchTokens}

// EstimateTokens estimates the token count for a text string.
// Uses a conservative estimate of ~4 characters per token for English text.
func EstimateTokens(text string) int {
	return estimateTokensWithRatio(text, DefaultCharsPerToken)
}

func estimateTokensWithRatio(text string, charsPerToken int) int {
	return (len(text) + charsPerToken - 1) / charsPerToken
}

// BatchEntry represents a single chunk with metadata for tracking its source.
type BatchEntry struct {
	// FileIndex is the index of the source file in the files slice
	FileIndex int
	// ChunkIndex is the index of the chunk within the file's chunks
	ChunkIndex int
	// Content is the text content to embed
	Content string
}

// Batch represents a collection of chunks to be embedded in a single API call.
type Batch struct {
	// Entries contains chunks with source file tracking
	Entries []BatchEntry
	// Index is the batch number for progress reporting (0-indexed)
	Index int
}

// Size returns the number of entries in the batch.
func (b *Batch) Size() int {
	return len(b.Entries)
}

// Contents returns the text contents of all entries for embedding.
func (b *Batch) Contents() []string {
	contents := make([]string, len(b.Entries))
	for i, entry := range b.Entries {
		contents[i] = entry.Content
	}
	return contents
}

// FileChunks represents chunks from a single file for batch formation.
type FileChunks struct {
	// FileIndex is the index of this file in the original files slice
	FileIndex int
	// Chunks is the list of text chunks from this file
	Chunks []string
}

// batchBuilder accumulates chunks into batches.
type batchBuilder struct {
	batches       []Batch
	current       Batch
	currentTokens int
	limits        BatchLimits
}

func newBatchBuilder(estimatedBatches int, limits BatchLimits) *batchBuilder {
	return &batchBuilder{
		batches: make([]Batch, 0, estimatedBatches),
		current: Batch{
			Index:   0,
			Entries: make([]BatchEntry, 0, limits.MaxSize),
		},
		limits: limits,
	}
}

func (b *batchBuilder) isFull(additionalTokens int) bool {
	if len(b.current.Entries) >= b.limits.MaxSize {
		return true
	}
	if len(b.current.Entries) > 0 && b.currentTokens+additionalTokens > b.limits.MaxTokens {
		return true
	}
	return false
}

func (b *batchBuilder) finalizeCurrent() {
	b.batches = append(b.batches, b.current)
	b.current = Batch{
		Index:   len(b.batches),
		Entries: make([]BatchEntry, 0, b.limits.MaxSize),
	}
	b.currentTokens = 0
}

func (b *batchBuilder) add(fileIdx, chunkIdx int, content string, tokens int) {
	b.current.Entries = append(b.current.Entries, BatchEntry{
		FileIndex:  fileIdx,
		ChunkIndex: chunkIdx,
		Content:    content,
	})
	b.currentTokens += tokens
}

func (b *batchBuilder) build() []Batch {
	if len(b.current.Entries) > 0 {
		b.batches = append(b.batches, b.current)
	}
	return b.batches
}

// FormBatches splits chunks from multiple files into batches respecting both
// size (input count) and token limits from the provided BatchLimits.
// Chunks maintain their file/chunk index tracking for result mapping.
func FormBatches(files []FileChunks, limits BatchLimits) []Batch {
	totalChunks := countTotalChunks(files)
	if totalChunks == 0 {
		return nil
	}

	estimatedBatches := (totalChunks + limits.MaxSize - 1) / limits.MaxSize
	builder := newBatchBuilder(estimatedBatches, limits)

	cpt := limits.charsPerToken()
	for _, file := range files {
		for chunkIdx, chunk := range file.Chunks {
			tokens := estimateTokensWithRatio(chunk, cpt)
			if builder.isFull(tokens) {
				builder.finalizeCurrent()
			}
			builder.add(file.FileIndex, chunkIdx, chunk, tokens)
		}
	}

	return builder.build()
}

func countTotalChunks(files []FileChunks) int {
	total := 0
	for _, f := range files {
		total += len(f.Chunks)
	}
	return total
}

// BatchResult contains the embeddings for a batch with file/chunk index mapping.
type BatchResult struct {
	// BatchIndex is the index of the batch this result belongs to
	BatchIndex int
	// Embeddings contains the embedding vectors in the same order as batch entries
	Embeddings [][]float32
}

// IncrementalBatchBuilder forms batches incrementally as chunks arrive from a
// streaming scan/indexing pipeline.
type IncrementalBatchBuilder struct {
	builder *batchBuilder
}

func NewIncrementalBatchBuilder(limits BatchLimits) *IncrementalBatchBuilder {
	if limits.MaxSize <= 0 {
		limits = DefaultBatchLimits
	}
	return &IncrementalBatchBuilder{
		builder: newBatchBuilder(1, limits),
	}
}

// Add appends a single entry and returns any batches that became ready.
func (b *IncrementalBatchBuilder) Add(entry BatchEntry) []Batch {
	if b == nil {
		return nil
	}
	tokens := estimateTokensWithRatio(entry.Content, b.builder.limits.charsPerToken())
	if b.builder.isFull(tokens) {
		b.builder.finalizeCurrent()
	}
	b.builder.current.Entries = append(b.builder.current.Entries, entry)
	b.builder.currentTokens += tokens

	if len(b.builder.batches) == 0 {
		return nil
	}

	ready := make([]Batch, len(b.builder.batches))
	copy(ready, b.builder.batches)
	b.builder.batches = b.builder.batches[:0]
	return ready
}

// Flush finalizes the current partially-filled batch, if any, and returns all
// ready batches accumulated since the last Add/Flush call.
func (b *IncrementalBatchBuilder) Flush() []Batch {
	if b == nil {
		return nil
	}
	if len(b.builder.current.Entries) > 0 {
		b.builder.finalizeCurrent()
	}
	if len(b.builder.batches) == 0 {
		return nil
	}

	ready := make([]Batch, len(b.builder.batches))
	copy(ready, b.builder.batches)
	b.builder.batches = b.builder.batches[:0]
	return ready
}

// MapResultsToFiles maps batch results back to per-file embeddings.
// Returns a slice where each index corresponds to a file, containing embeddings for that file's chunks.
func MapResultsToFiles(batches []Batch, results []BatchResult, numFiles int) [][][]float32 {
	chunkCounts := countChunksPerFile(batches, numFiles)
	fileEmbeddings := allocateFileEmbeddings(chunkCounts)
	populateEmbeddings(fileEmbeddings, batches, results)
	return fileEmbeddings
}

func countChunksPerFile(batches []Batch, numFiles int) []int {
	counts := make([]int, numFiles)
	for _, batch := range batches {
		for _, entry := range batch.Entries {
			if entry.ChunkIndex+1 > counts[entry.FileIndex] {
				counts[entry.FileIndex] = entry.ChunkIndex + 1
			}
		}
	}
	return counts
}

func allocateFileEmbeddings(chunkCounts []int) [][][]float32 {
	embeddings := make([][][]float32, len(chunkCounts))
	for i, count := range chunkCounts {
		if count > 0 {
			embeddings[i] = make([][]float32, count)
		}
	}
	return embeddings
}

func populateEmbeddings(fileEmbeddings [][][]float32, batches []Batch, results []BatchResult) {
	for _, result := range results {
		batch := batches[result.BatchIndex]
		for i, entry := range batch.Entries {
			if i < len(result.Embeddings) {
				fileEmbeddings[entry.FileIndex][entry.ChunkIndex] = result.Embeddings[i]
			}
		}
	}
}
