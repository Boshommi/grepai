package indexer

import (
	"context"
	"errors"
	"fmt"
	"log"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Boshommi/grepai/embedder"
	"github.com/Boshommi/grepai/store"
)

type Indexer struct {
	root          string
	store         store.VectorStore
	embedder      embedder.Embedder
	chunker       *Chunker
	scanner       *Scanner
	lastIndexTime time.Time
}

type IndexStats struct {
	FilesIndexed  int
	FilesSkipped  int
	ChunksCreated int
	FilesRemoved  int
	Duration      time.Duration
}

// ProgressInfo contains progress information for indexing
type ProgressInfo struct {
	Current     int    // Current file number (1-indexed)
	Total       int    // Total number of files
	CurrentFile string // Path of current file being processed
	KnownTotal  bool   // Whether Total is final and percentage-safe
}

// ProgressCallback is called for each file during indexing
type ProgressCallback func(info ProgressInfo)

// BatchProgressInfo contains progress information for batch embedding
type BatchProgressInfo struct {
	BatchIndex      int  // Current batch index (0-indexed)
	TotalBatches    int  // Total number of batches
	CompletedChunks int  // Number of chunks completed so far
	TotalChunks     int  // Total number of chunks to embed
	KnownTotal      bool // Whether TotalChunks/TotalBatches are final
	Retrying        bool // True if this is a retry attempt
	Attempt         int  // Retry attempt number (1-indexed, 0 if not retrying)
	StatusCode      int  // HTTP status code when retrying (429 = rate limited, 5xx = server error)
}

// BatchProgressCallback is called for batch embedding progress and retry visibility
type BatchProgressCallback func(info BatchProgressInfo)

func NewIndexer(
	root string,
	st store.VectorStore,
	emb embedder.Embedder,
	chunker *Chunker,
	scanner *Scanner,
	lastIndexTime time.Time,
) *Indexer {
	return &Indexer{
		root:          root,
		store:         st,
		embedder:      emb,
		chunker:       chunker,
		scanner:       scanner,
		lastIndexTime: lastIndexTime,
	}
}

// IndexAll performs a full index of the project (no progress reporting)
func (idx *Indexer) IndexAll(ctx context.Context) (*IndexStats, error) {
	return idx.IndexAllWithProgress(ctx, nil)
}

// IndexAllWithProgress performs a full index with progress reporting
func (idx *Indexer) IndexAllWithProgress(ctx context.Context, onProgress ProgressCallback) (*IndexStats, error) {
	return idx.IndexAllWithBatchProgress(ctx, onProgress, nil)
}

// IndexAllWithBatchProgress performs a full index with both file and batch progress reporting.
// When the embedder implements BatchEmbedder, files are processed in parallel using cross-file batching.
func (idx *Indexer) IndexAllWithBatchProgress(ctx context.Context, onProgress ProgressCallback, onBatchProgress BatchProgressCallback) (*IndexStats, error) {
	start := time.Now()
	stats := &IndexStats{}

	snapshots, err := idx.loadDocumentSnapshots(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to load document snapshots: %w", err)
	}
	remainingDocs := make(map[string]bool, len(snapshots))
	for path := range snapshots {
		remainingDocs[path] = true
	}

	counters := &indexPipelineCounters{}
	if batchEmbedder, ok := idx.embedder.(embedder.BatchEmbedder); ok {
		if err := idx.indexAllBatchPipeline(ctx, snapshots, remainingDocs, counters, onProgress, onBatchProgress, batchEmbedder); err != nil {
			return nil, err
		}
	} else {
		if err := idx.indexAllSequentialPipeline(ctx, snapshots, remainingDocs, counters, onProgress, onBatchProgress); err != nil {
			return nil, err
		}
	}

	for path := range remainingDocs {
		if err := idx.RemoveFile(ctx, path); err != nil {
			log.Printf("Failed to remove %s: %v", path, err)
			continue
		}
		stats.FilesRemoved++
	}

	stats.FilesIndexed = int(counters.filesIndexed.Load())
	stats.FilesSkipped = int(counters.filesSkipped.Load())
	stats.ChunksCreated = int(counters.chunksCreated.Load())

	stats.Duration = time.Since(start)
	return stats, nil
}

const (
	minScanWorkers       = 4
	maxScanWorkers       = 16
	pipelineBufferFactor = 2
	batchWindowSize      = 8
	batchFlushInterval   = 100 * time.Millisecond
	embeddingProbeFile   = "grepai_preflight_probe.go"
)

type indexPipelineCounters struct {
	filesIndexed  atomic.Int64
	filesSkipped  atomic.Int64
	chunksCreated atomic.Int64
}

type fileScanTask struct {
	meta             FileMeta
	snapshot         store.DocumentSnapshot
	hasSnapshot      bool
	existingChunkCnt int
}

type preparedFile struct {
	file             FileInfo
	chunkInfos       []ChunkInfo
	cachedVectors    map[int][]float32
	uncachedChunks   []ChunkInfo
	uncachedIndices  []int
	existingChunkCnt int
}

type batchFileState struct {
	prepared  *preparedFile
	vectors   [][]float32
	remaining int
}

type batchWindowRequest struct {
	windowIndex int
	batches     []embedder.Batch
	batchOffset int
}

type batchWindowResult struct {
	request   batchWindowRequest
	results   []embedder.BatchResult
	windowErr error
}

func isContextCancellation(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

func scanWorkerCount() int {
	workers := runtime.GOMAXPROCS(0) * 2
	if workers < minScanWorkers {
		workers = minScanWorkers
	}
	if workers > maxScanWorkers {
		workers = maxScanWorkers
	}
	return workers
}

func (idx *Indexer) loadDocumentSnapshots(ctx context.Context) (map[string]store.DocumentSnapshot, error) {
	if lister, ok := idx.store.(store.DocumentSnapshotLister); ok {
		snapshots, err := lister.ListDocumentSnapshots(ctx)
		if err != nil {
			return nil, err
		}
		out := make(map[string]store.DocumentSnapshot, len(snapshots))
		for _, snapshot := range snapshots {
			out[snapshot.Path] = snapshot
		}
		return out, nil
	}

	paths, err := idx.store.ListDocuments(ctx)
	if err != nil {
		return nil, err
	}

	out := make(map[string]store.DocumentSnapshot, len(paths))
	for _, path := range paths {
		doc, err := idx.store.GetDocument(ctx, path)
		if err != nil {
			return nil, err
		}
		if doc == nil {
			continue
		}
		out[path] = store.DocumentSnapshot{
			Path:       doc.Path,
			Hash:       doc.Hash,
			ModTime:    doc.ModTime,
			ChunkCount: len(doc.ChunkIDs),
		}
	}
	return out, nil
}

func (idx *Indexer) shouldSkipByLastIndex(meta FileMeta, snapshot store.DocumentSnapshot, hasSnapshot bool) bool {
	if idx.lastIndexTime.IsZero() || !hasSnapshot || snapshot.ChunkCount == 0 {
		return false
	}

	fileModTime := time.Unix(meta.ModTime, 0)
	return fileModTime.Before(idx.lastIndexTime) || fileModTime.Equal(idx.lastIndexTime)
}

func emitScanProgress(onProgress ProgressCallback, current, total int, file string, known bool) {
	if onProgress == nil {
		return
	}
	onProgress(ProgressInfo{
		Current:     current,
		Total:       total,
		CurrentFile: file,
		KnownTotal:  known,
	})
}

func emitBatchProgress(onProgress BatchProgressCallback, info BatchProgressInfo) {
	if onProgress == nil {
		return
	}
	onProgress(info)
}

func (idx *Indexer) walkScanTasks(
	ctx context.Context,
	snapshots map[string]store.DocumentSnapshot,
	remainingDocs map[string]bool,
	counters *indexPipelineCounters,
	onProgress ProgressCallback,
	scanTasks chan<- fileScanTask,
) error {
	discovered := 0
	err := idx.scanner.WalkMetadata(ctx, func(meta FileMeta) error {
		discovered++
		emitScanProgress(onProgress, discovered, discovered, meta.Path, false)

		snapshot, hasSnapshot := snapshots[meta.Path]
		delete(remainingDocs, meta.Path)
		if idx.shouldSkipByLastIndex(meta, snapshot, hasSnapshot) {
			counters.filesSkipped.Add(1)
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case scanTasks <- fileScanTask{
			meta:             meta,
			snapshot:         snapshot,
			hasSnapshot:      hasSnapshot,
			existingChunkCnt: snapshot.ChunkCount,
		}:
			return nil
		}
	}, func(_ string) {
		counters.filesSkipped.Add(1)
	})
	if err != nil {
		return err
	}

	emitScanProgress(onProgress, discovered, discovered, "", true)
	return nil
}

func (idx *Indexer) scanAndPrepareTask(ctx context.Context, task fileScanTask) (*preparedFile, bool, error) {
	file, err := idx.scanner.ScanFile(task.meta.Path)
	if err != nil {
		return nil, true, err
	}
	if file == nil {
		return nil, true, nil
	}

	if task.hasSnapshot && task.snapshot.Hash != "" && task.snapshot.Hash == file.Hash && task.snapshot.ChunkCount > 0 {
		return nil, false, nil
	}

	prepared, err := idx.prepareFileForIndexing(ctx, *file, task.existingChunkCnt)
	if err != nil {
		return nil, false, err
	}
	return prepared, false, nil
}

func (idx *Indexer) prepareFileForIndexing(ctx context.Context, file FileInfo, existingChunkCnt int) (*preparedFile, error) {
	chunkInfos := idx.chunker.ChunkWithContext(file.Path, file.Content)
	if len(chunkInfos) == 0 {
		return &preparedFile{
			file:             file,
			existingChunkCnt: existingChunkCnt,
		}, nil
	}

	cachedVectors, cacheHits, err := idx.lookupCachedEmbeddings(ctx, chunkInfos)
	if err != nil {
		return nil, err
	}
	if cacheHits > 0 {
		log.Printf("Reused %d cached embeddings for %s", cacheHits, file.Path)
	}

	prepared := &preparedFile{
		file:             file,
		chunkInfos:       chunkInfos,
		cachedVectors:    cachedVectors,
		existingChunkCnt: existingChunkCnt,
	}

	for i, chunk := range chunkInfos {
		if _, ok := cachedVectors[i]; ok {
			continue
		}
		prepared.uncachedIndices = append(prepared.uncachedIndices, i)
		prepared.uncachedChunks = append(prepared.uncachedChunks, chunk)
	}

	return prepared, nil
}

func (idx *Indexer) deleteExistingChunks(ctx context.Context, filePath string, existingChunkCnt int) error {
	if existingChunkCnt <= 0 {
		return nil
	}
	if err := idx.store.DeleteByFile(ctx, filePath); err != nil {
		return fmt.Errorf("failed to delete existing chunks for %s: %w", filePath, err)
	}
	return nil
}

func (idx *Indexer) savePreparedFile(ctx context.Context, prepared *preparedFile, finalChunks []ChunkInfo, vectors [][]float32) (int, error) {
	if err := idx.deleteExistingChunks(ctx, prepared.file.Path, prepared.existingChunkCnt); err != nil {
		return 0, err
	}
	if len(finalChunks) == 0 {
		return 0, nil
	}

	now := time.Now()
	chunks, chunkIDs := createStoreChunks(finalChunks, vectors, now)
	fd := fileChunkData{
		file:       prepared.file,
		chunkInfos: finalChunks,
	}
	if err := idx.saveFileData(ctx, fd, chunks, chunkIDs); err != nil {
		return 0, err
	}
	return len(chunks), nil
}

func (idx *Indexer) embedPreparedFileSequential(ctx context.Context, prepared *preparedFile) (int, int, error) {
	if len(prepared.chunkInfos) == 0 {
		if err := idx.deleteExistingChunks(ctx, prepared.file.Path, prepared.existingChunkCnt); err != nil {
			return 0, 0, err
		}
		return 0, 0, nil
	}

	cacheHits := len(prepared.cachedVectors)
	uncachedChunkCount := len(prepared.uncachedChunks)
	if uncachedChunkCount == 0 {
		vectors := make([][]float32, len(prepared.chunkInfos))
		for i := range prepared.chunkInfos {
			vectors[i] = prepared.cachedVectors[i]
		}
		chunksCreated, err := idx.savePreparedFile(ctx, prepared, prepared.chunkInfos, vectors)
		return chunksCreated, uncachedChunkCount, err
	}

	uncachedVectors, finalUncachedChunks, err := idx.embedWithReChunking(ctx, prepared.uncachedChunks)
	if err != nil {
		return 0, uncachedChunkCount, fmt.Errorf("failed to embed chunks: %w", err)
	}

	var vectors [][]float32
	var finalChunks []ChunkInfo

	if cacheHits == 0 {
		vectors = uncachedVectors
		finalChunks = finalUncachedChunks
	} else {
		vectors = make([][]float32, 0, len(prepared.chunkInfos))
		finalChunks = make([]ChunkInfo, 0, len(prepared.chunkInfos))

		uncachedIdx := 0
		for i, chunk := range prepared.chunkInfos {
			if vec, ok := prepared.cachedVectors[i]; ok {
				vectors = append(vectors, vec)
				finalChunks = append(finalChunks, chunk)
				continue
			}
			if uncachedIdx < len(uncachedVectors) && uncachedIdx < len(finalUncachedChunks) {
				vectors = append(vectors, uncachedVectors[uncachedIdx])
				finalChunks = append(finalChunks, finalUncachedChunks[uncachedIdx])
				uncachedIdx++
			}
		}
		for ; uncachedIdx < len(uncachedVectors); uncachedIdx++ {
			vectors = append(vectors, uncachedVectors[uncachedIdx])
			finalChunks = append(finalChunks, finalUncachedChunks[uncachedIdx])
		}
	}

	chunksCreated, err := idx.savePreparedFile(ctx, prepared, finalChunks, vectors)
	return chunksCreated, uncachedChunkCount, err
}

func (idx *Indexer) startPrepareWorkers(
	ctx context.Context,
	workers int,
	scanTasks <-chan fileScanTask,
	preparedCh chan<- *preparedFile,
	counters *indexPipelineCounters,
) *sync.WaitGroup {
	var workerWG sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		workerWG.Add(1)
		go func() {
			defer workerWG.Done()
			for task := range scanTasks {
				prepared, countedSkip, err := idx.scanAndPrepareTask(ctx, task)
				if err != nil {
					if isContextCancellation(err) || ctx.Err() != nil {
						return
					}
					log.Printf("Failed to scan %s: %v", task.meta.Path, err)
					if countedSkip {
						counters.filesSkipped.Add(1)
					}
					continue
				}
				if countedSkip {
					counters.filesSkipped.Add(1)
				}
				if prepared == nil {
					continue
				}

				select {
				case <-ctx.Done():
					return
				case preparedCh <- prepared:
				}
			}
		}()
	}
	return &workerWG
}

func (idx *Indexer) indexAllSequentialPipeline(
	ctx context.Context,
	snapshots map[string]store.DocumentSnapshot,
	remainingDocs map[string]bool,
	counters *indexPipelineCounters,
	onProgress ProgressCallback,
	onBatchProgress BatchProgressCallback,
) error {
	workers := scanWorkerCount()
	scanTasks := make(chan fileScanTask, workers*pipelineBufferFactor)
	preparedCh := make(chan *preparedFile, workers*pipelineBufferFactor)
	walkErrCh := make(chan error, 1)

	go func() {
		defer close(scanTasks)
		walkErrCh <- idx.walkScanTasks(ctx, snapshots, remainingDocs, counters, onProgress, scanTasks)
	}()

	workerWG := idx.startPrepareWorkers(ctx, workers, scanTasks, preparedCh, counters)
	go func() {
		workerWG.Wait()
		close(preparedCh)
	}()

	var (
		discoveredChunks atomic.Int64
		completedChunks  atomic.Int64
	)

	for prepared := range preparedCh {
		discoveredChunks.Add(int64(len(prepared.uncachedChunks)))
		emitBatchProgress(onBatchProgress, BatchProgressInfo{
			CompletedChunks: int(completedChunks.Load()),
			TotalChunks:     int(discoveredChunks.Load()),
			KnownTotal:      false,
		})

		chunksCreated, embeddedChunks, err := idx.embedPreparedFileSequential(ctx, prepared)
		if err != nil {
			if isContextCancellation(err) || ctx.Err() != nil {
				return err
			}
			log.Printf("Failed to index %s: %v", prepared.file.Path, err)
			continue
		}
		if chunksCreated > 0 {
			counters.filesIndexed.Add(1)
			counters.chunksCreated.Add(int64(chunksCreated))
		}

		if embeddedChunks > 0 {
			completedChunks.Add(int64(embeddedChunks))
		}
		emitBatchProgress(onBatchProgress, BatchProgressInfo{
			CompletedChunks: int(completedChunks.Load()),
			TotalChunks:     int(discoveredChunks.Load()),
			KnownTotal:      false,
		})
	}

	walkErr := <-walkErrCh
	if discoveredChunks.Load() > 0 {
		emitBatchProgress(onBatchProgress, BatchProgressInfo{
			CompletedChunks: int(completedChunks.Load()),
			TotalChunks:     int(discoveredChunks.Load()),
			KnownTotal:      true,
		})
	}
	if walkErr != nil {
		return fmt.Errorf("failed to scan files: %w", walkErr)
	}
	return nil
}

func reindexBatches(batches []embedder.Batch) []embedder.Batch {
	if len(batches) == 0 {
		return nil
	}

	reindexed := make([]embedder.Batch, len(batches))
	for i, batch := range batches {
		reindexed[i] = embedder.Batch{
			Index:   i,
			Entries: make([]embedder.BatchEntry, len(batch.Entries)),
		}
		copy(reindexed[i].Entries, batch.Entries)
	}
	return reindexed
}

func batchSizeTotal(batches []embedder.Batch) int {
	total := 0
	for _, batch := range batches {
		total += len(batch.Entries)
	}
	return total
}

func (idx *Indexer) startBatchEmbedWorker(
	ctx context.Context,
	batchEmb embedder.BatchEmbedder,
	requestCh <-chan batchWindowRequest,
	resultCh chan<- batchWindowResult,
	discoveredChunks *atomic.Int64,
	discoveredBatches *atomic.Int64,
	completedChunks *atomic.Int64,
	totalsKnown *atomic.Bool,
	onBatchProgress BatchProgressCallback,
) {
	go func() {
		for req := range requestCh {
			baseCompleted := int(completedChunks.Load())
			progress := func(batchIndex, totalBatches, windowCompleted, windowTotal int, retrying bool, attempt int, statusCode int) {
				globalCompleted := baseCompleted + windowCompleted
				completedChunks.Store(int64(globalCompleted))
				emitBatchProgress(onBatchProgress, BatchProgressInfo{
					BatchIndex:      req.batchOffset + batchIndex,
					TotalBatches:    int(discoveredBatches.Load()),
					CompletedChunks: globalCompleted,
					TotalChunks:     int(discoveredChunks.Load()),
					KnownTotal:      totalsKnown.Load(),
					Retrying:        retrying,
					Attempt:         attempt,
					StatusCode:      statusCode,
				})
			}

			results, err := batchEmb.EmbedBatches(ctx, req.batches, progress)
			if err == nil {
				completedChunks.Store(int64(baseCompleted + batchSizeTotal(req.batches)))
			}

			select {
			case <-ctx.Done():
				return
			case resultCh <- batchWindowResult{
				request:   req,
				results:   results,
				windowErr: err,
			}:
			}
		}
	}()
}

func (idx *Indexer) saveBatchReadyFile(ctx context.Context, state *batchFileState) (int, error) {
	return idx.savePreparedFile(ctx, state.prepared, state.prepared.chunkInfos, state.vectors)
}

func (idx *Indexer) indexAllBatchPipeline(
	ctx context.Context,
	snapshots map[string]store.DocumentSnapshot,
	remainingDocs map[string]bool,
	counters *indexPipelineCounters,
	onProgress ProgressCallback,
	onBatchProgress BatchProgressCallback,
	batchEmb embedder.BatchEmbedder,
) error {
	workers := scanWorkerCount()
	scanTasks := make(chan fileScanTask, workers*pipelineBufferFactor)
	preparedCh := make(chan *preparedFile, workers*pipelineBufferFactor)
	walkErrCh := make(chan error, 1)
	requestCh := make(chan batchWindowRequest, 1)
	resultCh := make(chan batchWindowResult, 1)

	go func() {
		defer close(scanTasks)
		walkErrCh <- idx.walkScanTasks(ctx, snapshots, remainingDocs, counters, onProgress, scanTasks)
	}()

	workerWG := idx.startPrepareWorkers(ctx, workers, scanTasks, preparedCh, counters)
	go func() {
		workerWG.Wait()
		close(preparedCh)
	}()

	limits := embedder.DefaultBatchLimits
	if limiter, ok := batchEmb.(embedder.BatchLimiter); ok {
		limits = limiter.BatchLimits()
	}
	builder := embedder.NewIncrementalBatchBuilder(limits)

	var (
		discoveredChunks  atomic.Int64
		discoveredBatches atomic.Int64
		completedChunks   atomic.Int64
		totalsKnown       atomic.Bool
	)
	idx.startBatchEmbedWorker(ctx, batchEmb, requestCh, resultCh, &discoveredChunks, &discoveredBatches, &completedChunks, &totalsKnown, onBatchProgress)

	pendingFiles := make(map[int]*batchFileState)
	windowQueue := make([]embedder.Batch, 0, batchWindowSize)
	windowInFlight := false
	preparedOpen := true
	nextFileID := 0
	nextBatchOffset := 0
	nextWindowIndex := 0
	ticker := time.NewTicker(batchFlushInterval)
	defer ticker.Stop()

	dispatchWindow := func(force bool) error {
		if windowInFlight || len(windowQueue) == 0 {
			return nil
		}
		if !force && len(windowQueue) < batchWindowSize {
			return nil
		}

		windowSize := len(windowQueue)
		if windowSize > batchWindowSize {
			windowSize = batchWindowSize
		}

		windowBatches := reindexBatches(windowQueue[:windowSize])
		req := batchWindowRequest{
			windowIndex: nextWindowIndex,
			batches:     windowBatches,
			batchOffset: nextBatchOffset,
		}
		nextWindowIndex++
		nextBatchOffset += len(windowBatches)
		windowQueue = windowQueue[windowSize:]
		windowInFlight = true

		select {
		case <-ctx.Done():
			return ctx.Err()
		case requestCh <- req:
			return nil
		}
	}

	for preparedOpen || windowInFlight || len(windowQueue) > 0 || len(pendingFiles) > 0 {
		select {
		case prepared, ok := <-preparedCh:
			if !ok {
				preparedOpen = false
				ready := builder.Flush()
				if len(ready) > 0 {
					discoveredBatches.Add(int64(len(ready)))
					windowQueue = append(windowQueue, ready...)
				}
				totalsKnown.Store(true)
				if discoveredChunks.Load() > 0 {
					emitBatchProgress(onBatchProgress, BatchProgressInfo{
						CompletedChunks: int(completedChunks.Load()),
						TotalChunks:     int(discoveredChunks.Load()),
						TotalBatches:    int(discoveredBatches.Load()),
						KnownTotal:      true,
					})
				}
				if err := dispatchWindow(true); err != nil {
					close(requestCh)
					return err
				}
				continue
			}

			discoveredChunks.Add(int64(len(prepared.uncachedChunks)))
			emitBatchProgress(onBatchProgress, BatchProgressInfo{
				CompletedChunks: int(completedChunks.Load()),
				TotalChunks:     int(discoveredChunks.Load()),
				TotalBatches:    int(discoveredBatches.Load()),
				KnownTotal:      false,
			})

			if len(prepared.chunkInfos) == 0 {
				if err := idx.deleteExistingChunks(ctx, prepared.file.Path, prepared.existingChunkCnt); err != nil {
					if isContextCancellation(err) || ctx.Err() != nil {
						close(requestCh)
						return err
					}
					log.Printf("Failed to index %s: %v", prepared.file.Path, err)
				}
				continue
			}

			if len(prepared.uncachedChunks) == 0 {
				vectors := make([][]float32, len(prepared.chunkInfos))
				for i := range prepared.chunkInfos {
					vectors[i] = prepared.cachedVectors[i]
				}
				chunksCreated, err := idx.savePreparedFile(ctx, prepared, prepared.chunkInfos, vectors)
				if err != nil {
					if isContextCancellation(err) || ctx.Err() != nil {
						close(requestCh)
						return err
					}
					log.Printf("Failed to index %s: %v", prepared.file.Path, err)
					continue
				}
				if chunksCreated > 0 {
					counters.filesIndexed.Add(1)
					counters.chunksCreated.Add(int64(chunksCreated))
				}
				continue
			}

			fileID := nextFileID
			nextFileID++
			state := &batchFileState{
				prepared:  prepared,
				vectors:   make([][]float32, len(prepared.chunkInfos)),
				remaining: len(prepared.uncachedChunks),
			}
			for i, vec := range prepared.cachedVectors {
				state.vectors[i] = vec
			}
			pendingFiles[fileID] = state

			for i, chunk := range prepared.uncachedChunks {
				ready := builder.Add(embedder.BatchEntry{
					FileIndex:  fileID,
					ChunkIndex: prepared.uncachedIndices[i],
					Content:    chunk.Content,
				})
				if len(ready) > 0 {
					discoveredBatches.Add(int64(len(ready)))
					windowQueue = append(windowQueue, ready...)
				}
			}

			if err := dispatchWindow(false); err != nil {
				close(requestCh)
				return err
			}

		case result := <-resultCh:
			windowInFlight = false
			if result.windowErr != nil {
				close(requestCh)
				if isContextCancellation(result.windowErr) || ctx.Err() != nil {
					return result.windowErr
				}
				return fmt.Errorf("failed to embed batches: %w", result.windowErr)
			}

			for _, batchResult := range result.results {
				batch := result.request.batches[batchResult.BatchIndex]
				for i, entry := range batch.Entries {
					if i >= len(batchResult.Embeddings) {
						continue
					}
					state, ok := pendingFiles[entry.FileIndex]
					if !ok {
						continue
					}
					state.vectors[entry.ChunkIndex] = batchResult.Embeddings[i]
					state.remaining--
					if state.remaining == 0 {
						chunksCreated, err := idx.saveBatchReadyFile(ctx, state)
						if err != nil {
							close(requestCh)
							return err
						}
						if chunksCreated > 0 {
							counters.filesIndexed.Add(1)
							counters.chunksCreated.Add(int64(chunksCreated))
						}
						delete(pendingFiles, entry.FileIndex)
					}
				}
			}

			if err := dispatchWindow(!preparedOpen); err != nil {
				close(requestCh)
				return err
			}

		case <-ticker.C:
			ready := builder.Flush()
			if len(ready) > 0 {
				discoveredBatches.Add(int64(len(ready)))
				windowQueue = append(windowQueue, ready...)
			}
			if err := dispatchWindow(true); err != nil {
				close(requestCh)
				return err
			}
		}
	}

	close(requestCh)
	walkErr := <-walkErrCh
	if discoveredChunks.Load() > 0 {
		emitBatchProgress(onBatchProgress, BatchProgressInfo{
			CompletedChunks: int(completedChunks.Load()),
			TotalChunks:     int(discoveredChunks.Load()),
			TotalBatches:    int(discoveredBatches.Load()),
			KnownTotal:      true,
		})
	}
	if walkErr != nil {
		return fmt.Errorf("failed to scan files: %w", walkErr)
	}
	return nil
}

// fileChunkData holds chunking information for a single file during batch processing.
type fileChunkData struct {
	fileIndex  int // Index in the files slice (for result mapping)
	file       FileInfo
	chunkInfos []ChunkInfo
}

// prepareFileChunks processes files by deleting existing chunks and creating new chunks.
// Returns the file data for storage and the file chunks for embedding.
func (idx *Indexer) prepareFileChunks(
	ctx context.Context,
	files []FileInfo,
) ([]fileChunkData, []embedder.FileChunks, error) {
	fileData := make([]fileChunkData, 0, len(files))
	fileChunks := make([]embedder.FileChunks, 0, len(files))

	for i, file := range files {
		if err := idx.store.DeleteByFile(ctx, file.Path); err != nil {
			return nil, nil, fmt.Errorf("failed to delete existing chunks for %s: %w", file.Path, err)
		}

		chunkInfos := idx.chunker.ChunkWithContext(file.Path, file.Content)
		if len(chunkInfos) == 0 {
			continue
		}

		contents := make([]string, len(chunkInfos))
		for j, c := range chunkInfos {
			contents[j] = c.Content
		}

		fileData = append(fileData, fileChunkData{
			fileIndex:  i,
			file:       file,
			chunkInfos: chunkInfos,
		})

		fileChunks = append(fileChunks, embedder.FileChunks{
			FileIndex: i,
			Chunks:    contents,
		})
	}

	return fileData, fileChunks, nil
}

// createStoreChunks creates store.Chunk objects from chunk info and embeddings.
func createStoreChunks(chunkInfos []ChunkInfo, embeddings [][]float32, now time.Time) ([]store.Chunk, []string) {
	chunks := make([]store.Chunk, len(chunkInfos))
	chunkIDs := make([]string, len(chunkInfos))

	for i, info := range chunkInfos {
		chunks[i] = store.Chunk{
			ID:          info.ID,
			FilePath:    info.FilePath,
			StartLine:   info.StartLine,
			EndLine:     info.EndLine,
			Content:     info.Content,
			Vector:      embeddings[i],
			Hash:        info.Hash,
			ContentHash: info.ContentHash,
			UpdatedAt:   now,
		}
		chunkIDs[i] = info.ID
	}

	return chunks, chunkIDs
}

// saveFileData saves chunks and document metadata for a single file.
func (idx *Indexer) saveFileData(ctx context.Context, fd fileChunkData, chunks []store.Chunk, chunkIDs []string) error {
	if err := idx.store.SaveChunks(ctx, chunks); err != nil {
		return fmt.Errorf("failed to save chunks for %s: %w", fd.file.Path, err)
	}

	doc := store.Document{
		Path:     fd.file.Path,
		Hash:     fd.file.Hash,
		ModTime:  time.Unix(fd.file.ModTime, 0),
		ChunkIDs: chunkIDs,
	}

	if err := idx.store.SaveDocument(ctx, doc); err != nil {
		return fmt.Errorf("failed to save document for %s: %w", fd.file.Path, err)
	}

	return nil
}

// wrapBatchProgress creates an embedder.BatchProgress callback from BatchProgressCallback.
func wrapBatchProgress(onProgress BatchProgressCallback) embedder.BatchProgress {
	if onProgress == nil {
		return nil
	}
	return func(batchIndex, totalBatches, completedChunks, totalChunks int, retrying bool, attempt int, statusCode int) {
		onProgress(BatchProgressInfo{
			BatchIndex:      batchIndex,
			TotalBatches:    totalBatches,
			CompletedChunks: completedChunks,
			TotalChunks:     totalChunks,
			KnownTotal:      true,
			Retrying:        retrying,
			Attempt:         attempt,
			StatusCode:      statusCode,
		})
	}
}

// indexFilesBatched indexes multiple files using cross-file batch embedding.
// It collects chunks from all files, forms batches, embeds them in parallel,
// then maps results back and stores them.
func (idx *Indexer) indexFilesBatched(
	ctx context.Context,
	files []FileInfo,
	batchEmb embedder.BatchEmbedder,
	onProgress BatchProgressCallback,
) (filesIndexed int, chunksCreated int, err error) {
	fileData, fileChunks, err := idx.prepareFileChunks(ctx, files)
	if err != nil {
		return 0, 0, err
	}

	if len(fileChunks) == 0 {
		return 0, 0, nil
	}

	// Check embedding cache for content-addressed deduplication
	cache, hasCache := idx.store.(store.EmbeddingCache)
	var totalCacheHits int

	// Pre-fill cached embeddings and filter out fully-cached files
	type preFilled struct {
		fdIndex   int
		vectors   [][]float32
		allCached bool
	}

	var preFilledFiles []preFilled
	var remainingFileData []fileChunkData
	var remainingFileChunks []embedder.FileChunks

	for i, fd := range fileData {
		if !hasCache {
			remainingFileData = append(remainingFileData, fd)
			remainingFileChunks = append(remainingFileChunks, fileChunks[i])
			continue
		}

		vecs := make([][]float32, len(fd.chunkInfos))
		allCached := true
		for j, chunk := range fd.chunkInfos {
			if chunk.ContentHash == "" {
				allCached = false
				continue
			}
			vec, found, err := cache.LookupByContentHash(ctx, chunk.ContentHash)
			if err != nil {
				log.Printf("Warning: cache lookup failed: %v", err)
				allCached = false
				continue
			}
			if found {
				vecs[j] = vec
				totalCacheHits++
			} else {
				allCached = false
			}
		}

		if allCached {
			preFilledFiles = append(preFilledFiles, preFilled{fdIndex: i, vectors: vecs, allCached: true})
		} else {
			remainingFileData = append(remainingFileData, fd)
			remainingFileChunks = append(remainingFileChunks, fileChunks[i])
		}
	}

	if totalCacheHits > 0 {
		log.Printf("Reused %d cached embeddings across %d files", totalCacheHits, len(preFilledFiles))
	}

	// Save fully-cached files immediately
	now := time.Now()
	for _, pf := range preFilledFiles {
		fd := fileData[pf.fdIndex]
		chunks, chunkIDs := createStoreChunks(fd.chunkInfos, pf.vectors, now)
		if err := idx.saveFileData(ctx, fd, chunks, chunkIDs); err != nil {
			return filesIndexed, chunksCreated, err
		}
		filesIndexed++
		chunksCreated += len(chunks)
	}

	// Embed remaining (non-cached) files
	if len(remainingFileChunks) > 0 {
		limits := embedder.DefaultBatchLimits
		if bl, ok := batchEmb.(embedder.BatchLimiter); ok {
			limits = bl.BatchLimits()
		}
		batches := embedder.FormBatches(remainingFileChunks, limits)
		results, err := batchEmb.EmbedBatches(ctx, batches, wrapBatchProgress(onProgress))
		if err != nil {
			return filesIndexed, chunksCreated, fmt.Errorf("failed to embed batches: %w", err)
		}

		fileEmbeddings := embedder.MapResultsToFiles(batches, results, len(files))

		for _, fd := range remainingFileData {
			embeddings := fileEmbeddings[fd.fileIndex]
			if len(embeddings) != len(fd.chunkInfos) {
				log.Printf("Warning: embedding count mismatch for %s: got %d, expected %d",
					fd.file.Path, len(embeddings), len(fd.chunkInfos))
				continue
			}
			chunks, chunkIDs := createStoreChunks(fd.chunkInfos, embeddings, now)
			if err := idx.saveFileData(ctx, fd, chunks, chunkIDs); err != nil {
				return filesIndexed, chunksCreated, err
			}
			filesIndexed++
			chunksCreated += len(chunks)
		}
	}

	return filesIndexed, chunksCreated, nil
}

// maxReChunkAttempts is the maximum number of times we'll try to re-chunk
// before giving up on a file.
const maxReChunkAttempts = 3

// IndexFile indexes a single file
func (idx *Indexer) IndexFile(ctx context.Context, file FileInfo) (int, error) {
	// Remove existing chunks for this file
	if err := idx.store.DeleteByFile(ctx, file.Path); err != nil {
		return 0, fmt.Errorf("failed to delete existing chunks: %w", err)
	}

	// Chunk the file
	chunkInfos := idx.chunker.ChunkWithContext(file.Path, file.Content)
	if len(chunkInfos) == 0 {
		return 0, nil
	}

	// Check embedding cache for content-addressed deduplication
	cachedVectors, cacheHits, err := idx.lookupCachedEmbeddings(ctx, chunkInfos)
	if err != nil {
		return 0, fmt.Errorf("failed to lookup cached embeddings: %w", err)
	}
	if cacheHits > 0 {
		log.Printf("Reused %d cached embeddings for %s", cacheHits, file.Path)
	}

	// Separate cached and uncached chunks
	var uncachedChunks []ChunkInfo
	for i, chunk := range chunkInfos {
		if _, ok := cachedVectors[i]; !ok {
			uncachedChunks = append(uncachedChunks, chunk)
		}
	}

	// Embed only uncached chunks
	var uncachedVectors [][]float32
	var finalUncachedChunks []ChunkInfo
	if len(uncachedChunks) > 0 {
		var err error
		uncachedVectors, finalUncachedChunks, err = idx.embedWithReChunking(ctx, uncachedChunks)
		if err != nil {
			return 0, fmt.Errorf("failed to embed chunks: %w", err)
		}
	}

	// Merge cached and freshly embedded results
	// If re-chunking happened, the final chunks may differ from original
	// In that case, we use the re-chunked results plus the cached ones
	var vectors [][]float32
	var finalChunks []ChunkInfo

	if cacheHits == 0 {
		// No cache hits - use embedding results directly
		vectors = uncachedVectors
		finalChunks = finalUncachedChunks
	} else if len(uncachedChunks) == 0 {
		// All cached - build vectors and chunks from cache
		vectors = make([][]float32, len(chunkInfos))
		for i := range chunkInfos {
			vectors[i] = cachedVectors[i]
		}
		finalChunks = chunkInfos
	} else {
		// Mix of cached and uncached - merge results
		// Note: if re-chunking changed uncached chunks, we can't easily merge
		// with the original indices. Fall back to simple merge.
		vectors = make([][]float32, 0, len(chunkInfos))
		finalChunks = make([]ChunkInfo, 0, len(chunkInfos))

		uncachedIdx := 0
		for i, chunk := range chunkInfos {
			if vec, ok := cachedVectors[i]; ok {
				vectors = append(vectors, vec)
				finalChunks = append(finalChunks, chunk)
			} else {
				// Check if re-chunking happened (uncachedVectors may have different length)
				if uncachedIdx < len(uncachedVectors) && uncachedIdx < len(finalUncachedChunks) {
					vectors = append(vectors, uncachedVectors[uncachedIdx])
					finalChunks = append(finalChunks, finalUncachedChunks[uncachedIdx])
					uncachedIdx++
				}
			}
		}
		// If re-chunking produced extra sub-chunks, append them
		for ; uncachedIdx < len(uncachedVectors); uncachedIdx++ {
			vectors = append(vectors, uncachedVectors[uncachedIdx])
			finalChunks = append(finalChunks, finalUncachedChunks[uncachedIdx])
		}
	}

	// Create store chunks
	now := time.Now()
	chunks := make([]store.Chunk, len(finalChunks))
	chunkIDs := make([]string, len(finalChunks))

	for i, info := range finalChunks {
		chunks[i] = store.Chunk{
			ID:          info.ID,
			FilePath:    info.FilePath,
			StartLine:   info.StartLine,
			EndLine:     info.EndLine,
			Content:     info.Content,
			Vector:      vectors[i],
			Hash:        info.Hash,
			ContentHash: info.ContentHash,
			UpdatedAt:   now,
		}
		chunkIDs[i] = info.ID
	}

	// Save chunks
	if err := idx.store.SaveChunks(ctx, chunks); err != nil {
		return 0, fmt.Errorf("failed to save chunks: %w", err)
	}

	// Save document metadata
	doc := store.Document{
		Path:     file.Path,
		Hash:     file.Hash,
		ModTime:  time.Unix(file.ModTime, 0),
		ChunkIDs: chunkIDs,
	}

	if err := idx.store.SaveDocument(ctx, doc); err != nil {
		return 0, fmt.Errorf("failed to save document: %w", err)
	}

	return len(chunks), nil
}

// embedWithReChunking attempts to embed chunks, automatically re-chunking
// any chunks that exceed the embedder's context limit.
func (idx *Indexer) embedWithReChunking(ctx context.Context, chunks []ChunkInfo) ([][]float32, []ChunkInfo, error) {
	contents := make([]string, len(chunks))
	for i, c := range chunks {
		contents[i] = c.Content
	}

	vectors, err := idx.embedder.EmbedBatch(ctx, contents)
	if err == nil {
		return vectors, chunks, nil
	}
	if isContextCancellation(err) || ctx.Err() != nil {
		return nil, nil, err
	}
	if embedder.AsContextLengthError(err) == nil {
		return nil, nil, err
	}

	var allVectors [][]float32
	var finalChunks []ChunkInfo
	for i, chunk := range chunks {
		chunkVectors, chunkInfos, err := idx.embedChunkWithReChunking(ctx, chunk, i, 1)
		if err != nil {
			return nil, nil, err
		}
		allVectors = append(allVectors, chunkVectors...)
		finalChunks = append(finalChunks, chunkInfos...)
	}

	return allVectors, finalChunks, nil
}

func (idx *Indexer) embedChunkWithReChunking(ctx context.Context, chunk ChunkInfo, chunkIndex, attempt int) ([][]float32, []ChunkInfo, error) {
	vector, err := idx.embedder.Embed(ctx, chunk.Content)
	if err == nil {
		return [][]float32{vector}, []ChunkInfo{chunk}, nil
	}
	if isContextCancellation(err) || ctx.Err() != nil {
		return nil, nil, err
	}
	if embedder.AsContextLengthError(err) == nil {
		return nil, nil, err
	}
	if attempt > maxReChunkAttempts {
		return nil, nil, fmt.Errorf("exceeded maximum re-chunk attempts (%d) for file", maxReChunkAttempts)
	}

	log.Printf("Re-chunking %s chunk %d (attempt %d/%d): context limit exceeded",
		chunk.FilePath, chunkIndex, attempt, maxReChunkAttempts)

	subChunks := idx.chunker.ReChunk(chunk, chunkIndex)
	if len(subChunks) == 0 {
		return nil, nil, fmt.Errorf("re-chunking produced no chunks for %s", chunk.FilePath)
	}

	log.Printf("Split chunk into %d sub-chunks", len(subChunks))

	var (
		allVectors [][]float32
		allChunks  []ChunkInfo
	)
	for subIndex, subChunk := range subChunks {
		subVectors, finalSubChunks, err := idx.embedChunkWithReChunking(ctx, subChunk, subIndex, attempt+1)
		if err != nil {
			return nil, nil, err
		}
		allVectors = append(allVectors, subVectors...)
		allChunks = append(allChunks, finalSubChunks...)
	}

	return allVectors, allChunks, nil
}

// lookupCachedEmbeddings checks if the store implements EmbeddingCache and returns
// cached vectors for chunks with matching content hashes. The returned map maps
// chunk index to cached vector. Chunks not in the map need fresh embedding.
func (idx *Indexer) lookupCachedEmbeddings(ctx context.Context, chunks []ChunkInfo) (map[int][]float32, int, error) {
	cache, ok := idx.store.(store.EmbeddingCache)
	if !ok {
		return nil, 0, nil
	}

	cached := make(map[int][]float32)
	for i, chunk := range chunks {
		if chunk.ContentHash == "" {
			continue
		}
		vec, found, err := cache.LookupByContentHash(ctx, chunk.ContentHash)
		if err != nil {
			if isContextCancellation(err) || ctx.Err() != nil {
				return nil, 0, err
			}
			log.Printf("Warning: cache lookup failed for content hash %s: %v", chunk.ContentHash[:8], err)
			continue
		}
		if found {
			cached[i] = vec
		}
	}

	return cached, len(cached), nil
}

func (idx *Indexer) PreflightEmbedding(ctx context.Context) error {
	if idx.embedder == nil || idx.chunker == nil {
		return nil
	}

	probe, ok := idx.buildEmbeddingProbeChunk()
	if !ok {
		return nil
	}

	if _, err := idx.embedder.EmbedBatch(ctx, []string{probe.Content}); err != nil {
		if isContextCancellation(err) || ctx.Err() != nil {
			return err
		}

		if ctxErr := embedder.AsContextLengthError(err); ctxErr != nil {
			message := fmt.Sprintf(
				"configured chunking.size=%d is incompatible with the embedding model context window",
				idx.chunker.ChunkSize(),
			)
			if ctxErr.MaxTokens > 0 {
				recommended := recommendedChunkSizeForLimit(idx.chunker.ChunkSize(), ctxErr.MaxTokens)
				message = fmt.Sprintf("%s (provider limit is about %d tokens; try %d)", message, ctxErr.MaxTokens, recommended)
			}
			return fmt.Errorf("%s: %w", message, err)
		}

		return fmt.Errorf("embedding batch probe failed: %w", err)
	}

	return nil
}

func (idx *Indexer) buildEmbeddingProbeChunk() (ChunkInfo, bool) {
	if idx.chunker == nil {
		return ChunkInfo{}, false
	}

	tokenCount := idx.chunker.ChunkSize()
	if tokenCount < 32 {
		tokenCount = 32
	}

	var builder strings.Builder
	builder.WriteString("File: ")
	builder.WriteString(embeddingProbeFile)
	builder.WriteString("\n\n")
	for i := 0; i < tokenCount; i++ {
		builder.WriteString("tok")
		builder.WriteString(fmt.Sprintf("%03d", i))
		builder.WriteString(" ")
	}

	return ChunkInfo{
		FilePath: embeddingProbeFile,
		Content:  builder.String(),
	}, true
}

func recommendedChunkSizeForLimit(currentChunkSize, maxTokens int) int {
	if maxTokens <= 0 {
		if currentChunkSize <= 64 {
			return 32
		}
		return currentChunkSize / 2
	}

	recommended := (maxTokens * 3) / 4
	if recommended < 32 {
		recommended = maxTokens / 2
	}
	if recommended < 16 {
		recommended = 16
	}
	if recommended >= currentChunkSize {
		recommended = currentChunkSize - 1
	}
	if recommended < 16 {
		recommended = 16
	}
	return recommended
}

// RemoveFile removes a file from the index
func (idx *Indexer) RemoveFile(ctx context.Context, path string) error {
	if err := idx.store.DeleteByFile(ctx, path); err != nil {
		return fmt.Errorf("failed to delete chunks: %w", err)
	}

	if err := idx.store.DeleteDocument(ctx, path); err != nil {
		return fmt.Errorf("failed to delete document: %w", err)
	}

	return nil
}

// NeedsReindex checks if a file needs reindexing
func (idx *Indexer) NeedsReindex(ctx context.Context, path string, hash string) (bool, error) {
	doc, err := idx.store.GetDocument(ctx, path)
	if err != nil {
		return false, err
	}

	if doc == nil {
		return true, nil
	}

	// Reindex if hash changed OR if document has no chunks (prior indexing failed)
	if doc.Hash != hash || len(doc.ChunkIDs) == 0 {
		return true, nil
	}

	return false, nil
}
