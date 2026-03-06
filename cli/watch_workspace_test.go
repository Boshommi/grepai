package cli

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/Boshommi/grepai/config"
	"github.com/Boshommi/grepai/daemon"
	"github.com/Boshommi/grepai/embedder"
	"github.com/Boshommi/grepai/indexer"
)

type stubWatchEmbedder struct {
	embedCalls atomic.Int32
	embedErr   error
}

func (s *stubWatchEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	s.embedCalls.Add(1)
	if s.embedErr != nil {
		return nil, s.embedErr
	}
	return []float32{0.1, 0.2, 0.3}, nil
}

func (s *stubWatchEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i := range texts {
		vectors[i] = []float32{0.1, 0.2, 0.3}
	}
	return vectors, nil
}

func (s *stubWatchEmbedder) Dimensions() int {
	return 3
}

func (s *stubWatchEmbedder) Close() error {
	return nil
}

func withWatchGlobals(t *testing.T, workspace string, status, stop, background bool) {
	t.Helper()
	oldWorkspace := watchWorkspace
	oldStatus := watchStatus
	oldStop := watchStop
	oldBackground := watchBackground
	oldLogDir := watchLogDir

	watchWorkspace = workspace
	watchStatus = status
	watchStop = stop
	watchBackground = background
	watchLogDir = ""

	t.Cleanup(func() {
		watchWorkspace = oldWorkspace
		watchStatus = oldStatus
		watchStop = oldStop
		watchBackground = oldBackground
		watchLogDir = oldLogDir
	})
}

func setupWorkspaceHome(t *testing.T, ws config.Workspace) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)

	cfg := config.DefaultWorkspaceConfig()
	cfg.Workspaces[ws.Name] = ws
	if err := config.SaveWorkspaceConfig(cfg); err != nil {
		t.Fatalf("SaveWorkspaceConfig() failed: %v", err)
	}
}

func TestShowWorkspaceWatchStatusNotRunning(t *testing.T) {
	logDir := t.TempDir()
	ws := &config.Workspace{Name: "ws"}

	if err := showWorkspaceWatchStatus(logDir, ws); err != nil {
		t.Fatalf("showWorkspaceWatchStatus() failed: %v", err)
	}
}

func TestShowWorkspaceWatchStatusRunning(t *testing.T) {
	logDir := t.TempDir()
	ws := &config.Workspace{
		Name: "ws",
		Projects: []config.ProjectEntry{
			{Name: "p1", Path: t.TempDir()},
			{Name: "p2", Path: t.TempDir()},
		},
	}

	pidPath := daemon.GetWorkspacePIDFile(logDir, ws.Name)
	content := strconv.Itoa(os.Getpid()) + "\n"
	if err := os.WriteFile(pidPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write workspace PID file: %v", err)
	}

	if err := showWorkspaceWatchStatus(logDir, ws); err != nil {
		t.Fatalf("showWorkspaceWatchStatus() failed: %v", err)
	}
}

func TestStopWorkspaceWatchDaemonNotRunning(t *testing.T) {
	logDir := t.TempDir()
	if err := stopWorkspaceWatchDaemon(logDir, "ws"); err != nil {
		t.Fatalf("stopWorkspaceWatchDaemon() failed: %v", err)
	}
}

func TestStartBackgroundWorkspaceWatchAlreadyRunning(t *testing.T) {
	logDir := t.TempDir()
	ws := &config.Workspace{Name: "ws"}

	pidPath := daemon.GetWorkspacePIDFile(logDir, ws.Name)
	content := strconv.Itoa(os.Getpid()) + "\n"
	if err := os.WriteFile(pidPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write workspace PID file: %v", err)
	}

	err := startBackgroundWorkspaceWatch(logDir, ws)
	if err == nil {
		t.Fatal("startBackgroundWorkspaceWatch() should fail when already running")
	}
	if !strings.Contains(err.Error(), "already running") {
		t.Fatalf("startBackgroundWorkspaceWatch() error = %q, want message containing %q", err.Error(), "already running")
	}
}

func TestRunWorkspaceWatchNoConfig(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	withWatchGlobals(t, "ws", false, false, false)

	err := runWorkspaceWatch(t.TempDir())
	if err == nil {
		t.Fatal("runWorkspaceWatch() should fail with no workspace config")
	}
	if !strings.Contains(err.Error(), "no workspaces configured") {
		t.Fatalf("runWorkspaceWatch() error = %q, want message containing %q", err.Error(), "no workspaces configured")
	}
}

func TestRunWorkspaceWatchStatusPath(t *testing.T) {
	projectPath := t.TempDir()
	ws := config.Workspace{
		Name: "ws",
		Store: config.StoreConfig{
			Backend: "postgres",
			Postgres: config.PostgresConfig{
				DSN: "postgres://localhost/test",
			},
		},
		Embedder: config.EmbedderConfig{Provider: "ollama"},
		Projects: []config.ProjectEntry{
			{Name: "p1", Path: projectPath},
		},
	}
	setupWorkspaceHome(t, ws)
	withWatchGlobals(t, ws.Name, true, false, false)

	if err := runWorkspaceWatch(t.TempDir()); err != nil {
		t.Fatalf("runWorkspaceWatch() failed: %v", err)
	}
}

func TestRunWorkspaceWatchRejectsUnsupportedBackend(t *testing.T) {
	projectPath := t.TempDir()
	ws := config.Workspace{
		Name: "ws",
		Store: config.StoreConfig{
			Backend: "gob",
		},
		Embedder: config.EmbedderConfig{Provider: "ollama"},
		Projects: []config.ProjectEntry{
			{Name: "p1", Path: projectPath},
		},
	}
	setupWorkspaceHome(t, ws)
	withWatchGlobals(t, ws.Name, false, false, false)

	err := runWorkspaceWatch(t.TempDir())
	if err == nil {
		t.Fatal("runWorkspaceWatch() should fail for unsupported backend")
	}
	if !strings.Contains(err.Error(), "not supported") {
		t.Fatalf("runWorkspaceWatch() error = %q, want message containing %q", err.Error(), "not supported")
	}
}

func TestRunWorkspaceWatchDetectsAlreadyRunning(t *testing.T) {
	projectPath := t.TempDir()
	logDir := t.TempDir()
	ws := config.Workspace{
		Name: "ws",
		Store: config.StoreConfig{
			Backend: "postgres",
			Postgres: config.PostgresConfig{
				DSN: "postgres://localhost/test",
			},
		},
		Embedder: config.EmbedderConfig{Provider: "ollama"},
		Projects: []config.ProjectEntry{
			{Name: "p1", Path: projectPath},
		},
	}
	setupWorkspaceHome(t, ws)
	withWatchGlobals(t, ws.Name, false, false, false)

	pidPath := daemon.GetWorkspacePIDFile(logDir, ws.Name)
	content := strconv.Itoa(os.Getpid()) + "\n"
	if err := os.WriteFile(pidPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write workspace PID file: %v", err)
	}

	err := runWorkspaceWatch(logDir)
	if err == nil {
		t.Fatal("runWorkspaceWatch() should fail when already running")
	}
	if !strings.Contains(err.Error(), "already running") {
		t.Fatalf("runWorkspaceWatch() error = %q, want message containing %q", err.Error(), "already running")
	}
}

func TestInitializeEmbedderUnknownProvider(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Embedder.Provider = "unknown"

	_, err := initializeEmbedder(context.Background(), cfg)
	if err == nil {
		t.Fatal("initializeEmbedder() should fail for unknown provider")
	}
	if !strings.Contains(err.Error(), "unknown embedding provider") {
		t.Fatalf("initializeEmbedder() error = %q, want message containing %q", err.Error(), "unknown embedding provider")
	}
}

func TestInitializeEmbedder_PerformsPreflightEmbedding(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Embedder.Provider = "openrouter"
	cfg.Embedder.APIKey = "test-key"
	cfg.Embedder.Model = "openai/text-embedding-3-small"

	stub := &stubWatchEmbedder{}
	oldFactory := watchEmbedderFactory
	watchEmbedderFactory = func(cfg *config.Config) (embedder.Embedder, error) {
		return stub, nil
	}
	t.Cleanup(func() {
		watchEmbedderFactory = oldFactory
	})

	emb, err := initializeEmbedder(context.Background(), cfg)
	if err != nil {
		t.Fatalf("initializeEmbedder() failed: %v", err)
	}
	defer emb.Close()

	if stub.embedCalls.Load() != 1 {
		t.Fatalf("expected exactly one preflight embedding request, got %d", stub.embedCalls.Load())
	}
}

func TestInitializeEmbedder_FailsWhenPreflightEmbeddingFails(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Embedder.Provider = "openrouter"
	cfg.Embedder.APIKey = "test-key"
	cfg.Embedder.Model = "openai/text-embedding-3-small"

	stub := &stubWatchEmbedder{embedErr: errors.New("boom")}
	oldFactory := watchEmbedderFactory
	watchEmbedderFactory = func(cfg *config.Config) (embedder.Embedder, error) {
		return stub, nil
	}
	t.Cleanup(func() {
		watchEmbedderFactory = oldFactory
	})

	_, err := initializeEmbedder(context.Background(), cfg)
	if err == nil {
		t.Fatal("initializeEmbedder() should fail when preflight embedding fails")
	}
	if !strings.Contains(err.Error(), "embedding preflight failed") {
		t.Fatalf("initializeEmbedder() error = %q, want message containing %q", err.Error(), "embedding preflight failed")
	}
	if stub.embedCalls.Load() != 1 {
		t.Fatalf("expected exactly one preflight embedding request, got %d", stub.embedCalls.Load())
	}
}

func TestInitializeStoreUnknownBackend(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.Store.Backend = "unknown"

	_, err := initializeStore(context.Background(), cfg, t.TempDir())
	if err == nil {
		t.Fatal("initializeStore() should fail for unknown backend")
	}
	if !strings.Contains(err.Error(), "unknown storage backend") {
		t.Fatalf("initializeStore() error = %q, want message containing %q", err.Error(), "unknown storage backend")
	}
}

func TestInitializeWorkspaceStoreUnsupportedBackend(t *testing.T) {
	ws := &config.Workspace{
		Name: "ws",
		Store: config.StoreConfig{
			Backend: "gob",
		},
	}

	_, err := initializeWorkspaceStore(context.Background(), ws)
	if err == nil {
		t.Fatal("initializeWorkspaceStore() should fail for unsupported backend")
	}
	if !strings.Contains(err.Error(), "unsupported backend for workspace") {
		t.Fatalf("initializeWorkspaceStore() error = %q, want message containing %q", err.Error(), "unsupported backend for workspace")
	}
}

func TestDiscoverWorktreesForWatchNonGitDirectory(t *testing.T) {
	got := discoverWorktreesForWatch(t.TempDir())
	if len(got) != 0 {
		t.Fatalf("discoverWorktreesForWatch() returned %d worktrees, want 0", len(got))
	}
}

func TestIsTracedLanguage(t *testing.T) {
	langs := []string{".go", ".py"}
	if !isTracedLanguage(".go", langs) {
		t.Fatal("isTracedLanguage(.go) = false, want true")
	}
	if isTracedLanguage(".js", langs) {
		t.Fatal("isTracedLanguage(.js) = true, want false")
	}
}

func TestPrintProgressAndBatchProgress(t *testing.T) {
	printProgress(indexer.ProgressInfo{Current: 0, Total: 0, CurrentFile: "ignored", KnownTotal: true})
	printProgress(indexer.ProgressInfo{Current: 1, Total: 2, CurrentFile: filepath.Join("very", "long", "path", "to", "file.go"), KnownTotal: true})

	printBatchProgress(indexer.BatchProgressInfo{
		Retrying:   true,
		StatusCode: 429,
		BatchIndex: 0,
		Attempt:    2,
	})
	printBatchProgress(indexer.BatchProgressInfo{
		Retrying:        false,
		TotalChunks:     10,
		CompletedChunks: 5,
	})
}
