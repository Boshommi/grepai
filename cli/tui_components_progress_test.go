package cli

import (
	"strings"
	"testing"
)

func TestProgressModel_ViewUsesDiscoveredStatusWhenTotalsUnknown(t *testing.T) {
	m := newProgressModel(newTUITheme())
	m.setSize(80)
	m.setScanProgress(3, 3, false)
	m.setEmbedProgress(2, 5, false)

	view := m.View()
	if !strings.Contains(view, "3 discovered") {
		t.Fatalf("expected scan discovered status in view, got %q", view)
	}
	if !strings.Contains(view, "2/5 discovered") {
		t.Fatalf("expected embed discovered status in view, got %q", view)
	}
}

func TestProgressModel_ViewUsesFixedTotalsWhenKnown(t *testing.T) {
	m := newProgressModel(newTUITheme())
	m.setSize(80)
	m.setScanProgress(3, 10, true)
	m.setEmbedProgress(4, 8, true)

	view := m.View()
	if !strings.Contains(view, "3/10") {
		t.Fatalf("expected scan fixed total in view, got %q", view)
	}
	if !strings.Contains(view, "4/8") {
		t.Fatalf("expected embed fixed total in view, got %q", view)
	}
}
