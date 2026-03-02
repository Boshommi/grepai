package main

import (
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra/doc"
	"github.com/Boshommi/grepai/cli"
)

func main() {
	outputDir := "./docs/src/content/docs/commands"

	// Ensure output directory exists
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	rootCmd := cli.GetRootCmd()

	// Custom file prepender to add Starlight frontmatter
	filePrepender := func(filename string) string {
		name := filepath.Base(filename)
		name = strings.TrimSuffix(name, filepath.Ext(name))
		name = strings.ReplaceAll(name, "_", " ")

		// Make title more readable
		title := name
		if title == "grepai" {
			title = "grepai (root)"
		}

		return "---\ntitle: " + title + "\ndescription: CLI reference for " + name + "\n---\n\n"
	}

	// Custom link handler for internal links
	linkHandler := func(name string) string {
		base := strings.TrimSuffix(name, filepath.Ext(name))
		return "/grepai/commands/" + strings.ToLower(base) + "/"
	}

	err := doc.GenMarkdownTreeCustom(rootCmd, outputDir, filePrepender, linkHandler)
	if err != nil {
		log.Fatalf("Failed to generate documentation: %v", err)
	}

	log.Printf("Documentation generated successfully in %s", outputDir)
}
