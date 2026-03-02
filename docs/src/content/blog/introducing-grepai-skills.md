---
title: "Introducing grepai-skills: 27 Skills to Supercharge Your AI Agent"
description: "Announcing grepai-skills, a collection of 27 AI agent skills that teach Claude Code, Cursor, and other AI assistants how to master semantic code search."
pubDate: 2026-01-28
author: Yoan Bernabeu
tags:
  - announcement
  - skills
  - claude-code
  - cursor
---

## TL;DR

We're releasing **[grepai-skills](https://github.com/Boshommi/grepai-skills)**, a collection of 27 ready-to-use skills that teach your AI coding agent how to leverage grepai effectively. This open-source project lives in its own dedicated repository. One command installs everything:

```bash
npx skills add yoanbernabeu/grepai-skills
```

Works with Claude Code, Cursor, Windsurf, Codex, and 30+ other AI agents.

---

## The Problem: AI Agents Don't Know Your Tools

You've installed grepai. Your codebase is indexed. Semantic search is ready. But when you ask your AI agent to search for "authentication logic," it still falls back to `grep` and `find`, launching subagents, burning tokens, and missing the semantic understanding grepai provides.

**Why?** Because AI agents don't magically know how to use every tool in your environment. They need guidance—specific instructions on when and how to use grepai instead of traditional search.

That's exactly what **skills** solve.

---

## What Are Skills?

Skills are knowledge modules that AI agents can load to understand specific tools. Think of them as instruction manuals your agent can actually read and follow.

Instead of hoping your agent figures out grepai on its own, skills provide:

- **Step-by-step workflows** for common tasks
- **Best practices** for writing effective queries
- **Configuration examples** for different setups
- **Troubleshooting guides** when things go wrong

The [skills ecosystem](https://skills.sh) already hosts over 30,000 skills for various tools, and grepai now joins the party.

---

## Meet grepai-skills

We've created **27 skills** organized into 9 categories, covering everything from installation to advanced call graph analysis:

| Category | Skills | What You'll Learn |
|----------|--------|-------------------|
| **Getting Started** | 3 | Installation, Ollama setup, quickstart |
| **Configuration** | 3 | Init, config reference, ignore patterns |
| **Embeddings** | 3 | Ollama, OpenAI, LM Studio providers |
| **Storage** | 3 | GOB, PostgreSQL, Qdrant backends |
| **Indexing** | 2 | Watch daemon, chunking optimization |
| **Search** | 4 | Basics, advanced, tips, boosting |
| **Call Graph** | 3 | Callers, callees, dependency graphs |
| **Integration** | 3 | Claude Code, Cursor, MCP tools |
| **Advanced** | 3 | Workspaces, languages, troubleshooting |

---

## Installation

### One Command, All Skills

```bash
npx skills add yoanbernabeu/grepai-skills
```

This works with any compatible agent. The CLI automatically detects which agents you're using and installs skills to the right locations.

### Install Specific Categories

Don't need everything? Install only what you need:

```bash
# Just search-related skills
npx skills add yoanbernabeu/grepai-skills --skill grepai-search-basics

# Install globally (available in all projects)
npx skills add yoanbernabeu/grepai-skills -g

# See all available skills
npx skills add yoanbernabeu/grepai-skills --list
```

### Claude Code Plugin

If you prefer the plugin system:

```bash
/plugin marketplace add yoanbernabeu/grepai-skills
/plugin install grepai-complete@grepai-skills
```

---

## How It Works

Once skills are installed, your AI agent gains contextual knowledge about grepai. Ask naturally:

> "Search for error handling code in this project"

Your agent now knows to use `grepai search "error handling"` instead of launching multiple grep searches across your codebase.

> "What functions call the Login function?"

The agent uses `grepai trace callers "Login"` to map dependencies instantly.

> "Why are my search results poor?"

The troubleshooting skill kicks in with specific diagnostic steps.

**No prompting gymnastics. No manual instructions. Just natural conversation.**

---

## Why This Matters

In our [recent benchmark](/grepai/blog/benchmark-grepai-vs-grep-claude-code/), we showed that grepai can reduce API costs by 27.5% and input tokens by 97% compared to traditional grep workflows. But those savings only happen when the agent actually *uses* grepai.

Skills bridge that gap. They transform grepai from "a tool your agent might use" to "the default way your agent searches code."

---

## Get Started

1. **Install grepai**: Follow the [installation guide](/grepai/installation/) or run `brew install yoanbernabeu/tap/grepai`
2. **Initialize your project**: `grepai init && grepai watch`
3. **Install skills**: `npx skills add yoanbernabeu/grepai-skills`
4. **Ask naturally** about your codebase

The skills take care of the rest.

---

**Resources:**
- [Skills Documentation](/grepai/skills/)
- [grepai-skills Repository](https://github.com/Boshommi/grepai-skills)
- [Skills Ecosystem](https://skills.sh)
