# Applied AI Director Coach

A free, static personal coaching tracker to prepare for Director, Data Science roles on Applied AI teams — specifically targeting Walmart Global Tech's Applied AI team.

**Live site:** https://reshshah.github.io/ds-coach/

> No API key. No backend. No cost. All coaching happens in Claude.ai — this app is your daily tracker, reference card, and notes log.

---

## How It Works

1. Open the app and find today's session in the **30-Day Plan** tab
2. Tap the session to get a **pre-written prompt** — copy it with one tap
3. Paste the prompt into **Claude.ai** and do your 1-hour coaching session
4. Come back to the app and **log your notes** from the session
5. Mark the day **Done** — your progress is saved locally

Everything is stored in your browser's `localStorage` — no login, no server, no cost.

---

## What's Inside

| Tab | What it does |
|-----|-------------|
| **Home** | Progress dashboard + auto-updated prompt for your next session |
| **30-Day Plan** | Daily session cards with copy-ready Claude.ai prompts and notes |
| **Interview Questions** | 8 Director-level questions with hints, prompts, and answer notes |
| **My Notes** | All your saved session notes with 🔊 Listen button for commuting |
| **Target Role** | Full JD, requirements breakdown, and technical depth expected |

---

## 30-Day Plan Overview

| Week | Focus |
|------|-------|
| Week 1 | Foundations: role decoding, demand forecasting, leadership, communication |
| Week 2 | Gen AI and agentic systems: LLMs, RAG, multi-agent frameworks, retail use cases |
| Week 3 | Optimization, MLOps, experimentation, data architecture |
| Week 4 | Interview mastery: behavioral, system design, mock interviews, negotiation |

---

## Accessing the Site

**URL:** https://reshshah.github.io/ds-coach/

**Add to phone home screen:**
- iPhone: Open URL in Safari → Share → Add to Home Screen
- Android: Open URL in Chrome → menu → Add to Home Screen

It will sit on your home screen like a native app — tap to open before your commute.

**Listen while driving:**
Every session note and interview question has a 🔊 Listen button that reads it aloud using your browser's built-in speech synthesis. No extra setup needed.

---

## Repo Structure

```
ds-coach/
├── index.html    # Entire app — one self-contained file, no dependencies
└── README.md     # This file
```

That's it. One file. No build step, no package.json, no node_modules.

---

## Making Changes

### Edit the coaching prompts or day topics
Open `index.html` → find `var WEEKS = [...]` → edit the `topic`, `summary`, or `prompt` field for any day.

### Edit the interview questions
Find `var QUESTIONS = [...]` → edit or add questions following this pattern:

```javascript
{
  id: 9,
  type: 'ml',              // ml, behavioral, or strategy
  company: 'Walmart',
  title: 'Your question title',
  question: 'Full question text here.',
  hints: [
    'Hint 1 — what to think about first',
    'Hint 2 — key concept to address',
    'Hint 3 — tradeoff to discuss',
    'Hint 4 — how to close your answer'
  ]
}
```

### Deploy changes
1. Edit `index.html` directly on GitHub (pencil icon) or locally
2. Commit to `main` branch
3. GitHub Pages updates automatically within 30-60 seconds

---

## GitHub Pages Setup (for reference)

If you ever need to re-enable:
1. Go to `github.com/reshshah/ds-coach` → **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: `main` → Folder: `/ (root)`
4. Click **Save**
5. Site live at `https://reshshah.github.io/ds-coach/` within ~60 seconds

---

## Using This With Claude.ai

This app is designed to work alongside Claude.ai — not instead of it.

**Daily workflow:**
1. Open `https://reshshah.github.io/ds-coach/` on your phone
2. Tap today's day → copy the pre-written prompt
3. Open [claude.ai](https://claude.ai) → paste the prompt → do your session
4. Come back to the app → log your notes → mark Done

**Your Claude project** (`claude.ai/project/...`) has your resume and job description saved — so Claude always has full context when you start a coaching session. No need to re-explain your background each time.

**On your commute:** open the My Notes tab → tap 🔊 Listen → hear your previous session notes read aloud while driving.

---

## Target Role

**Director, Data Science — Applied AI**
Walmart Global Tech · Applied AI Team (300+ scientists and engineers)
Base salary: $169,000 — $338,000 + annual bonus + stock

Key focus areas: demand forecasting, generative AI, agentic AI systems, mathematical optimization, MLOps, people leadership, stakeholder communication.

---

## Tech Stack

- **Frontend:** Vanilla HTML, CSS, JavaScript — zero dependencies, zero build step
- **Storage:** Browser `localStorage` — data stays on your device
- **Audio:** Browser Web Speech API — free, no API key needed
- **Hosting:** GitHub Pages — free
- **Coaching:** Claude.ai (your existing subscription)

---

*Built with Claude · Targeting Walmart Applied AI Director · 30 days · 1hr/day · $0 to run*
