# ChatGPT Session Notes: AI Tools for Video Parsing

**Date:** 2026-02-15, 2:48 AM
**Source:** `~/Desktop/AI Tools for Video Parsing.pdf` (4 pages)
**ChatGPT session URL:** `https://chatgpt.com/c/6990c105-a2f8-832c-b0ed-eef01243fc1b`

## The Request

> "What AI tools exist for video parsing? It would be great to have a video tool that can watch my ludobot videos and evaluate the extent to which they embody the input concept."

Goal: build a **Ludobot Video Critic** — automated evaluation of concept embodiment from video, not just telemetry.

## ChatGPT's Landscape Survey (5 categories)

### 1. Managed "Video Intelligence" APIs (fastest to try)
- **Google Cloud Video Intelligence** — labels, person detection, text detection, object tracking, shot changes, speech transcription
- **Azure AI Video Indexer** — indexing + "30+ model" style insight extraction with transcripts, objects, OCR, topics, summaries, JSON outputs
- **Amazon Rekognition Video** — label detection plus segment markers like shot changes/black frames

Best when: you want bulk processing and are happy with cloud.

### 2. Specialist "Search Inside Video" Tools (semantic retrieval across clips)
- **Twelve Labs** — builds embeddings for video and supports semantic search and summarization over video libraries

ChatGPT calls this the "search engine for your gait zoo vibe" — find all clips where the robot crabwalks / cartwheels / drags a leg / recovers from a shove.

### 3. Multimodal LLMs That Reason About Video
- **Gemini video understanding** (Gemini API / Vertex AI) — explicitly supports processing audio + visual streams and answering questions / generating descriptions
- **OpenAI approach (API)** — sample frames from video and provide them as images for "video understanding via vision"

Rule of thumb: LLMs are excellent judges *if you give them evidence* (timestamps, keyframes, derived motion stats), not just raw pixels.

### 4. Local-First Building Blocks (reproducible + hackable)

**Segmentation / tracking:**
- **SAM 2** — promptable segmentation in images *and videos*
- **ByteTrack** — robust multi-object tracking once you have detections
- **Ultralytics YOLO (track mode)** — detection + tracking workflows

**Pose / keypoints:**
- **DeepLabCut** — markerless pose estimation, "object agnostic" — label what you care about
- **SLEAP** — pose tracking toolkit with a labeling/training GUI (built for animals but works for "things with parts")

**Action recognition / classifiers:**
- **MMAction2** — open-source toolbox for action recognition, spatiotemporal detection, skeleton-based recognition, video retrieval

ChatGPT notes: since Ludobot videos are PyBullet screen-captures with simple backgrounds, local tools can get *shockingly far*.

### 5. The Actual Architecture: A "Concept Embodiment Scorer"

A two-layer system:

**Layer A: Turn video into measurements**
- Centroid trajectory (where does the robot go, how fast, how straight)
- Heading / yaw over time (does it rotate, does it face travel direction)
- Periodicity (is there a stable cycle, step frequency proxy via oscillations)
- Stability proxies (falls, flips, sudden vertical jolts, time spent "prone")

Achievable with SAM 2 segmentation + heading + simple geometry, then stabilize identity with tracking. For limb timing: DeepLabCut style pose tracking (label 6-10 points).

**Layer B: Convert "Concept X" into a rubric, then score**

Per-concept rubrics, e.g.:
- **"Crabwalk"**: sideways translation dominates forward; yaw stays within threshold; alternating leg phases visible
- **"Cartwheel"**: repeated full-body roll events; angular velocity spikes; translation maybe secondary
- **"Resilience to disruption"**: recovery time after perturbation; gait returns to prior frequency; path deviation bounded

Then: compute a 0-5 score from metrics, and ask a multimodal LLM for a human-facing critique with timestamps ("at 00:07 it transitions into..."). Gemini is especially direct for this.

## Recommended Starter Stack for Ludobot Videos

1. **Gemini video understanding** — quick qualitative judgments and time-stamped explanations
2. **SAM 2 + tracking** — lightweight local measurement layer (segmentation + centroid path) so the judge has numbers
3. **Twelve Labs** — if you want a searchable "gait zoo by concept" style embeddings/search

## The Key Insight: Error Back-Propagation from Video

Kathryn's response:

> "Twelve Labs and Gemini video seem like my best bet in terms of what I am trying to do: to create a mechanism for error back-propagation based on what the video looks like, not just telemetry data."

This is the missing piece in the call-and-response pipeline. Currently:
1. Human says concept → LLM generates weights → robot executes → **human evaluates** (manual)

With a video critic:
1. Human says concept → LLM generates weights → robot executes → **vision model evaluates** (automated) → feedback to LLM

This closes the loop. The vision model becomes the "error signal" — it watches the video and tells the LLM whether the behavior matches the concept, creating a feedback channel that doesn't require human-in-the-loop evaluation for every trial.

## Relationship to Other Work

- The gait glyph + CLIP embedding idea from the [full ChatGPT conversation](chatgpt_full_conversation_notes.md) is a lighter version of the same goal (visual similarity without per-concept rubrics)
- The sensor design in [sensor_design_specification.md](sensor_design_specification.md) improves the robot's internal sensing; this improves external evaluation
- Together, they address both sides of the communication channel: the robot's ability to express (sensors/body) and the human's ability to evaluate (video critic)

## Source

- PDF: `~/Desktop/AI Tools for Video Parsing.pdf` (4 pages, 2026-02-15 2:48 AM)
