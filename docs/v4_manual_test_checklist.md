# E.D.I.T.H. V4 Manual Verification Checklist & Operational Guide

This document provides a verified, reproducible manual test checklist for interactive testing in the running E.D.I.T.H. Interactive CLI (`py -3.12 engine/edith_brain.py`).

> [!IMPORTANT]
> **Designated Test Target:** In accordance with explicit user instructions, all automated and manual message tests targeting external numbers must strictly use `8591211342` (`+918591211342`), mapped to `Test Recipient` in `data/contacts.json`.

---

## 1. WhatsApp Messaging & Communication Workflows

### Scenario 1.1: Exact Verbatim Dictation (Pre-Authorized Transmission)
- **User Input:**
  ```text
  Send a WhatsApp message to 8591211342 saying I will be 15 minutes late
  ```
- **Expected System Response:**
  - Contact resolves directly to `+918591211342` (`Test Recipient`).
  - Identifies verbatim dictation pattern.
  - Automatically transmits to WhatsApp Desktop / Web via `whatsapp_send` action without extra confirmation prompts.
  - Output indicates: `Message sent to Test Recipient (+918591211342) via WhatsApp: 'I will be 15 minutes late'`

### Scenario 1.2: Substantive Assistant Composition (Safety Gate & Preview)
- **User Input:**
  ```text
  Draft a WhatsApp message to 8591211342 apologizing for missing the meeting and proposing tomorrow 11 AM
  ```
- **Expected System Response:**
  - Assistant synthesizes a polite, substantive message body.
  - Retains draft in context and displays preview with version `v1`.
  - Prompts for user authorization before sending:
    - `"Draft prepared for Test Recipient (+918591211342): 'Hi, I apologize for missing our meeting today... Would tomorrow at 11:00 AM work?' Shall I send this?"`

### Scenario 1.3: In-Place Revision Cycle
- **Follow-up Input 1:**
  ```text
  make that message shorter
  ```
- **Expected System Response:**
  - In-place revision produces `v2` (concise version).
  - Shows updated preview.
- **Follow-up Input 2:**
  ```text
  add that I can share the meeting slides in advance
  ```
- **Expected System Response:**
  - Produces `v3` containing the appended note.
  - Prompts for confirmation: `Shall I send this?`

### Scenario 1.4: Explicit User Confirmation & Dispatch
- **Follow-up Input:**
  ```text
  yes, send it
  ```
- **Expected System Response:**
  - Dispatches `v3` to `8591211342`.
  - Confirms transmission and clears active draft.

### Scenario 1.5: Explicit Cancellation Safety Gate
- **User Input 1:**
  ```text
  Draft a message to Rahul Sharma saying the code is ready
  ```
- **Follow-up Input 2:**
  ```text
  cancel that
  ```
- **Expected System Response:**
  - Immediately aborts transmission and clears draft: `"Cancelled. The pending message draft has been discarded."`

### Scenario 1.6: First-Name Ambiguity Resolution
- **User Input 1:**
  ```text
  Text Rahul on WhatsApp saying the servers are rebooting
  ```
- **Expected System Response:**
  - Detects multiple contacts matching "Rahul" (`Rahul Sharma`, `Rahul Verma`).
  - Asks clarifying disambiguation query: `"Which Rahul did you mean? Rahul Sharma or Rahul Verma?"`
- **Follow-up Input 2:**
  ```text
  Rahul Sharma
  ```
- **Expected System Response:**
  - Resolves contact unambiguously to `Rahul Sharma` and executes action.

---

## 2. Cross-Application Compound Workflows

### Scenario 2.1: File Discovery + WhatsApp Transmission
- **User Input:**
  ```text
  Find the latest project report and send it to 8591211342 on WhatsApp
  ```
- **Expected System Response:**
  - Executes 3-step compound plan:
    1. `file_operation`: Finds most recently modified report matching query.
    2. `file_operation`: Verifies file accessibility and path.
    3. `whatsapp_send`: Prepares message with attachment path and awaits authorization.

### Scenario 2.2: Resume Discovery + Email Drafting
- **User Input:**
  ```text
  Find my resume, attach it to an email to the recruiter, and save the email as a draft
  ```
- **Expected System Response:**
  - Discovers latest resume (`.pdf` or `.docx`).
  - Synthesizes professional job application draft email with file attachment.
  - Writes draft safely without sending.

### Scenario 2.3: Document Extractive Summarization + Notepad Insertion
- **User Input:**
  ```text
  Open the project notes, create a short summary, and put it in Notepad
  ```
- **Expected System Response:**
  - Discovers `project_notes.txt`.
  - Performs local extractive summarization (zero cloud APIs).
  - Launches/focuses Notepad and faithfully types summary text.

---

## 3. Telemetry & Execution Trace Logging

All interactions automatically record diagnostic traces to `logs/traces/`:
- **Trace Fields:** `timestamp`, `session_id`, `turn_index`, `user_input`, `predicted_intent`, `plan_id`, `step_count`, `duration_ms`, `success`.
- **Sensitive Data Redaction:** Automatically masks phone numbers (except designated test target `8591211342`), passwords, tokens, and personal names.
