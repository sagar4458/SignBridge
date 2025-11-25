# SignBridge — Roadmap

This is a living document. What's here reflects where the project actually stands and what I think is worth fixing, in the order I think makes sense to fix it.

---

## Where Things Stand Now

v2 is working. The landmark-based architecture (ViT-Tiny + BiLSTM on MediaPipe coordinates) is background-invariant and significantly more robust than v1's image-patch approach. The full pipeline — webcam to spoken sentence — is running end to end.

The honest gap: the model was trained on a Kaggle dataset with studio-quality images. Real hands in real rooms still cause accuracy drops for certain letters, particularly the ones that look geometrically similar (M/N/T/S, or R/U). This is a dataset problem, not an architecture problem — and there's a clear path to fixing it.

---

## Short-Term — Next 3–7 Days

### Fine-tune on personal hand data

This is the highest-impact thing that can be done right now with zero new infrastructure.

- Write `collect_landmarks.py` — a script that opens the webcam, prompts each letter one by one, and records 50–100 frames of MediaPipe landmarks per letter. Takes about 30 minutes to collect.
- Write `fine_tune_v2.py` — loads the existing `signbridge_v2.pth`, freezes the ViT layers, fine-tunes only the BiLSTM and classifier head on the personal dataset for 10–15 epochs.
- Replace the model file and re-test.

Expected outcome: accuracy on your own hand should jump to 85–95% for most letters. The fine-tuned model will know your hand proportions, your lighting, your webcam characteristics.

This also produces a demo-ready version — something that actually works reliably in a real environment rather than just on the training dataset.

---

## Medium-Term — 2–4 Weeks

### Make it accessible to others

Right now this requires cloning a repo, installing dependencies, and running a local server. That's fine for a portfolio piece but it limits who can actually use it.

**Live demo deployment**
Host the backend on a cheap cloud VM (AWS EC2 free tier or Google Cloud). Add a TLS certificate (Let's Encrypt) so the browser allows camera access over HTTPS. This gives a public URL that anyone can visit and try without installing anything.

**Demo video**
Record a proper walkthrough under good lighting with the fine-tuned model — showing letter recognition, word formation, LLM correction, and TTS in action. This is the most important thing for communicating what the project actually does to someone who hasn't used it.

**"Train on Your Own Hand" guide**
Add a section to the README with step-by-step instructions for running `collect_landmarks.py` and `fine_tune_v2.py`. This turns the project from a personal tool into something others can personalise for themselves.

**Blog post**
Write up the architecture decisions — why landmarks over image patches, why ViT treats each landmark as a token, why BiLSTM adds temporal memory. The dataset gap and the fine-tuning solution. Honest about what works and what doesn't.

---

## Long-Term — 1–3 Months

### A model that works for anyone, not just for me

The fine-tuned personal model is a workaround. The real fix is better training data.

**Google ASL Fingerspelling Dataset**
Google released a dataset with 3M+ fingerspelled characters from 100+ signers filmed in real-world conditions — diverse backgrounds, lighting, hand sizes, skin tones. Fine-tuning on this is the path to a model that generalises without per-user calibration.

**User profile system**
Even with a strong base model, a short calibration session (sign each letter once, 10–15 seconds total) that adapts the classifier for a specific user would meaningfully improve accuracy. This is few-shot personalisation — well-understood technique, practical to implement on top of the existing architecture.

**Mobile version**
The landmark-based v2 model processes 63 numbers per frame, not video frames. That's lightweight. Converting to ONNX and running inference in the browser via WebAssembly, or building a simple Android/iOS app, would make this genuinely portable — no laptop needed.

**Deeper LLM integration**
Right now Llama 3.2 corrects ASL grammar to natural English at the sentence level. The next step is context-aware correction — if someone has been spelling "doctor" and "hospital", the LLM should understand the conversation domain and make better corrections for ambiguous letters.

---

## Stretch Goals

These are further out and depend on whether the project gets traction or stays personal.

**Dynamic gestures (J and Z)**
J and Z in ASL involve motion — they're drawn in the air, not held static. The BiLSTM in v2 is already positioned to handle this with the right training data. Would need a motion-captured dataset or a custom collection session.

**Two-handed signs**
MediaPipe supports two-hand tracking. Some ASL letters and most common word signs use both hands. This would require a significant architecture change (two parallel landmark streams) but is technically tractable.

**Continuous signing**
Isolated letter recognition is solved. Continuous signing — natural hand transitions between letters with no pause — is the hard problem. This is where most production systems still fall short. It requires a completely different dataset and probably a sequence-to-sequence model rather than a per-frame classifier. Worth exploring eventually.

**Standalone kiosk**
A Raspberry Pi with a camera, a screen, and this model running offline. No internet, no cloud, no laptop. Relevant for the hospital counter / public terminal use case where infrastructure is constrained.

**Open dataset contribution**
Log sessions (with user consent) — landmarks, labels, attention weights. Aggregate into a diverse real-world dataset and release it. The Kaggle dataset problem is community-wide. Contributing better data is the highest-leverage thing this project could do for the field.

---

*Last updated: June 2026*