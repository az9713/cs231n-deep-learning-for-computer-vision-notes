# CS231N — Deep Learning for Computer Vision

An independent enhanced textbook following all 18 Stanford CS231N Spring 2025 recordings. Local HTML edition; no hosted site or PDF is included.

Open [index.html](index.html) in a browser. The book includes 18 chapters, primary research links within the explanations, timestamped lecture navigation, mathematical derivations, original numerical examples, and six worked exercises per chapter. Bundled MathJax and reader assets support offline reading.

The [learning route](index.html#learning-route) explains prerequisites and chapter dependencies. The [source map](source-map.html) separates lecture material, research extensions, and original teaching constructions. The [interactive numerical laboratory](numerical-lab.html) explores softmax temperature, rare-event precision, and numerical integration.

## Reproduce numerical checks

Install the NumPy version range in requirements.txt, then run:

    python numerical_lab.py
    python verify_examples.py

The second command executes the actual Python block embedded in each of the 18 chapters, checks source intervals and mathematical delimiters, and updates numerical-validation.json. These are mechanism checks and finite differences, not reproductions of large-model benchmarks.

Optional interactive browser checks require Playwright and an available Chrome installation:

    python lab_browser_checks.py

## Evidence and limits

- [source-manifest.json](source-manifest.json): exact playlist order, recording IDs, source provenance, and source hashes.
- [editorial-audit.json](editorial-audit.json): the nine writing and review passes.
- [numerical-validation.json](numerical-validation.json): all 18 executed chapter examples and cumulative numerical checks.
- [reader-validation.json](reader-validation.json): desktop/mobile mathematics, navigation, controls, local links, and offline browser checks.
- [lab-browser-checks.json](lab-browser-checks.json): actual interactive controls and known numerical results.
- [completion.json](completion.json): course completion gates and boundaries.

All 18 machine transcripts were read in full, together with available official slide text. Transcripts are locally generated speech recognition, not human-edited captions. Some slides are primarily figures; reading their extracted text does not establish complete visual review. Lecture 18 has no official slide deck in the source manifest. Important names, equations, and research claims were checked against primary sources.

The factory-inspection case and numerical values are original teaching constructions. Guest chapters use additional visual-grounding, towel-folding, and hospital-monitoring cases. Source-reuse percentages are not claimed. Raw recordings, transcripts, downloaded slides, and private acquisition files are outside this repository.

This is an independent study resource, not an official Stanford textbook or a claim of institutional endorsement. External recordings and papers remain subject to their own terms. The bundled MathJax license is in assets/mathjax-LICENSE.

![Reader preview](assets/preview.png)
