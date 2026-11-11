# Reproducing an experiment

Keep the model file unchanged and record its SHA-256, llama.cpp build/version, hardware, context, GPU-layer setting, and preset. Use the exported JSON as the complete fixture and result record. The runner uses deterministic decoding settings, one server slot, prompt cache disabled in the request, and a new server for each K/V format. Numerical execution is not guaranteed bitwise identical across devices or builds. Failed initialization and overflow are excluded from accuracy denominators.
